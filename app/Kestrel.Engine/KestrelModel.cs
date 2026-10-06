using System.Numerics.Tensors;

namespace Kestrel.Engine;

/// <summary>One product-key-memory lookup: which of the site's slots a token read.</summary>
public readonly record struct PkmHit(int Site, int Token, int[] Slots, float[] Weights);

/// <summary>Native Kestrel forward pass — a line-for-line port of kestrel/model.py,
/// in recurrent (token-at-a-time) form:
/// <list type="bullet">
/// <item>GLA blocks: S_t = g_t·S_{t-1} + k_tᵀv_t, o_t = q_t·S_t (exactly the chunked scan, unchunked).</item>
/// <item>Attention blocks (every 4th): GQA + QK-RMSNorm + RoPE with position interpolation, KV-cached.</item>
/// <item>Looped core: the core blocks run R times, each pass with its own state.</item>
/// <item>Product-key memory at the configured sites.</item>
/// </list>
/// Prompts are processed T tokens at a time (weights reused across tokens), generation 1 at a time.</summary>
public sealed class KestrelModel
{
    public KestrelConfig Config { get; }
    public BpeTokenizer Tokenizer { get; }
    public string Path { get; }
    public long ParameterCount { get; }

    readonly float[] _embed, _normF, _ropeInv;
    readonly Block[] _blocks;

    sealed class Lin
    {
        public float[] W = null!; public float[]? B; public int Out, In;
        public void Apply(float[] x, float[] y, int t) => Ops.Linear(W, B, Out, In, x, 0, y, 0, t);
    }

    sealed class Pkm
    {
        public int Site;
        public Lin Wq = null!;
        public float[] LnW = null!, LnB = null!, Keys = null!, Values = null!;
        public float Gate;     // tanh(gate), precomputed
    }

    sealed class Block
    {
        public int Index; public bool IsAttn;
        public float[] Norm1 = null!, Norm2 = null!; public float[]? Norm3;
        // GLA
        public float[] ConvW = null!, OutNorm = null!, QScale = null!;
        public Lin Wqkv = null!, Wg = null!, Wog = null!;
        // attention
        public Lin Wq = null!, Wk = null!, Wv = null!;
        public Lin Wo = null!;
        // MLP
        public Lin Up = null!, Down = null!;
        public Pkm? Pkm;
    }

    KestrelModel(string path, IProgress<double>? progress)
    {
        Path = path;
        var g = GgufFile.Open(path);
        Config = KestrelConfig.FromGguf(g);
        Tokenizer = BpeTokenizer.FromGguf(g);
        var c = Config;
        long nParams = 0;
        int total = c.NBlocks + 2, done = 0;

        float[] T(string name) { var a = g.ReadFloats(name); nParams += a.Length; return a; }
        Lin L(string name, int o, int i, bool bias = false) =>
            new() { W = T(name + ".weight"), B = bias ? T(name + ".bias") : null, Out = o, In = i };

        _embed = T("embed.weight");
        _normF = T("norm_f.weight");
        progress?.Report(++done / (double)total);

        int d = c.DModel, hd = c.HeadDim;
        _blocks = new Block[c.NBlocks];
        for (int idx = 0; idx < c.NBlocks; idx++)
        {
            string p = idx < c.NEntry ? $"entry.{idx}"
                     : idx < c.NEntry + c.NCore ? $"core.{idx - c.NEntry}"
                     : $"exit.{idx - c.NEntry - c.NCore}";
            var b = new Block { Index = idx, IsAttn = c.IsAttention(idx) };
            b.Norm1 = T($"{p}.norm1.weight");
            b.Norm2 = T($"{p}.norm2.weight");
            if (b.IsAttn)
            {
                b.Wq = L($"{p}.mixer.wq", c.NHeads * hd, d);
                b.Wk = L($"{p}.mixer.wk", c.NKvHeads * hd, d);
                b.Wv = L($"{p}.mixer.wv", c.NKvHeads * hd, d);
                b.Wo = L($"{p}.mixer.wo", d, c.NHeads * hd);
            }
            else
            {
                b.ConvW = T($"{p}.mixer.conv.weight");             // (d, k)
                b.Wqkv = L($"{p}.mixer.wqkv", 3 * d, d);
                b.Wg = L($"{p}.mixer.wg", c.NHeads, d, bias: true);
                b.Wog = L($"{p}.mixer.wog", d, d);
                b.Wo = L($"{p}.mixer.wo", d, d);
                b.OutNorm = T($"{p}.mixer.out_norm.weight");
                b.QScale = T($"{p}.mixer.q_scale");
            }
            b.Up = L($"{p}.mlp.w_up", 2 * c.DFf, d);
            b.Down = L($"{p}.mlp.w_down", d, c.DFf);
            if (c.HasPkm(idx))
            {
                b.Norm3 = T($"{p}.norm3.weight");
                b.Pkm = new Pkm
                {
                    Site = Array.IndexOf(c.PkmSites, idx),
                    Wq = L($"{p}.pkm.wq", c.PkmDKey, d),
                    LnW = T($"{p}.pkm.q_norm.weight"),
                    LnB = T($"{p}.pkm.q_norm.bias"),
                    Keys = T($"{p}.pkm.keys"),                     // (2, nKeys, dKey/2)
                    Values = T($"{p}.pkm.values.weight"),          // (nKeys², d)
                    Gate = MathF.Tanh(T($"{p}.pkm.gate")[0]),
                };
            }
            _blocks[idx] = b;
            progress?.Report(++done / (double)total);
        }

        _ropeInv = new float[hd / 2];
        for (int i = 0; i < hd / 2; i++)
            _ropeInv[i] = (float)(1.0 / Math.Pow(c.RopeTheta, 2.0 * i / hd));
        ParameterCount = nParams;
        progress?.Report(1.0);
    }

    public static KestrelModel Load(string path, IProgress<double>? progress = null) => new(path, progress);

    public InferenceState NewState(int? loops = null) => new(Config, loops ?? Config.RDefault);

    /// <summary>Called for every PKM lookup during a forward pass (null = off).</summary>
    public Action<PkmHit>? OnPkm { get; set; }

    // ------------------------------------------------------------------ forward

    /// <summary>Feed <paramref name="tokens"/> through the model, advancing <paramref name="st"/>.
    /// Returns logits for the last token (vocab), or for every token when
    /// <paramref name="allLogits"/> (T × vocab). <paramref name="meanHidden"/>, if given,
    /// receives the mean of the final-norm hidden states (the Roost embedding).</summary>
    public float[] Forward(InferenceState st, ReadOnlySpan<int> tokens, bool allLogits = false,
                           float[]? meanHidden = null)
    {
        var c = Config;
        int t = tokens.Length, d = c.DModel;
        if (t == 0) throw new ArgumentException("no tokens");
        if (st.Position + t > c.ContextLength)
            throw new ContextFullException(st.Position, t, c.ContextLength);

        var x = new float[t * d];
        for (int i = 0; i < t; i++)
            Array.Copy(_embed, (long)tokens[i] * d, x, (long)i * d, d);

        for (int i = 0; i < c.NEntry; i++) RunBlock(_blocks[i], st.Entry[i], x, t, st.Position);
        for (int r = 0; r < st.Loops; r++)
            for (int i = 0; i < c.NCore; i++)
                RunBlock(_blocks[c.NEntry + i], st.Core[r, i], x, t, st.Position);
        for (int i = 0; i < c.NExit; i++)
            RunBlock(_blocks[c.NEntry + c.NCore + i], st.Exit[i], x, t, st.Position);

        st.Position += t;
        st.Tokens.AddRange(tokens.ToArray());

        int first = allLogits || meanHidden != null ? 0 : t - 1;
        int rows = t - first;
        var h = new float[rows * d];
        for (int i = 0; i < rows; i++)
            Ops.RmsNorm(x.AsSpan((first + i) * d, d), _normF, h.AsSpan(i * d, d));

        if (meanHidden != null)
        {
            Array.Clear(meanHidden);
            for (int i = 0; i < rows; i++) TensorPrimitives.Add(meanHidden, h.AsSpan(i * d, d), meanHidden);
            TensorPrimitives.Multiply(meanHidden, 1f / rows, meanHidden);
        }

        int outRows = allLogits ? rows : 1;
        int hOff = allLogits ? 0 : (rows - 1) * d;
        var logits = new float[outRows * c.VocabSize];
        Ops.Linear(_embed, null, c.VocabSize, d, h, hOff, logits, 0, outRows);   // tied LM head
        return logits;
    }

    void RunBlock(Block b, BlockState bs, float[] x, int t, int pos0)
    {
        int d = Config.DModel;
        var h = new float[t * d];
        for (int i = 0; i < t; i++) Ops.RmsNorm(x.AsSpan(i * d, d), b.Norm1, h.AsSpan(i * d, d));
        var y = b.IsAttn ? Attention(b, bs, h, t) : Gla(b, bs, h, t);
        TensorPrimitives.Add(x, y, x);

        for (int i = 0; i < t; i++) Ops.RmsNorm(x.AsSpan(i * d, d), b.Norm2, h.AsSpan(i * d, d));
        TensorPrimitives.Add(x, Mlp(b, h, t), x);

        if (b.Pkm != null)
        {
            for (int i = 0; i < t; i++) Ops.RmsNorm(x.AsSpan(i * d, d), b.Norm3, h.AsSpan(i * d, d));
            TensorPrimitives.Add(x, ProductKeyMemory(b.Pkm, h, t, pos0), x);
        }
    }

    float[] Mlp(Block b, float[] h, int t)
    {
        int ff = Config.DFf;
        var up = new float[t * 2 * ff];
        b.Up.Apply(h, up, t);
        var act = new float[t * ff];
        for (int i = 0; i < t; i++)
            for (int j = 0; j < ff; j++)
                act[i * ff + j] = up[i * 2 * ff + j] * Ops.Silu(up[i * 2 * ff + ff + j]);
        var y = new float[t * Config.DModel];
        b.Down.Apply(act, y, t);
        return y;
    }

    float[] Gla(Block b, BlockState bs, float[] h, int t)
    {
        var c = Config;
        int d = c.DModel, k = c.ConvKernel, nh = c.NHeads, hd = c.HeadDim;

        // causal depthwise conv over [cached k-1 inputs, this segment], then SiLU
        var pad = new float[(k - 1 + t) * d];
        Array.Copy(bs.Conv!, pad, (k - 1) * d);
        Array.Copy(h, 0, pad, (k - 1) * d, t * d);
        var xc = new float[t * d];
        for (int i = 0; i < t; i++)
            for (int ch = 0; ch < d; ch++)
            {
                float s = 0f;
                for (int j = 0; j < k; j++) s += b.ConvW[ch * k + j] * pad[(i + j) * d + ch];
                xc[i * d + ch] = Ops.Silu(s);
            }
        Array.Copy(pad, t * d, bs.Conv!, 0, (k - 1) * d);

        var qkv = new float[t * 3 * d];
        b.Wqkv.Apply(xc, qkv, t);
        var gl = new float[t * nh];
        b.Wg.Apply(xc, gl, t);
        var og = new float[t * d];
        b.Wog.Apply(xc, og, t);

        var o = new float[t * d];
        var S = bs.S!;
        Parallel.For(0, nh, head =>
        {
            var q = new float[hd]; var kk = new float[hd];
            var sh = S.AsSpan(head * hd * hd, hd * hd);
            for (int i = 0; i < t; i++)
            {
                int bq = i * 3 * d + head * hd;
                qkv.AsSpan(bq, hd).CopyTo(q);
                qkv.AsSpan(bq + d, hd).CopyTo(kk);
                var v = new ReadOnlySpan<float>(qkv, bq + 2 * d, hd);
                Ops.L2Normalize(q);
                TensorPrimitives.Multiply(q, b.QScale[head], q);
                Ops.L2Normalize(kk);

                double logG = -Ops.Softplus(gl[i * nh + head]);
                float g = (float)Math.Exp(logG);
                TensorPrimitives.Multiply(kk, (float)(1.0 - Math.Exp(logG)), kk);   // k <- (1-g)k

                var oh = o.AsSpan(i * d + head * hd, hd);
                for (int r = 0; r < hd; r++)
                {
                    var row = sh.Slice(r * hd, hd);
                    TensorPrimitives.Multiply(row, g, row);                 // S <- g S
                    TensorPrimitives.MultiplyAdd(v, kk[r], row, row);       //      + k^T v
                    TensorPrimitives.MultiplyAdd(row, q[r], oh, oh);        // o = q S
                }
                Ops.RmsNorm(oh, b.OutNorm, oh);
            }
        });

        for (int i = 0; i < t * d; i++) o[i] *= Ops.Silu(og[i]);
        var y = new float[t * d];
        b.Wo.Apply(o, y, t);
        return y;
    }

    float[] Attention(Block b, BlockState bs, float[] h, int t)
    {
        var c = Config;
        int nh = c.NHeads, kvh = c.NKvHeads, hd = c.HeadDim, rep = nh / kvh;
        var q = new float[t * nh * hd];
        var k = new float[t * kvh * hd];
        var v = new float[t * kvh * hd];
        b.Wq.Apply(h, q, t);
        b.Wk.Apply(h, k, t);
        b.Wv.Apply(h, v, t);

        int pos0 = bs.Len;
        for (int i = 0; i < t; i++)
        {
            float pos = (pos0 + i) / c.RopeScale;
            for (int hh = 0; hh < nh; hh++) NormRope(q.AsSpan((i * nh + hh) * hd, hd), pos);
            for (int hh = 0; hh < kvh; hh++) NormRope(k.AsSpan((i * kvh + hh) * hd, hd), pos);
        }
        Array.Copy(k, 0, bs.K!, pos0 * kvh * hd, t * kvh * hd);
        Array.Copy(v, 0, bs.V!, pos0 * kvh * hd, t * kvh * hd);
        bs.Len += t;

        var o = new float[t * nh * hd];
        float scale = 1f / MathF.Sqrt(hd);
        var K = bs.K!; var V = bs.V!;
        Parallel.For(0, nh * t, job =>
        {
            int hh = job % nh, i = job / nh, kh = hh / rep, n = pos0 + i + 1;
            var qh = new ReadOnlySpan<float>(q, (i * nh + hh) * hd, hd);
            var p = new float[n];
            for (int j = 0; j < n; j++)
                p[j] = TensorPrimitives.Dot(qh, new ReadOnlySpan<float>(K, (j * kvh + kh) * hd, hd)) * scale;
            Ops.SoftmaxInPlace(p);
            var oh = o.AsSpan((i * nh + hh) * hd, hd);
            for (int j = 0; j < n; j++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(V, (j * kvh + kh) * hd, hd), p[j], oh, oh);
        });

        var y = new float[t * c.DModel];
        b.Wo.Apply(o, y, t);
        return y;
    }

    /// <summary>Parameter-free RMS norm over the head dim, then interleaved-pair RoPE.</summary>
    void NormRope(Span<float> x, float pos)
    {
        Ops.RmsNorm(x, ReadOnlySpan<float>.Empty, x);
        for (int i = 0; i < x.Length / 2; i++)
        {
            float f = pos * _ropeInv[i], cs = MathF.Cos(f), sn = MathF.Sin(f);
            float a = x[2 * i], b = x[2 * i + 1];
            x[2 * i] = a * cs - b * sn;
            x[2 * i + 1] = a * sn + b * cs;
        }
    }

    float[] ProductKeyMemory(Pkm m, float[] h, int t, int pos0)
    {
        var c = Config;
        int d = c.DModel, dk = c.PkmDKey, half = dk / 2, nk = c.PkmNKeys;
        int topk = c.PkmTopK, kh = Math.Min(topk, nk);
        var q = new float[t * dk];
        m.Wq.Apply(h, q, t);
        var y = new float[t * d];
        var onPkm = OnPkm;

        Parallel.For(0, t, i =>
        {
            var qi = q.AsSpan(i * dk, dk);
            Ops.LayerNorm(qi, m.LnW, m.LnB);
            var s1 = new float[nk]; var s2 = new float[nk];
            for (int n = 0; n < nk; n++)
            {
                s1[n] = TensorPrimitives.Dot(qi[..half], new ReadOnlySpan<float>(m.Keys, n * half, half));
                s2[n] = TensorPrimitives.Dot(qi[half..], new ReadOnlySpan<float>(m.Keys, (nk + n) * half, half));
            }
            Span<int> i1 = stackalloc int[kh], i2 = stackalloc int[kh];
            Span<float> v1 = stackalloc float[kh], v2 = stackalloc float[kh];
            Ops.TopK(s1, kh, i1, v1);
            Ops.TopK(s2, kh, i2, v2);
            Span<float> cand = stackalloc float[kh * kh];
            for (int a = 0; a < kh; a++)
                for (int bb = 0; bb < kh; bb++) cand[a * kh + bb] = v1[a] + v2[bb];
            Span<int> flat = stackalloc int[topk];
            var score = new float[topk];
            Ops.TopK(cand, topk, flat, score);
            var slots = new int[topk];
            for (int j = 0; j < topk; j++) slots[j] = i1[flat[j] / kh] * nk + i2[flat[j] % kh];
            Ops.SoftmaxInPlace(score);
            var yi = y.AsSpan(i * d, d);
            for (int j = 0; j < topk; j++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(m.Values, slots[j] * d, d),
                                             score[j] * m.Gate, yi, yi);
            onPkm?.Invoke(new PkmHit(m.Site, pos0 + i, slots, score));
        });
        return y;
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>Roost embedding: mean-pooled final hidden state from a fresh state at the
    /// default loop count, L2-normalized — identical to kestrel/roost/store.py.</summary>
    public float[] Embed(string text, int maxLen = 256)
    {
        var ids = Tokenizer.Encode(text);
        if (ids.Count > maxLen) ids = ids.GetRange(0, maxLen);
        if (ids.Count == 0) ids.Add(0);
        var v = new float[Config.DModel];
        var saved = OnPkm; OnPkm = null;
        try { Forward(NewState(), ids.ToArray(), meanHidden: v); }
        finally { OnPkm = saved; }
        TensorPrimitives.Multiply(v, 1f / (TensorPrimitives.Norm(v) + 1e-8f), v);
        return v;
    }
}

public sealed class ContextFullException(int position, int adding, int limit)
    : Exception($"context full: {position} + {adding} tokens exceeds {limit}")
{
    public int Position { get; } = position;
    public int Limit { get; } = limit;
}
