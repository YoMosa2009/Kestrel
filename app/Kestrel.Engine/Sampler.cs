using System.Numerics.Tensors;

namespace Kestrel.Engine;

public sealed record SamplingOptions
{
    public float Temperature { get; init; } = 0.7f;
    public int TopK { get; init; } = 40;
    public float TopP { get; init; } = 0.95f;
    /// <summary>Divides positive logits (multiplies negative ones) of recently seen tokens.
    /// 1.0 = off. A small model benefits from ~1.1 against loops.</summary>
    public float RepetitionPenalty { get; init; } = 1.1f;
    public int RepetitionWindow { get; init; } = 64;
    public int? Seed { get; init; }
}

/// <summary>Temperature / top-k / top-p sampling, as in kestrel/generate.py, plus an
/// optional repetition penalty.</summary>
public sealed class Sampler
{
    readonly Random _rng;
    public SamplingOptions Options { get; set; }

    public Sampler(SamplingOptions options)
    {
        Options = options;
        _rng = options.Seed is int s ? new Random(s) : new Random();
    }

    public int Sample(float[] logits, IReadOnlyList<int> history)
    {
        var o = Options;
        var l = (float[])logits.Clone();
        if (o.RepetitionPenalty != 1f)
        {
            var seen = new HashSet<int>();
            for (int i = Math.Max(0, history.Count - o.RepetitionWindow); i < history.Count; i++)
                seen.Add(history[i]);
            foreach (int id in seen)
                l[id] = l[id] > 0 ? l[id] / o.RepetitionPenalty : l[id] * o.RepetitionPenalty;
        }
        if (o.Temperature <= 1e-4f) return TensorPrimitives.IndexOfMax(l);

        int k = o.TopK > 0 ? Math.Min(o.TopK, l.Length) : l.Length;
        var idx = new int[k];
        var val = new float[k];
        Ops.TopK(l, k, idx, val);
        for (int i = 0; i < k; i++) val[i] /= o.Temperature;
        Ops.SoftmaxInPlace(val);                          // sorted descending

        int keep = k;
        if (o.TopP > 0 && o.TopP < 1)
        {
            float cum = 0;
            for (int i = 0; i < k; i++)
            {
                if (cum >= o.TopP) { keep = i; break; }   // drop once the mass before it reached p
                cum += val[i];
            }
        }
        float total = 0;
        for (int i = 0; i < keep; i++) total += val[i];
        double r = _rng.NextDouble() * total;
        for (int i = 0; i < keep; i++)
        {
            r -= val[i];
            if (r <= 0) return idx[i];
        }
        return idx[keep - 1];
    }
}
