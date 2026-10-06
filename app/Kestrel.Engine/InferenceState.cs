namespace Kestrel.Engine;

/// <summary>Streaming state of one block instance. GLA blocks carry an (H, D, D) recurrent
/// matrix plus the last k-1 conv inputs; attention blocks carry a KV cache.</summary>
public sealed class BlockState
{
    public float[]? S;       // GLA: heads * hd * hd
    public float[]? Conv;    // GLA: (k-1) * d, oldest row first
    public float[]? K, V;    // attention: capacity * kvHeads * hd
    public int Len;          // attention: cached positions

    internal BlockState CloneShallowKv() => new()
    {
        S = (float[]?)S?.Clone(), Conv = (float[]?)Conv?.Clone(), K = K, V = V, Len = Len,
    };
}

/// <summary>Everything the model has absorbed so far in one conversation. Because GLA
/// state is O(1) per block and the attention KV cache is bounded by the context, the
/// whole thing is a few MB and can be saved to disk and restored exactly — a session
/// resumes where it left off, with no re-reading of the transcript.</summary>
public sealed class InferenceState
{
    public const int FormatVersion = 1;

    public int Loops { get; }
    public int Position { get; internal set; }
    public List<int> Tokens { get; } = new();

    internal readonly BlockState[] Entry, Exit;
    internal readonly BlockState[,] Core;    // [loop, core block]

    internal InferenceState(KestrelConfig c, int loops)
    {
        if (loops < 1 || loops > c.RMax)
            throw new ArgumentOutOfRangeException(nameof(loops), $"loops must be 1..{c.RMax}");
        Loops = loops;
        Entry = Make(c, c.NEntry, 0);
        Exit = Make(c, c.NExit, c.NEntry + c.NCore);
        Core = new BlockState[loops, c.NCore];
        for (int r = 0; r < loops; r++)
            for (int i = 0; i < c.NCore; i++) Core[r, i] = New(c, c.NEntry + i);
    }

    InferenceState(InferenceState src)
    {
        Loops = src.Loops;
        Position = src.Position;
        Tokens.AddRange(src.Tokens);
        Entry = src.Entry.Select(b => b.CloneShallowKv()).ToArray();
        Exit = src.Exit.Select(b => b.CloneShallowKv()).ToArray();
        Core = new BlockState[src.Core.GetLength(0), src.Core.GetLength(1)];
        for (int r = 0; r < Core.GetLength(0); r++)
            for (int i = 0; i < Core.GetLength(1); i++) Core[r, i] = src.Core[r, i].CloneShallowKv();
    }

    /// <summary>A cheap rewind point (copies GLA state, shares the append-only KV arrays).
    /// Valid to <see cref="Restore"/> into the state it was taken from.</summary>
    public InferenceState Snapshot() => new(this);

    public void Restore(InferenceState snap)
    {
        if (snap.Loops != Loops) throw new InvalidOperationException("snapshot has a different loop count");
        Position = snap.Position;
        Tokens.Clear(); Tokens.AddRange(snap.Tokens);
        for (int i = 0; i < Entry.Length; i++) Copy(snap.Entry[i], Entry[i]);
        for (int i = 0; i < Exit.Length; i++) Copy(snap.Exit[i], Exit[i]);
        for (int r = 0; r < Loops; r++)
            for (int i = 0; i < Core.GetLength(1); i++) Copy(snap.Core[r, i], Core[r, i]);

        static void Copy(BlockState from, BlockState to)
        {
            from.S?.CopyTo(to.S!, 0);
            from.Conv?.CopyTo(to.Conv!, 0);
            to.Len = from.Len;
        }
    }

    internal IEnumerable<BlockState> All()
    {
        foreach (var b in Entry) yield return b;
        foreach (var b in Core) yield return b;
        foreach (var b in Exit) yield return b;
    }

    static BlockState[] Make(KestrelConfig c, int n, int first) =>
        Enumerable.Range(0, n).Select(i => New(c, first + i)).ToArray();

    static BlockState New(KestrelConfig c, int index)
    {
        if (c.IsAttention(index))
        {
            int kv = c.ContextLength * c.NKvHeads * c.HeadDim;
            return new BlockState { K = new float[kv], V = new float[kv] };
        }
        return new BlockState
        {
            S = new float[c.NHeads * c.HeadDim * c.HeadDim],
            Conv = new float[(c.ConvKernel - 1) * c.DModel],
        };
    }

    // ------------------------------------------------------------------ persistence

    static int[] Fingerprint(KestrelConfig c) => new[]
    {
        c.VocabSize, c.DModel, c.NHeads, c.NKvHeads, c.HeadDim, c.NEntry, c.NCore, c.NExit,
        c.ConvKernel, c.ContextLength, c.AttnEvery,
    };

    public void Save(Stream s, KestrelConfig c)
    {
        using var w = new BinaryWriter(s, System.Text.Encoding.UTF8, leaveOpen: true);
        w.Write("KSES"u8);
        w.Write(FormatVersion);
        foreach (int f in Fingerprint(c)) w.Write(f);
        w.Write(Loops);
        w.Write(Position);
        w.Write(Tokens.Count);
        foreach (int t in Tokens) w.Write(t);
        int kvRow = c.NKvHeads * c.HeadDim;
        foreach (var b in All())
        {
            if (b.S != null) { WriteFloats(w, b.S); WriteFloats(w, b.Conv!); }
            else
            {
                w.Write(b.Len);
                WriteFloats(w, b.K!.AsSpan(0, b.Len * kvRow));
                WriteFloats(w, b.V!.AsSpan(0, b.Len * kvRow));
            }
        }
    }

    public static InferenceState Load(Stream s, KestrelConfig c)
    {
        using var r = new BinaryReader(s, System.Text.Encoding.UTF8, leaveOpen: true);
        if (!r.ReadBytes(4).AsSpan().SequenceEqual("KSES"u8))
            throw new InvalidDataException("not a Kestrel session file");
        int ver = r.ReadInt32();
        if (ver != FormatVersion) throw new InvalidDataException($"session format v{ver} not supported");
        foreach (int f in Fingerprint(c))
            if (r.ReadInt32() != f)
                throw new InvalidDataException("this session was saved by a different model");
        var st = new InferenceState(c, r.ReadInt32()) { Position = r.ReadInt32() };
        int n = r.ReadInt32();
        for (int i = 0; i < n; i++) st.Tokens.Add(r.ReadInt32());
        int kvRow = c.NKvHeads * c.HeadDim;
        foreach (var b in st.All())
        {
            if (b.S != null) { ReadFloats(r, b.S); ReadFloats(r, b.Conv!); }
            else
            {
                b.Len = r.ReadInt32();
                ReadFloats(r, b.K.AsSpan(0, b.Len * kvRow));
                ReadFloats(r, b.V.AsSpan(0, b.Len * kvRow));
            }
        }
        return st;
    }

    static void WriteFloats(BinaryWriter w, ReadOnlySpan<float> a) =>
        w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a));

    static void ReadFloats(BinaryReader r, Span<float> a) =>
        r.BaseStream.ReadExactly(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a));
}
