namespace Kestrel.Engine;

/// <summary>Model hyper-parameters, read from the .gguf metadata (kestrel.*).
/// Mirrors kestrel/config.py.</summary>
public sealed class KestrelConfig
{
    public int VocabSize, DModel, NEntry, NCore, NExit, AttnEvery, NHeads, NKvHeads;
    public int ConvKernel, DFf, RMax, RDefault, PkmNKeys, PkmDKey, PkmTopK;
    public float RopeTheta, RopeScale;
    public int ContextLength;
    public int[] PkmSites = Array.Empty<int>();
    public long TokensSeen;
    public string Name = "Kestrel", ChatTemplate = "User: {prompt}\n\nAssistant:";

    public int HeadDim => DModel / NHeads;
    public int NBlocks => NEntry + NCore + NExit;
    public bool IsAttention(int index) => (index + 1) % AttnEvery == 0;
    public bool HasPkm(int index) => Array.IndexOf(PkmSites, index) >= 0;

    public static KestrelConfig FromGguf(GgufFile g)
    {
        if (g.Architecture != "kestrel")
            throw new InvalidDataException(
                $"this is a '{g.Architecture}' model; Kestrel Studio runs architecture 'kestrel'");
        int I(string k) => g.GetInt("kestrel." + k);
        return new KestrelConfig
        {
            VocabSize = I("vocab_size"), DModel = I("d_model"),
            NEntry = I("n_entry"), NCore = I("n_core"), NExit = I("n_exit"),
            AttnEvery = I("attn_every"), NHeads = I("n_heads"), NKvHeads = I("n_kv_heads"),
            ConvKernel = I("conv_kernel"), DFf = I("d_ff"), RMax = I("r_max"), RDefault = I("r_default"),
            PkmNKeys = I("pkm_n_keys"), PkmDKey = I("pkm_d_key"), PkmTopK = I("pkm_topk"),
            PkmSites = g.GetInts("kestrel.pkm_sites"),
            RopeTheta = g.GetFloat("kestrel.rope_theta"),
            RopeScale = g.GetFloat("kestrel.rope_scale", 1f),
            ContextLength = g.GetInt("kestrel.context_length", 1024),
            TokensSeen = g.Has("kestrel.tokens_seen") ? Convert.ToInt64(g.Metadata["kestrel.tokens_seen"]) : 0,
            Name = g.GetString("general.name", "Kestrel"),
            ChatTemplate = g.GetString("kestrel.chat_template", "User: {prompt}\n\nAssistant:"),
        };
    }

    public string Describe() =>
        $"{Name} | d{DModel} | {NEntry}+{NCore}x{RDefault}+{NExit} blocks | " +
        $"ctx {ContextLength} | {TokensSeen / 1e9:F2}B tokens trained";
}
