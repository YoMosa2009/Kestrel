using System.Diagnostics;
using System.Numerics.Tensors;
using System.Text.Json;
using Kestrel.Engine;

// kestrel info   <model.gguf>
// kestrel parity <model.gguf> <fixture.json>     exit code 0 = engine matches PyTorch
// kestrel bench  <model.gguf> [loops]
// kestrel chat   <model.gguf> [loops]

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: kestrel info|parity|bench|chat <model.gguf> [fixture.json|loops]");
    return 2;
}

if (args[0] == "tokcheck")             // kestrel tokcheck <tokenizer.json> <cases.json>
    return TokCheck(args[1], args[2]);

var sw = Stopwatch.StartNew();
var model = KestrelModel.Load(args[1]);
Console.WriteLine($"{model.Config.Describe()} | {model.ParameterCount / 1e6:F1}M params | " +
                  $"loaded in {sw.Elapsed.TotalSeconds:F1}s");

switch (args[0])
{
    case "info":
        return 0;
    case "parity":
        return Parity(model, args[2]);
    case "bench":
        Bench(model, args.Length > 2 ? int.Parse(args[2]) : model.Config.RDefault);
        return 0;
    case "chat":
        Chat(model, args.Length > 2 ? int.Parse(args[2]) : model.Config.RDefault);
        return 0;
    default:
        Console.Error.WriteLine($"unknown command {args[0]}");
        return 2;
}

static int Parity(KestrelModel model, string fixturePath)
{
    using var doc = JsonDocument.Parse(File.ReadAllText(fixturePath));
    bool ok = true;
    // The fixture is float32 PyTorch. An exact (f32) export must match to ~1e-4; f16 rounding
    // moves logits ~1e-2 and 8-bit quantization ~1e-1 - argmax agreement is the real test there.
    var types = GgufFile.Open(model.Path).Tensors.Values.Select(t => t.Type).ToHashSet();
    float tol = types.Contains(GgufFile.Q8_0) ? 0.5f : types.Contains(GgufFile.F16) ? 0.05f : 0.01f;
    Console.WriteLine($"logit tolerance {tol} ({string.Join("/", types.Select(t => t switch
        { GgufFile.F32 => "f32", GgufFile.F16 => "f16", GgufFile.Q8_0 => "q8_0", _ => t.ToString() }))})");
    foreach (var rec in doc.RootElement.GetProperty("records").EnumerateArray())
    {
        string prompt = rec.GetProperty("prompt").GetString()!;
        var want = rec.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var got = model.Tokenizer.Encode(prompt).ToArray();
        bool tokOk = want.SequenceEqual(got);
        Console.WriteLine($"\n[{Short(prompt)}] tokens {got.Length}: {(tokOk ? "match" : "MISMATCH")}");
        if (!tokOk)
        {
            ok = false;
            Console.WriteLine($"  want {string.Join(",", want)}\n  got  {string.Join(",", got)}");
        }

        foreach (var byR in rec.GetProperty("by_r").EnumerateObject())
        {
            int r = int.Parse(byR.Name);
            var st = model.NewState(r);
            var logits = model.Forward(st, want, allLogits: true);
            int V = model.Config.VocabSize, T = want.Length;
            var last = logits.AsSpan((T - 1) * V, V);

            var topIds = byR.Value.GetProperty("top_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var topLog = byR.Value.GetProperty("top_logits").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            float maxDiff = 0;
            for (int i = 0; i < topIds.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(last[topIds[i]] - topLog[i]));
            var argWant = byR.Value.GetProperty("argmax_per_pos").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int agree = 0;
            for (int t = 0; t < T; t++)
                if (TensorPrimitives.IndexOfMax(logits.AsSpan(t * V, V)) == argWant[t]) agree++;
            double agreePct = 100.0 * agree / T;
            bool rOk = maxDiff < tol && agreePct >= 95;
            ok &= rOk;
            Console.WriteLine($"  R={r}: top-32 logit max|diff| {maxDiff:F4} | argmax agree {agree}/{T} " +
                              $"({agreePct:F0}%) {(rOk ? "OK" : "FAIL")}");
        }

        var embWant = rec.GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        float cos = TensorPrimitives.Dot(model.Embed(prompt), embWant);
        bool eOk = cos > 0.999f;
        ok &= eOk;
        Console.WriteLine($"  roost embedding cosine {cos:F5} {(eOk ? "OK" : "FAIL")}");
    }
    Console.WriteLine(ok ? "\nPARITY PASS" : "\nPARITY FAIL");
    return ok ? 0 : 1;

    static string Short(string s) => (s.Length > 40 ? s[..40] + "..." : s).Replace("\n", "\\n");
}

/// <summary>Tokenizer-only parity against Hugging Face ids (no model needed).</summary>
static int TokCheck(string tokJson, string casesPath)
{
    using var tj = JsonDocument.Parse(File.ReadAllText(tokJson));
    var m = tj.RootElement.GetProperty("model");
    var vocab = m.GetProperty("vocab").EnumerateObject().ToDictionary(p => p.Value.GetInt32(), p => p.Name);
    var tokens = Enumerable.Range(0, vocab.Count).Select(i => vocab[i]).ToArray();
    var merges = m.GetProperty("merges").EnumerateArray()
        .Select(e => e.ValueKind == JsonValueKind.String ? e.GetString()!
                     : e[0].GetString() + " " + e[1].GetString()).ToArray();
    var tok = new BpeTokenizer(tokens, merges, eotId: 0, splitDigits: true);

    using var cj = JsonDocument.Parse(File.ReadAllText(casesPath));
    int bad = 0, n = 0;
    foreach (var c in cj.RootElement.EnumerateArray())
    {
        n++;
        string text = c.GetProperty("text").GetString()!;
        var want = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var got = tok.Encode(text).ToArray();
        bool roundTrip = tok.Decode(got) == text;
        if (!want.SequenceEqual(got) || !roundTrip)
        {
            if (++bad <= 5)
                Console.WriteLine($"MISMATCH {JsonSerializer.Serialize(text[..Math.Min(80, text.Length)])}\n" +
                                  $"  want {string.Join(",", want.Take(30))}\n  got  {string.Join(",", got.Take(30))}" +
                                  $"  roundtrip={roundTrip}");
        }
    }
    Console.WriteLine($"tokenizer: {n - bad}/{n} cases identical to Hugging Face");
    return bad == 0 ? 0 : 1;
}

static void Bench(KestrelModel model, int loops)
{
    var ids = model.Tokenizer.Encode(string.Concat(Enumerable.Repeat(
        "The kestrel hovers over the field, scanning the grass for movement. ", 30)));
    ids = ids.GetRange(0, Math.Min(512, ids.Count));
    var st = model.NewState(loops);
    var sw = Stopwatch.StartNew();
    var logits = model.Forward(st, ids.ToArray());
    double pre = sw.Elapsed.TotalSeconds;
    sw.Restart();
    int n = 64;
    for (int i = 0; i < n; i++)
        logits = model.Forward(st, new[] { TensorPrimitives.IndexOfMax(logits) });
    double gen = sw.Elapsed.TotalSeconds;
    Console.WriteLine($"R={loops} | prefill {ids.Count} tok: {ids.Count / pre:F0} tok/s | " +
                      $"decode {n} tok: {n / gen:F1} tok/s | threads {Environment.ProcessorCount}");
}

static void Chat(KestrelModel model, int loops)
{
    var chat = new ChatSession(model, loops);
    var sampler = new Sampler(new SamplingOptions());
    Console.WriteLine("Chat with Kestrel. Empty line quits.\n");
    while (true)
    {
        Console.Write("you> ");
        var line = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(line)) break;
        Console.Write("kestrel> ");
        var stats = chat.Send(line, sampler, 300, Console.Write);
        Console.WriteLine($"\n  [{stats.NewTokens} tok, {stats.GenTokPerSec:F1} tok/s, stop={stats.StopReason}]\n");
    }
}
