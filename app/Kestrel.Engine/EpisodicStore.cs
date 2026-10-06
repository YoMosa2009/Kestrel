using System.Numerics.Tensors;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Kestrel.Engine;

/// <summary>One remembered fact. Field names match kestrel/roost/store.py so the same
/// episodic.jsonl works in Python (consolidation on Colab) and in the app.</summary>
public sealed class Teachable
{
    [JsonPropertyName("text")] public string Text { get; set; } = "";
    [JsonPropertyName("variants")] public List<string> Variants { get; set; } = new();
    [JsonPropertyName("kind")] public string Kind { get; set; } = "fact";
    [JsonPropertyName("source")] public string Source { get; set; } = "user";
    [JsonPropertyName("ts")] public double Ts { get; set; }
    [JsonPropertyName("uses")] public int Uses { get; set; }
    [JsonPropertyName("last_used")] public double LastUsed { get; set; }

    [JsonIgnore] internal float[]? Embedding;
}

/// <summary>Roost episodic memory: an append-only JSONL of things the user taught,
/// retrieved by cosine similarity in the model's OWN representation space (mean-pooled
/// final hidden states — no separate embedding model).</summary>
public sealed class EpisodicStore
{
    static readonly JsonSerializerOptions Json = new() { WriteIndented = false };

    readonly List<Teachable> _items = new();
    readonly object _lock = new();
    public string Path { get; }
    public IReadOnlyList<Teachable> Items { get { lock (_lock) return _items.ToList(); } }

    public EpisodicStore(string path)
    {
        Path = path;
        if (!File.Exists(path)) return;
        foreach (var line in File.ReadLines(path))
            if (!string.IsNullOrWhiteSpace(line))
                _items.Add(JsonSerializer.Deserialize<Teachable>(line, Json)!);
    }

    static double Now => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;

    public Teachable Add(string text)
    {
        text = text.Trim();
        var t = new Teachable { Text = text, Ts = Now };
        lock (_lock)
        {
            var dup = _items.FirstOrDefault(i => string.Equals(i.Text, text, StringComparison.OrdinalIgnoreCase));
            if (dup != null) return dup;
            _items.Add(t);
        }
        Flush();
        return t;
    }

    public void Remove(Teachable t)
    {
        lock (_lock) _items.Remove(t);
        Flush();
    }

    public void Flush()
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(Path))!);
        var tmp = Path + ".tmp";
        lock (_lock)
            File.WriteAllLines(tmp, _items.Select(i => JsonSerializer.Serialize(i, Json)));
        File.Move(tmp, Path, overwrite: true);
    }

    /// <summary>Top-k facts by cosine similarity to <paramref name="query"/>, with a minimum
    /// similarity so unrelated chit-chat does not drag irrelevant facts into the prompt.</summary>
    public List<(float Score, Teachable Item)> Retrieve(KestrelModel model, string query, int k = 4,
                                                        float minScore = 0.0f)
    {
        List<Teachable> items;
        lock (_lock) items = _items.ToList();
        if (items.Count == 0) return new();
        foreach (var it in items) it.Embedding ??= model.Embed(it.Text);
        var q = model.Embed(query);
        var hits = items.Select(it => (Score: TensorPrimitives.Dot(it.Embedding!, q), Item: it))
                        .Where(h => h.Score >= minScore)
                        .OrderByDescending(h => h.Score).Take(k).ToList();
        foreach (var h in hits) { h.Item.Uses++; h.Item.LastUsed = Now; }
        if (hits.Count > 0) Flush();
        return hits;
    }

    public static string AsContext(IEnumerable<Teachable> facts)
    {
        var lines = facts.Select(f => "- " + f.Text).ToList();
        return lines.Count == 0 ? "" : "Known facts:\n" + string.Join("\n", lines) + "\n\n";
    }
}
