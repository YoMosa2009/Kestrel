using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace Kestrel.Engine;

public sealed record ChatTurn(string Role, string Content)
{
    /// <summary>Roost facts that were injected ahead of this (user) turn, for display.</summary>
    public List<string> Recalled { get; init; } = new();
}

public sealed record GenerationStats(int PromptTokens, double PromptSeconds, int NewTokens,
                                     double GenSeconds, string StopReason)
{
    public double PromptTokPerSec => PromptTokens / Math.Max(PromptSeconds, 1e-9);
    public double GenTokPerSec => NewTokens / Math.Max(GenSeconds, 1e-9);
}

/// <summary>A conversation with a Kestrel model, kept as a live recurrent state so each
/// new turn only feeds the new tokens. Uses the exact SFT template:
/// <c>Role: content</c> turns joined by blank lines, header and content tokenized
/// separately (as scripts/prepare_sft.py did), &lt;|endoftext|&gt; ends the reply.</summary>
public sealed class ChatSession
{
    static readonly string[] RoleHeads = { "\n\nUser:", "\n\nSystem:", "\n\nAssistant:", "\n\nTool:" };

    public KestrelModel Model { get; }
    public InferenceState State { get; private set; }
    public List<ChatTurn> Turns { get; } = new();
    public string? SystemPrompt { get; private set; }
    public EpisodicStore? Memory { get; set; }
    public int MemoryTopK { get; set; } = 3;
    public float MemoryMinScore { get; set; } = 0.5f;
    public int Loops => State.Loops;

    public ChatSession(KestrelModel model, int? loops = null, string? systemPrompt = null)
    {
        Model = model;
        State = model.NewState(loops);
        SystemPrompt = string.IsNullOrWhiteSpace(systemPrompt) ? null : systemPrompt;
    }

    // ------------------------------------------------------------------ template

    List<int> TurnTokens(string role, string content, bool first)
    {
        var tok = Model.Tokenizer;
        var ids = tok.Encode(first ? $"{role}: " : $"\n\n{role}: ");
        ids.AddRange(tok.Encode(content));
        return ids;
    }

    List<int> TranscriptTokens(IEnumerable<ChatTurn> turns)
    {
        var ids = new List<int>();
        bool first = true;
        if (SystemPrompt != null) { ids.AddRange(TurnTokens("System", SystemPrompt, true)); first = false; }
        foreach (var t in turns)
        {
            ids.AddRange(TurnTokens(t.Role, t.Content, first));
            first = false;
        }
        return ids;
    }

    /// <summary>Re-read the transcript into a fresh state — after a loop-count change, a
    /// system-prompt change, or when the context fills (oldest turns are dropped).</summary>
    public void Rebuild(int? loops = null, string? systemPrompt = null, bool changeSystem = false)
    {
        if (changeSystem) SystemPrompt = string.IsNullOrWhiteSpace(systemPrompt) ? null : systemPrompt;
        var st = Model.NewState(loops ?? State.Loops);
        int budget = Model.Config.ContextLength * 3 / 4;
        int skip = 0;
        List<int> ids;
        while ((ids = TranscriptTokens(Turns.Skip(skip))).Count > budget && skip < Turns.Count) skip++;
        if (skip > 0) Turns.RemoveRange(0, skip);
        if (ids.Count > 0) Feed(st, ids);
        State = st;
    }

    void Feed(InferenceState st, List<int> ids)
    {
        const int Chunk = 256;           // bounds peak activation memory on long prompts
        for (int i = 0; i < ids.Count; i += Chunk)
            Model.Forward(st, ids.GetRange(i, Math.Min(Chunk, ids.Count - i)).ToArray());
    }

    // ------------------------------------------------------------------ chat

    /// <summary>Send a user message and stream the reply. <paramref name="onText"/> receives
    /// text as it is generated; the returned stats say why generation stopped.</summary>
    public GenerationStats Send(string userText, Sampler sampler, int maxNewTokens,
                                Action<string>? onText = null, CancellationToken ct = default)
    {
        var recalled = new List<string>();
        string content = userText;
        if (Memory != null)
        {
            var hits = Memory.Retrieve(Model, userText, MemoryTopK, MemoryMinScore);
            recalled = hits.Select(h => h.Item.Text).ToList();
            content = EpisodicStore.AsContext(hits.Select(h => h.Item)) + userText;
        }
        bool first = Turns.Count == 0 && SystemPrompt == null && State.Position == 0;
        var prompt = TurnTokens("User", content, first);
        prompt.AddRange(Model.Tokenizer.Encode("\n\nAssistant: "));

        int ctx = Model.Config.ContextLength;
        if (State.Position + prompt.Count + maxNewTokens > ctx)
        {
            Rebuild();
            if (State.Position + prompt.Count + maxNewTokens > ctx)
                throw new ContextFullException(State.Position, prompt.Count + maxNewTokens, ctx);
        }
        Turns.Add(new ChatTurn("User", userText) { Recalled = recalled });

        var sw = Stopwatch.StartNew();
        var lastIds = prompt.ToArray();
        float[] logits = FeedPrompt(lastIds);
        double promptSec = sw.Elapsed.TotalSeconds;

        var reply = new StringBuilder();
        var dec = new StreamingDecoder(Model.Tokenizer);
        var snaps = new List<(InferenceState Snap, int ReplyLen)>();
        int n = 0, emitted = 0;
        string stop = "length";
        sw.Restart();
        while (n < maxNewTokens)
        {
            if (ct.IsCancellationRequested) { stop = "cancelled"; break; }
            int id = sampler.Sample(logits, State.Tokens);
            if (id == Model.Tokenizer.EotId) { stop = "eos"; break; }
            if (State.Position + 1 > ctx) { stop = "context"; break; }

            string piece = dec.Push(id);
            if (piece.Contains('\n'))
            {
                snaps.Add((State.Snapshot(), reply.Length));
                if (snaps.Count > 4) snaps.RemoveAt(0);
            }
            reply.Append(piece);
            logits = Model.Forward(State, new[] { id });
            n++;

            // the model starting a new "Role:" turn means its own turn is over
            int cut = FindRoleHead(reply);
            if (cut >= 0)
            {
                stop = "turn";
                var (snap, len) = snaps.LastOrDefault(s => s.ReplyLen <= cut);
                if (snap != null)
                {
                    State.Restore(snap);
                    string tail = reply.ToString(len, cut - len);
                    if (tail.Length > 0) Model.Forward(State, Model.Tokenizer.Encode(tail).ToArray());
                }
                reply.Length = cut;
                break;
            }
            // hold back a possible partial role header so the UI never flashes "\n\nUse"
            int safe = SafeEmitLength(reply);
            if (safe > emitted) { onText?.Invoke(reply.ToString(emitted, safe - emitted)); emitted = safe; }
        }
        if (reply.Length > emitted) onText?.Invoke(reply.ToString(emitted, reply.Length - emitted));

        Turns.Add(new ChatTurn("Assistant", reply.ToString().Trim()));
        return new GenerationStats(prompt.Count, promptSec, n, sw.Elapsed.TotalSeconds, stop);
    }

    float[] FeedPrompt(int[] ids)
    {
        float[] logits = Array.Empty<float>();
        for (int i = 0; i < ids.Length; i += 256)
            logits = Model.Forward(State, ids.AsSpan(i, Math.Min(256, ids.Length - i)));
        return logits;
    }

    static int FindRoleHead(StringBuilder sb)
    {
        string s = sb.ToString();
        int best = -1;
        foreach (var h in RoleHeads)
        {
            int i = s.IndexOf(h, StringComparison.Ordinal);
            if (i >= 0 && (best < 0 || i < best)) best = i;
        }
        return best;
    }

    static int SafeEmitLength(StringBuilder sb)
    {
        string s = sb.ToString();
        int hold = s.LastIndexOf("\n\n", StringComparison.Ordinal);
        if (hold < 0) return s.EndsWith('\n') ? s.Length - 1 : s.Length;
        string tail = s[hold..];
        return RoleHeads.Any(h => h.StartsWith(tail, StringComparison.Ordinal)) ? hold : s.Length;
    }

    // ------------------------------------------------------------------ persistence

    /// <summary>Save the conversation AND the model's exact internal state. Loading it
    /// resumes instantly — nothing is re-read.</summary>
    public void Save(string path)
    {
        using var fs = File.Create(path + ".tmp");
        var meta = JsonSerializer.SerializeToUtf8Bytes(new SessionMeta
        {
            Model = Model.Config.Name, SystemPrompt = SystemPrompt, Turns = Turns,
            Saved = DateTimeOffset.Now,
        });
        fs.Write(BitConverter.GetBytes(meta.Length));
        fs.Write(meta);
        State.Save(fs, Model.Config);
        fs.Close();
        File.Move(path + ".tmp", path, overwrite: true);
    }

    public static ChatSession Load(KestrelModel model, string path)
    {
        using var fs = File.OpenRead(path);
        var lenBuf = new byte[4];
        fs.ReadExactly(lenBuf);
        var metaBuf = new byte[BitConverter.ToInt32(lenBuf)];
        fs.ReadExactly(metaBuf);
        var meta = JsonSerializer.Deserialize<SessionMeta>(metaBuf)!;
        var s = new ChatSession(model, systemPrompt: meta.SystemPrompt);
        s.Turns.AddRange(meta.Turns);
        s.State = InferenceState.Load(fs, model.Config);
        return s;
    }

    sealed class SessionMeta
    {
        public string Model { get; set; } = "";
        public string? SystemPrompt { get; set; }
        public List<ChatTurn> Turns { get; set; } = new();
        public DateTimeOffset Saved { get; set; }
    }
}
