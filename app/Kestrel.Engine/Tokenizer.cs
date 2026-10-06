using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Kestrel.Engine;

/// <summary>Byte-level BPE matching the Hugging Face tokenizer Kestrel was trained with
/// (tokenizer/kestrel-bpe.json): pre-tokenize = Digits(individual) then the GPT-2
/// ByteLevel regex, then rank-ordered BPE merges over the byte-to-unicode alphabet.
/// The one special token, &lt;|endoftext|&gt;, is matched literally before BPE.</summary>
public sealed class BpeTokenizer
{
    public const string EndOfText = "<|endoftext|>";

    readonly Dictionary<string, int> _vocab;
    readonly string[] _tokens;
    readonly Dictionary<(string, string), int> _ranks;
    readonly bool _splitDigits;
    readonly Dictionary<string, int[]> _cache = new();
    readonly object _cacheLock = new();

    static readonly Regex Gpt2 = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);
    static readonly char[] ByteToChar = BuildByteMap();
    static readonly Dictionary<char, byte> CharToByte =
        Enumerable.Range(0, 256).ToDictionary(b => ByteToChar[b], b => (byte)b);

    public int EotId { get; }
    public int VocabSize => _tokens.Length;

    public BpeTokenizer(string[] tokens, string[] merges, int eotId, bool splitDigits)
    {
        _tokens = tokens;
        _vocab = new Dictionary<string, int>(tokens.Length);
        for (int i = 0; i < tokens.Length; i++) _vocab[tokens[i]] = i;
        _ranks = new Dictionary<(string, string), int>(merges.Length);
        for (int i = 0; i < merges.Length; i++)
        {
            int sp = merges[i].IndexOf(' ');
            _ranks[(merges[i][..sp], merges[i][(sp + 1)..])] = i;
        }
        EotId = eotId;
        _splitDigits = splitDigits;
    }

    public static BpeTokenizer FromGguf(GgufFile g) => new(
        g.GetStrings("tokenizer.ggml.tokens"),
        g.GetStrings("tokenizer.ggml.merges"),
        g.GetInt("tokenizer.ggml.eos_token_id"),
        g.GetString("tokenizer.ggml.pre", "gpt2") == "kestrel-digits");

    // ------------------------------------------------------------------ encode

    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        int start = 0;
        while (true)
        {
            int at = text.IndexOf(EndOfText, start, StringComparison.Ordinal);
            EncodeOrdinary(at < 0 ? text[start..] : text[start..at], ids);
            if (at < 0) break;
            ids.Add(EotId);
            start = at + EndOfText.Length;
        }
        return ids;
    }

    void EncodeOrdinary(string text, List<int> ids)
    {
        foreach (var piece in PreTokenize(text))
            foreach (Match m in Gpt2.Matches(piece))
                ids.AddRange(EncodeWord(m.Value));
    }

    /// <summary>HF Digits(individual_digits=true): every numeric char becomes its own piece.</summary>
    IEnumerable<string> PreTokenize(string text)
    {
        if (!_splitDigits) { yield return text; yield break; }
        int start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (!IsNumeric(text[i])) continue;
            if (i > start) yield return text[start..i];
            yield return text[i].ToString();
            start = i + 1;
        }
        if (start < text.Length) yield return text[start..];
    }

    static bool IsNumeric(char c) => CharUnicodeInfo.GetUnicodeCategory(c) is
        UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber;

    int[] EncodeWord(string word)
    {
        lock (_cacheLock)
            if (_cache.TryGetValue(word, out var hit)) return hit;

        var bytes = Encoding.UTF8.GetBytes(word);
        var syms = new List<string>(bytes.Length);
        foreach (var b in bytes) syms.Add(ByteToChar[b].ToString());

        while (syms.Count > 1)
        {
            int best = int.MaxValue, at = -1;
            for (int i = 0; i < syms.Count - 1; i++)
                if (_ranks.TryGetValue((syms[i], syms[i + 1]), out int r) && r < best) { best = r; at = i; }
            if (at < 0) break;
            string a = syms[at], b = syms[at + 1], merged = a + b;
            var next = new List<string>(syms.Count);
            for (int i = 0; i < syms.Count; i++)
            {
                if (i < syms.Count - 1 && syms[i] == a && syms[i + 1] == b) { next.Add(merged); i++; }
                else next.Add(syms[i]);
            }
            syms = next;
        }

        var ids = new int[syms.Count];
        for (int i = 0; i < syms.Count; i++)
            ids[i] = _vocab.TryGetValue(syms[i], out int id) ? id
                : throw new InvalidDataException($"BPE symbol '{syms[i]}' missing from vocab");
        lock (_cacheLock)
            if (_cache.Count < 100_000) _cache[word] = ids;
        return ids;
    }

    // ------------------------------------------------------------------ decode

    public byte[] DecodeBytes(IEnumerable<int> ids)
    {
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if (id == EotId) { bytes.AddRange(Encoding.UTF8.GetBytes(EndOfText)); continue; }
            foreach (char c in _tokens[id])
                if (CharToByte.TryGetValue(c, out byte b)) bytes.Add(b);
        }
        return bytes.ToArray();
    }

    public string Decode(IEnumerable<int> ids) => Encoding.UTF8.GetString(DecodeBytes(ids));

    public string TokenText(int id) => _tokens[id];

    /// <summary>GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100+.</summary>
    static char[] BuildByteMap()
    {
        var map = new char[256];
        var direct = new bool[256];
        foreach (var (lo, hi) in new[] { ('!', '~'), ('¡', '¬'), ('®', 'ÿ') })
            for (int c = lo; c <= hi; c++) direct[c] = true;
        int extra = 0;
        for (int b = 0; b < 256; b++)
            map[b] = direct[b] ? (char)b : (char)(256 + extra++);
        return map;
    }
}

/// <summary>Turns a stream of token ids into text without splitting multi-byte UTF-8
/// characters across tokens (byte-level BPE routinely does).</summary>
public sealed class StreamingDecoder
{
    readonly BpeTokenizer _tok;
    readonly Decoder _utf8 = new UTF8Encoding(false).GetDecoder();
    readonly char[] _buf = new char[1024];   // > longest token's UTF-8 length

    public StreamingDecoder(BpeTokenizer tok) => _tok = tok;

    public string Push(int id)
    {
        var bytes = _tok.DecodeBytes(new[] { id });
        var sb = new StringBuilder();
        int n = _utf8.GetChars(bytes, 0, bytes.Length, _buf, 0, flush: false);
        sb.Append(_buf, 0, n);
        return sb.ToString();
    }
}
