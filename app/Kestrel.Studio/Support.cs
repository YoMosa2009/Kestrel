using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Kestrel.Studio;

/// <summary>Everything Kestrel Studio keeps lives under %LOCALAPPDATA%\KestrelStudio, so the
/// app is a single exe that can be copied to any PC.</summary>
public static class AppPaths
{
    /// <summary>Release label of this build of the app (bundled with the same-numbered model).</summary>
    public const string StudioVersion = "V2";

    /// <summary>Folder the exe runs from: a .gguf shipped next to it is offered without copying.</summary>
    public static readonly string ExeDir = AppContext.BaseDirectory;

    public static readonly string Root = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "KestrelStudio");
    public static readonly string Models = Path.Combine(Root, "models");
    public static readonly string Sessions = Path.Combine(Root, "sessions");
    public static readonly string Memory = Path.Combine(Root, "memory", "episodic.jsonl");
    public static readonly string SettingsFile = Path.Combine(Root, "settings.json");

    public static void Ensure()
    {
        Directory.CreateDirectory(Models);
        Directory.CreateDirectory(Sessions);
        Directory.CreateDirectory(Path.GetDirectoryName(Memory)!);
    }
}

public sealed class Settings
{
    public string? LastModel { get; set; }
    public int Loops { get; set; } = 2;
    public float Temperature { get; set; } = 0.7f;
    public int TopK { get; set; } = 40;
    public float TopP { get; set; } = 0.95f;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public int MaxTokens { get; set; } = 400;
    public string SystemPrompt { get; set; } = "";
    public bool UseMemory { get; set; } = true;

    public static Settings Load()
    {
        try
        {
            if (File.Exists(AppPaths.SettingsFile))
                return JsonSerializer.Deserialize<Settings>(File.ReadAllText(AppPaths.SettingsFile)) ?? new();
        }
        catch { /* a corrupt settings file must never stop the app from starting */ }
        return new();
    }

    public void Save() =>
        File.WriteAllText(AppPaths.SettingsFile, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
}

/// <summary>One chat bubble. Text grows while the reply streams in.</summary>
public sealed class ChatMessage : INotifyPropertyChanged
{
    string _text = "", _footer = "";
    public string Role { get; init; } = "Assistant";
    public bool IsUser => Role == "User";
    public HorizontalAlignment Align => IsUser ? HorizontalAlignment.Right : HorizontalAlignment.Left;
    public Brush Bubble => (Brush)Application.Current.Resources[IsUser ? "UserBubble" : "BotBubble"];
    public string Header => IsUser ? "You" : "Kestrel";

    public string Text { get => _text; set { _text = value; On(); } }
    public string Footer { get => _footer; set { _footer = value; On(); On(nameof(HasFooter)); } }
    public bool HasFooter => _footer.Length > 0;

    public event PropertyChangedEventHandler? PropertyChanged;
    void On([CallerMemberName] string? p = null) => PropertyChanged?.Invoke(this, new(p));
}

public sealed record ModelEntry(string Path, string Name, long Bytes, bool Bundled = false)
{
    public string Label => $"{Name}  ({Bytes / 1e6:F0} MB){(Bundled ? "  · bundled" : "")}";
}

/// <summary>Live picture of one product-key-memory site: a heat-mapped nKeys × nKeys grid
/// of slots, brightening where tokens read and fading over time. This is the part of the
/// model Roost writes new knowledge into.</summary>
public sealed class PkmHeatmap
{
    readonly int _n;
    readonly float[] _heat;
    readonly int[] _pending;
    public WriteableBitmap Bitmap { get; }
    public long TotalReads { get; private set; }
    public int DistinctSlots => _everTouched.Count;
    readonly HashSet<int> _everTouched = new();

    public PkmHeatmap(int nKeys)
    {
        _n = nKeys;
        _heat = new float[nKeys * nKeys];
        _pending = new int[nKeys * nKeys];
        Bitmap = new WriteableBitmap(nKeys, nKeys, 96, 96, PixelFormats.Bgra32, null);
        Render();
    }

    /// <summary>Thread-safe: called from inference worker threads.</summary>
    public void Hit(int[] slots)
    {
        foreach (int s in slots) Interlocked.Increment(ref _pending[s]);
    }

    /// <summary>UI thread: fold pending hits in, decay, redraw.</summary>
    public void Tick()
    {
        for (int i = 0; i < _heat.Length; i++)
        {
            int p = Interlocked.Exchange(ref _pending[i], 0);
            if (p > 0) { _everTouched.Add(i); TotalReads += p; }
            _heat[i] = Math.Min(1f, _heat[i] * 0.93f + p * 0.35f);
        }
        Render();
    }

    void Render()
    {
        var px = new byte[_n * _n * 4];
        for (int i = 0; i < _heat.Length; i++)
        {
            float h = _heat[i];
            bool seen = _everTouched.Contains(i);
            // dark slate -> amber -> pale yellow
            byte r = (byte)(32 + h * 223), g = (byte)(36 + h * 170), b = (byte)(48 + h * 40);
            if (!seen && h == 0) { r = 26; g = 29; b = 38; }
            px[i * 4] = b; px[i * 4 + 1] = g; px[i * 4 + 2] = r; px[i * 4 + 3] = 255;
        }
        Bitmap.WritePixels(new Int32Rect(0, 0, _n, _n), px, _n * 4, 0);
    }
}
