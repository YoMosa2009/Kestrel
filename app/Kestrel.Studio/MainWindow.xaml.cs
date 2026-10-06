using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using Kestrel.Engine;
using Microsoft.Win32;

namespace Kestrel.Studio;

public partial class MainWindow : Window
{
    readonly Settings _settings = Settings.Load();
    readonly ObservableCollection<ChatMessage> _messages = new();
    readonly EpisodicStore _memory;
    readonly DispatcherTimer _tick = new() { Interval = TimeSpan.FromMilliseconds(120) };

    KestrelModel? _model;
    ChatSession? _chat;
    PkmHeatmap[] _heatmaps = Array.Empty<PkmHeatmap>();
    CancellationTokenSource? _cts;
    bool _busy, _ready;

    public MainWindow()
    {
        AppPaths.Ensure();
        InitializeComponent();
        Title = $"Kestrel Studio {AppPaths.StudioVersion}";
        _memory = new EpisodicStore(AppPaths.Memory);
        Messages.ItemsSource = _messages;

        LoopsSlider.Value = _settings.Loops;
        TempSlider.Value = _settings.Temperature;
        TopPSlider.Value = _settings.TopP;
        TopKSlider.Value = _settings.TopK;
        RepSlider.Value = _settings.RepetitionPenalty;
        MaxSlider.Value = _settings.MaxTokens;
        SystemBox.Text = _settings.SystemPrompt;
        UseMemory.IsChecked = _settings.UseMemory;
        _ready = true;
        UpdateLabels();
        RefreshFacts();
        RefreshSessions();
        RefreshModels(select: _settings.LastModel);

        _tick.Tick += (_, _) =>
        {
            foreach (var h in _heatmaps) h.Tick();
            UpdateStateInfo();
        };
        _tick.Start();
        SetBusy(false, _model == null ? "Import a Kestrel model to start" : "ready");
    }

    // ================================================================== models

    void RefreshModels(string? select = null)
    {
        var bundled = Directory.EnumerateFiles(AppPaths.ExeDir, "*.gguf")
            .Select(p => new ModelEntry(p, Path.GetFileNameWithoutExtension(p), new FileInfo(p).Length, Bundled: true));
        var library = Directory.EnumerateFiles(AppPaths.Models, "*.gguf")
            .Select(p => new ModelEntry(p, Path.GetFileNameWithoutExtension(p), new FileInfo(p).Length));
        // newest release first (v2 before v1), bundled copy preferred over a library duplicate
        var entries = bundled.Concat(library)
            .GroupBy(e => e.Name, StringComparer.OrdinalIgnoreCase).Select(g => g.First())
            .OrderByDescending(e => e.Name, StringComparer.OrdinalIgnoreCase).ToList();
        ModelList.ItemsSource = entries;
        var pick = entries.FirstOrDefault(e => e.Path == select) ?? entries.FirstOrDefault();
        if (pick != null) ModelList.SelectedItem = pick;
    }

    void OnModelPicked(object sender, SelectionChangedEventArgs e)
    {
        if (ModelList.SelectedItem is ModelEntry m && m.Path != _model?.Path) _ = LoadModelAsync(m.Path);
    }

    async Task LoadModelAsync(string path)
    {
        if (_busy) return;
        SetBusy(true, $"loading {Path.GetFileName(path)} …");
        LoadBar.Visibility = Visibility.Visible;
        var progress = new Progress<double>(v => LoadBar.Value = v);
        try
        {
            var sw = Stopwatch.StartNew();
            var model = await Task.Run(() => KestrelModel.Load(path, progress));
            _model = model;
            _settings.LastModel = path;
            _settings.Save();
            var c = model.Config;
            ModelName.Text = c.Name;
            Title = $"Kestrel Studio {AppPaths.StudioVersion} — {c.Name}";
            ModelInfo.Text = $"{model.ParameterCount / 1e6:F0}M parameters · {c.NEntry}+{c.NCore}×R+{c.NExit} blocks\n" +
                             $"context {c.ContextLength:N0} tokens · trained on {c.TokensSeen / 1e9:F2}B tokens\n" +
                             $"loaded in {sw.Elapsed.TotalSeconds:F1}s · {Environment.ProcessorCount} CPU threads";
            LoopsSlider.Maximum = c.RMax;
            BuildHeatmaps(c);
            model.OnPkm = hit => _heatmaps[hit.Site].Hit(hit.Slots);
            NewChat();
            SetBusy(false, "ready");
        }
        catch (Exception ex)
        {
            SetBusy(false, "load failed");
            MessageBox.Show(this, ex.Message, "Could not load model", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally { LoadBar.Visibility = Visibility.Collapsed; }
    }

    void OnImport(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Kestrel model (*.gguf)|*.gguf", Title = "Import a Kestrel model" };
        if (dlg.ShowDialog(this) == true) _ = ImportAsync(dlg.FileName);
    }

    void OnDrop(object sender, DragEventArgs e)
    {
        if (e.Data.GetData(DataFormats.FileDrop) is string[] files)
            foreach (var f in files.Where(f => f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)))
            { _ = ImportAsync(f); break; }
    }

    /// <summary>Validate (must be architecture "kestrel"), then copy into the app's model library.</summary>
    async Task ImportAsync(string src)
    {
        try
        {
            var g = GgufFile.Open(src);
            KestrelConfig.FromGguf(g);           // throws with a clear message for non-Kestrel files
            string dst = Path.Combine(AppPaths.Models, Path.GetFileName(src));
            if (!string.Equals(Path.GetFullPath(src), Path.GetFullPath(dst), StringComparison.OrdinalIgnoreCase))
            {
                SetBusy(true, $"importing {Path.GetFileName(src)} …");
                await Task.Run(() => File.Copy(src, dst, overwrite: true));
                SetBusy(false, "imported");
            }
            RefreshModels(select: dst);
        }
        catch (Exception ex)
        {
            SetBusy(false, "import failed");
            MessageBox.Show(this, ex.Message, "Not a Kestrel model", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    void OnOpenFolder(object sender, RoutedEventArgs e) =>
        Process.Start(new ProcessStartInfo("explorer.exe", AppPaths.Root) { UseShellExecute = true });

    void BuildHeatmaps(KestrelConfig c)
    {
        _heatmaps = c.PkmSites.Select(_ => new PkmHeatmap(c.PkmNKeys)).ToArray();
        PkmPanels.Items.Clear();
        for (int i = 0; i < _heatmaps.Length; i++)
        {
            var panel = new StackPanel { Margin = new Thickness(0, 0, 0, 10) };
            panel.Children.Add(new TextBlock
            {
                Text = $"Site {i + 1} · block {c.PkmSites[i]} · {c.PkmNKeys * c.PkmNKeys:N0} slots",
                Foreground = (Brush)FindResource("Muted"), FontSize = 11, Margin = new Thickness(0, 0, 0, 4),
            });
            var img = new Image { Source = _heatmaps[i].Bitmap, Width = 256, Height = 256, HorizontalAlignment = HorizontalAlignment.Left };
            RenderOptions.SetBitmapScalingMode(img, BitmapScalingMode.NearestNeighbor);
            panel.Children.Add(img);
            PkmPanels.Items.Add(panel);
        }
    }

    // ================================================================== chat

    void OnInputKey(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && Keyboard.Modifiers != ModifierKeys.Shift) { e.Handled = true; OnSend(sender, e); }
    }

    async void OnSend(object sender, RoutedEventArgs e)
    {
        string text = Input.Text.Trim();
        if (text.Length == 0 || _busy) return;
        if (_model == null || _chat == null)
        {
            MessageBox.Show(this, "Import or pick a Kestrel model first.", "Kestrel Studio");
            return;
        }
        Input.Clear();
        _messages.Add(new ChatMessage { Role = "User", Text = text });
        var reply = new ChatMessage { Role = "Assistant" };
        _messages.Add(reply);
        ChatScroll.ScrollToEnd();

        _chat.Memory = UseMemory.IsChecked == true ? _memory : null;
        var sampler = new Sampler(CurrentSampling());
        int max = (int)MaxSlider.Value;
        _cts = new CancellationTokenSource();
        var ct = _cts.Token;
        SetBusy(true, "thinking …", generating: true);
        try
        {
            var chat = _chat;
            var stats = await Task.Run(() => chat.Send(text, sampler, max,
                piece => Dispatcher.BeginInvoke(() => { reply.Text += piece; ChatScroll.ScrollToEnd(); }), ct));
            await Dispatcher.InvokeAsync(() => { }, DispatcherPriority.Background);   // flush queued text
            reply.Text = chat.Turns[^1].Content;
            var recalled = chat.Turns[^2].Recalled;
            reply.Footer = $"R={chat.Loops} · {stats.NewTokens} tokens · {stats.GenTokPerSec:F1} tok/s · " +
                           $"prompt {stats.PromptTokPerSec:F0} tok/s" +
                           (stats.StopReason is "length" or "cancelled" ? $" · {stats.StopReason}" : "");
            RecalledText.Text = recalled.Count > 0 ? "Recalled: " + string.Join(" · ", recalled) : "";
            if (recalled.Count > 0) _messages[^2].Footer = $"recalled {recalled.Count} memor{(recalled.Count == 1 ? "y" : "ies")}";
            SpeedText.Text = $"{stats.GenTokPerSec:F1} tok/s";
            SetBusy(false, "ready");
        }
        catch (Exception ex)
        {
            reply.Footer = "error: " + ex.Message;
            SetBusy(false, "error");
        }
        UpdateStateInfo();
    }

    void OnStop(object sender, RoutedEventArgs e) => _cts?.Cancel();

    SamplingOptions CurrentSampling() => new()
    {
        Temperature = (float)TempSlider.Value,
        TopP = (float)TopPSlider.Value,
        TopK = (int)TopKSlider.Value,
        RepetitionPenalty = (float)RepSlider.Value,
    };

    void NewChat()
    {
        if (_model == null) return;
        _chat = new ChatSession(_model, (int)LoopsSlider.Value, SystemBox.Text);
        _messages.Clear();
        RecalledText.Text = "";
        UpdateStateInfo();
    }

    void OnNewChat(object sender, RoutedEventArgs e) { if (!_busy) NewChat(); }

    async void OnLoopsChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!_ready) return;
        UpdateLabels();
        _settings.Loops = (int)LoopsSlider.Value;
        _settings.Save();
        if (_chat == null || _busy || _chat.Loops == (int)LoopsSlider.Value) return;
        await RebuildAsync(() => _chat.Rebuild(loops: (int)LoopsSlider.Value), $"re-reading chat at R={(int)LoopsSlider.Value} …");
    }

    async void OnApplySystem(object sender, RoutedEventArgs e)
    {
        _settings.SystemPrompt = SystemBox.Text;
        _settings.Save();
        if (_chat == null || _busy) return;
        await RebuildAsync(() => _chat.Rebuild(systemPrompt: SystemBox.Text, changeSystem: true), "applying system prompt …");
    }

    async Task RebuildAsync(Action rebuild, string status)
    {
        SetBusy(true, status);
        try { await Task.Run(rebuild); SetBusy(false, "ready"); }
        catch (Exception ex) { SetBusy(false, "error"); MessageBox.Show(this, ex.Message, "Kestrel Studio"); }
        UpdateStateInfo();
    }

    // ================================================================== sessions

    void RefreshSessions()
    {
        SessionList.ItemsSource = Directory.EnumerateFiles(AppPaths.Sessions, "*.kses")
            .OrderByDescending(File.GetLastWriteTime)
            .Select(Path.GetFileNameWithoutExtension).ToList();
    }

    void OnSaveSession(object sender, RoutedEventArgs e)
    {
        if (_chat == null || _busy || _chat.Turns.Count == 0) return;
        string first = _chat.Turns[0].Content;
        string name = new string(first.Take(40).Select(ch => Path.GetInvalidFileNameChars().Contains(ch) ? '_' : ch).ToArray()).Trim();
        string path = Path.Combine(AppPaths.Sessions, $"{DateTime.Now:yyyy-MM-dd HHmm} {name}.kses");
        _chat.Save(path);
        RefreshSessions();
        StatusText.Text = $"session saved ({new FileInfo(path).Length / 1e6:F1} MB)";
    }

    async void OnOpenSession(object sender, MouseButtonEventArgs e)
    {
        if (_model == null || _busy || SessionList.SelectedItem is not string name) return;
        string path = Path.Combine(AppPaths.Sessions, name + ".kses");
        try
        {
            SetBusy(true, "restoring session …");
            var model = _model;
            _chat = await Task.Run(() => ChatSession.Load(model, path));
            _messages.Clear();
            foreach (var t in _chat.Turns) _messages.Add(new ChatMessage { Role = t.Role, Text = t.Content });
            _ready = false; LoopsSlider.Value = _chat.Loops; _ready = true;
            UpdateLabels();
            SetBusy(false, $"session restored — {_chat.State.Position:N0} tokens of state, nothing re-read");
            ChatScroll.ScrollToEnd();
        }
        catch (Exception ex)
        {
            SetBusy(false, "restore failed");
            MessageBox.Show(this, ex.Message, "Could not restore session");
        }
        UpdateStateInfo();
    }

    // ================================================================== memory

    void RefreshFacts()
    {
        FactList.ItemsSource = _memory.Items.OrderByDescending(i => i.Ts).ToList();
        MemoryCount.Text = $"{_memory.Items.Count} memories · {AppPaths.Memory}";
    }

    void OnTeach(object sender, RoutedEventArgs e)
    {
        foreach (var line in TeachBox.Text.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0))
            _memory.Add(line);
        TeachBox.Clear();
        RefreshFacts();
    }

    void OnForget(object sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: Teachable t }) { _memory.Remove(t); RefreshFacts(); }
    }

    void OnUseMemory(object sender, RoutedEventArgs e)
    {
        _settings.UseMemory = UseMemory.IsChecked == true;
        _settings.Save();
    }

    // ================================================================== ui state

    void OnSampling(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!_ready) return;
        UpdateLabels();
        _settings.Temperature = (float)TempSlider.Value;
        _settings.TopP = (float)TopPSlider.Value;
        _settings.TopK = (int)TopKSlider.Value;
        _settings.RepetitionPenalty = (float)RepSlider.Value;
        _settings.MaxTokens = (int)MaxSlider.Value;
        _settings.Save();
    }

    void UpdateLabels()
    {
        int r = (int)LoopsSlider.Value;
        LoopsText.Text = $"R = {r} " + (r == 1 ? "(fast)" : r == 2 ? "(trained)" : "(deep)");
        TempText.Text = $"Temperature {TempSlider.Value:F2}" + (TempSlider.Value < 0.01 ? " (greedy)" : "");
        TopPText.Text = $"Top-p {TopPSlider.Value:F2}";
        TopKText.Text = $"Top-k {(int)TopKSlider.Value}" + (TopKSlider.Value == 0 ? " (off)" : "");
        RepText.Text = $"Repetition penalty {RepSlider.Value:F2}";
        MaxText.Text = $"Max reply length {(int)MaxSlider.Value} tokens";
    }

    void UpdateStateInfo()
    {
        if (_model == null || _chat == null) { StateInfo.Text = "no model"; CtxText.Text = ""; return; }
        var c = _model.Config;
        int pos = _chat.State.Position;
        CtxBar.Value = pos / (double)c.ContextLength;
        CtxText.Text = $"context {pos:N0} / {c.ContextLength:N0} tokens";
        int Attn(int from, int n) => Enumerable.Range(from, n).Count(c.IsAttention);
        int attn = Attn(0, c.NEntry) + _chat.Loops * Attn(c.NEntry, c.NCore) + Attn(c.NEntry + c.NCore, c.NExit);
        int gla = c.NEntry + _chat.Loops * c.NCore + c.NExit - attn;
        double glaMb = gla * (c.NHeads * c.HeadDim * c.HeadDim + (c.ConvKernel - 1) * c.DModel) * 4 / 1e6;
        double kvMb = attn * (double)pos * c.NKvHeads * c.HeadDim * 2 * 4 / 1e6;
        StateInfo.Text =
            $"Loops R = {_chat.Loops}: {c.NEntry + _chat.Loops * c.NCore + c.NExit} block passes per token\n" +
            $"Recurrent (GLA) state ≈ {glaMb:F1} MB — fixed size, however long the chat\n" +
            $"Attention KV cache ≈ {kvMb:F1} MB — grows with the chat\n" +
            $"Memory reads: " + string.Join(", ", _heatmaps.Select((h, i) => $"site {i + 1} {h.TotalReads:N0} ({h.DistinctSlots:N0} slots)"));
    }

    void SetBusy(bool busy, string status, bool generating = false)
    {
        _busy = busy;
        StatusText.Text = status;
        StateDot.Background = new SolidColorBrush(busy ? Color.FromRgb(0xE8, 0x92, 0x3C)
            : _model != null ? Color.FromRgb(0x3D, 0xDC, 0x84) : Color.FromRgb(0x55, 0x5B, 0x6B));
        BtnSend.IsEnabled = !busy;
        BtnStop.IsEnabled = generating;
        BtnNew.IsEnabled = BtnSave.IsEnabled = ModelList.IsEnabled = LoopsSlider.IsEnabled = !busy;
    }

    void OnClosing(object? sender, System.ComponentModel.CancelEventArgs e) => _cts?.Cancel();
}
