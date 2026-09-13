using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Threading;
using IOPath = System.IO.Path;

namespace KestrelMonitor;

/// <summary>
/// Read-only view of a Kestrel training run plus the three run controls.
///
/// Everything is file-based: the trainer publishes status.json (atomically) and
/// appends metrics.jsonl; the UI writes control.json. That means the app can be
/// closed, crash, or be started late without affecting a multi-day run, and the
/// same files are readable by anything else - including Claude over the shell.
/// </summary>
public partial class MainWindow : Window
{
    // v0.1 control, measured at seq 1024 (docs/10). The run trains at seq 1024,
    // so this is the only baseline that is comparable to the in-run probe.
    static readonly Dictionary<string, double> Baseline = new()
    {
        ["code"] = 1.9670, ["tool"] = 2.0694, ["know"] = 2.2954, ["inst"] = 2.3533,
        ["math"] = 2.5720, ["cli"] = 2.6582, ["docs"] = 2.6926, ["sec"] = 2.7032,
        ["repo"] = 2.7341, ["web"] = 3.1842,
    };

    static readonly Brush[] Palette =
    {
        Brush("#4C8DFF"), Brush("#3DDC84"), Brush("#FFB454"), Brush("#FF6B8A"),
        Brush("#B287FF"), Brush("#4ED8D8"), Brush("#E6E8EC"), Brush("#FF9F43"),
        Brush("#7FD97F"), Brush("#9AA3B8"),
    };

    readonly string _dir;
    readonly DispatcherTimer _timer = new();
    long _metricsLen = -1;
    int _msgCount = -1;

    readonly List<(int step, double loss)> _loss = new();
    readonly List<(int step, double rate)> _rate = new();
    readonly List<(int step, double lr)> _lr = new();
    readonly List<(int step, Dictionary<string, double> vals)> _val = new();

    public MainWindow()
    {
        InitializeComponent();
        var args = Environment.GetCommandLineArgs();
        _dir = args.Length > 1
            ? args[1]
            : IOPath.GetFullPath(IOPath.Combine(AppContext.BaseDirectory,
                                            "..", "..", "..", "..", "..",
                                            "experiments", "nano_d"));
        RunPath.Text = _dir;
        _timer.Interval = TimeSpan.FromSeconds(2);
        _timer.Tick += (_, _) => Refresh();
        _timer.Start();
        Refresh();
    }

    static SolidColorBrush Brush(string hex)
    {
        var b = new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
        b.Freeze();
        return b;
    }

    string P(string name) => IOPath.Combine(_dir, name);

    // ------------------------------------------------------------------ poll

    void Refresh()
    {
        LoadMetrics();
        LoadStatus();
        LoadMessages();
        Redraw();
    }

    static string? ReadShared(string path)
    {
        // The trainer may be mid-write; FileShare.ReadWrite avoids fighting it.
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read,
                                          FileShare.ReadWrite);
            using var sr = new StreamReader(fs);
            return sr.ReadToEnd();
        }
        catch { return null; }
    }

    void LoadStatus()
    {
        var txt = ReadShared(P("status.json"));
        if (txt is null) { StateText.Text = "waiting for trainer"; return; }

        JsonElement s;
        try { s = JsonDocument.Parse(txt).RootElement; }
        catch { return; }   // caught mid-write; next tick will get it

        string state = Str(s, "state") ?? "?";
        StateText.Text = state switch
        {
            "running" => "Running",
            "paused" => "Paused",
            "stopped" => "Stopped",
            "done" => "Complete",
            "starting" => "Starting",
            _ => state,
        };
        StateDot.Background = state switch
        {
            "running" => Brush("#3DDC84"),
            "paused" => Brush("#FFB454"),
            "done" => Brush("#4C8DFF"),
            _ => Brush("#FF6B8A"),
        };

        VStep.Text = $"{Num(s, "step"):#,0}";
        double prog = Num(s, "progress");
        VProg.Text = $"{prog * 100:0.00}%";
        ProgBar.Width = System.Math.Max(0, ((Border)ProgBar.Parent).ActualWidth * prog);
        VTok.Text = $"{Num(s, "tokens_new") / 1e9:0.000}B";
        VCum.Text = $"{Num(s, "tokens_seen") / 1e9:0.000}B";
        if (s.TryGetProperty("loss", out var l) && l.ValueKind == JsonValueKind.Number)
            VLoss.Text = $"{l.GetDouble():0.000}";
        if (s.TryGetProperty("tok_s", out var t) && t.ValueKind == JsonValueKind.Number)
            VRate.Text = $"{t.GetDouble():#,0}";
        VEta.Text = s.TryGetProperty("eta_s", out var e) && e.ValueKind == JsonValueKind.Number
            ? Dur(e.GetDouble()) : "-";

        BtnPause.IsEnabled = state == "running" || state == "starting";
        BtnResume.IsEnabled = state == "paused";
        BtnStop.IsEnabled = state is "running" or "paused" or "starting";

        if (s.TryGetProperty("last_probe", out var pr) && pr.ValueKind == JsonValueKind.Object)
            ShowProbe(pr);
    }

    static string? Str(JsonElement e, string k) =>
        e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    static double Num(JsonElement e, string k) =>
        e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : 0;

    static string Dur(double sec)
    {
        var ts = TimeSpan.FromSeconds(sec);
        if (ts.TotalDays >= 1) return $"{(int)ts.TotalDays}d {ts.Hours}h";
        if (ts.TotalHours >= 1) return $"{(int)ts.TotalHours}h {ts.Minutes}m";
        return $"{(int)ts.TotalMinutes}m";
    }

    void ShowProbe(JsonElement p)
    {
        var sb = new System.Text.StringBuilder();
        if (p.TryGetProperty("step", out var st)) sb.AppendLine($"step       {st}");
        if (p.TryGetProperty("loss", out var l)) sb.AppendLine($"loss       {l.GetDouble():0.0000}");
        if (p.TryGetProperty("pkm_gate_mean", out var g))
            sb.AppendLine($"pkm gate   {g.GetDouble():0.00000}");
        if (p.TryGetProperty("loop_gain", out var lg) && lg.ValueKind == JsonValueKind.Object)
        {
            sb.Append("loop gain  ");
            foreach (var r in lg.EnumerateObject())
                sb.Append($"{r.Name}={r.Value.GetDouble():0.000}  ");
            sb.AppendLine();
        }
        if (p.TryGetProperty("stage", out var sg) && sg.ValueKind == JsonValueKind.Object)
        {
            var top = sg.EnumerateObject().OrderByDescending(x => x.Value.GetDouble()).Take(4);
            sb.Append("curriculum ");
            foreach (var r in top) sb.Append($"{r.Name}:{r.Value.GetDouble():0.00} ");
        }
        ProbeInfo.Text = sb.ToString();

        if (!p.TryGetProperty("val_by_domain", out var vd) ||
            vd.ValueKind != JsonValueKind.Object) return;

        var rows = new List<object>();
        foreach (var r in vd.EnumerateObject().OrderBy(x => x.Name))
        {
            double now = r.Value.GetDouble();
            double bas = Baseline.TryGetValue(r.Name, out var b) ? b : double.NaN;
            double d = now - bas;
            rows.Add(new
            {
                Name = r.Name,
                Now = now.ToString("0.0000", CultureInfo.InvariantCulture),
                Base = double.IsNaN(bas) ? "-" : bas.ToString("0.0000", CultureInfo.InvariantCulture),
                Delta = double.IsNaN(d) ? "" : (d <= 0 ? "" : "+") + d.ToString("0.0000", CultureInfo.InvariantCulture),
                Colour = double.IsNaN(d) ? Palette[6] : (d <= 0 ? Brush("#3DDC84") : Brush("#FF6B8A")),
            });
        }
        ValTable.ItemsSource = rows;
    }

    // --------------------------------------------------------------- metrics

    void LoadMetrics()
    {
        var path = P("metrics.jsonl");
        if (!File.Exists(path)) return;
        var len = new FileInfo(path).Length;
        if (len == _metricsLen) return;      // unchanged since last tick
        _metricsLen = len;

        var txt = ReadShared(path);
        if (txt is null) return;

        _loss.Clear(); _rate.Clear(); _lr.Clear(); _val.Clear();
        foreach (var line in txt.Split('\n'))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            JsonElement r;
            try { r = JsonDocument.Parse(line).RootElement; } catch { continue; }
            if (!r.TryGetProperty("step", out var sv)) continue;
            int step = sv.GetInt32();

            if (r.TryGetProperty("loss", out var lv) && lv.ValueKind == JsonValueKind.Number)
            {
                _loss.Add((step, lv.GetDouble()));
                if (r.TryGetProperty("tok_s", out var tv)) _rate.Add((step, tv.GetDouble()));
                if (r.TryGetProperty("lr_mult", out var mv)) _lr.Add((step, mv.GetDouble()));
            }
            else if (r.TryGetProperty("probe", out var pv) &&
                     pv.TryGetProperty("val_by_domain", out var vd))
            {
                var d = new Dictionary<string, double>();
                foreach (var kv in vd.EnumerateObject()) d[kv.Name] = kv.Value.GetDouble();
                if (d.Count > 0) _val.Add((step, d));
            }
        }
    }

    // ---------------------------------------------------------------- charts

    void Redraw()
    {
        LossSub.Text = _loss.Count > 0
            ? $"{_loss.Count} points  |  latest {_loss[^1].loss:0.000}  |  "
              + $"min {_loss.Min(p => p.loss):0.000}"
            : "waiting for the first logged step";
        Line(CLoss, _loss.Select(p => ((double)p.step, p.loss)).ToList(), Palette[0], true);

        RateSub.Text = _rate.Count > 0
            ? $"median {Median(_rate.Select(p => p.rate).ToList()):#,0} tok/s" : "";
        Line(CRate, _rate.Select(p => ((double)p.step, p.rate)).ToList(), Palette[1], true);
        Line(CLr, _lr.Select(p => ((double)p.step, p.lr)).ToList(), Palette[2], true);

        ValSub.Text = _val.Count > 0
            ? $"{_val.Count} probes  |  mean {_val[^1].vals.Values.Average():0.0000} "
              + $"(v0.1 control 2.5229)"
            : "first probe lands at the next eval interval";
        DrawVal();
    }

    static double Median(List<double> xs)
    {
        if (xs.Count == 0) return 0;
        var s = xs.OrderBy(x => x).ToList();
        return s[s.Count / 2];
    }

    static void Axis(Canvas c, double lo, double hi)
    {
        for (int i = 0; i <= 4; i++)
        {
            double y = c.ActualHeight * i / 4.0;
            c.Children.Add(new Line
            {
                X1 = 34, X2 = c.ActualWidth, Y1 = y, Y2 = y,
                Stroke = Brush("#272B38"), StrokeThickness = 1,
            });
            c.Children.Add(new TextBlock
            {
                Text = (hi - (hi - lo) * i / 4.0).ToString("0.##"),
                Foreground = Brush("#8A90A0"), FontSize = 9,
                Margin = new Thickness(0, y - 7, 0, 0),
            });
        }
    }

    static void Line(Canvas c, List<(double x, double y)> pts, Brush stroke, bool axis)
    {
        c.Children.Clear();
        if (c.ActualWidth < 10 || pts.Count < 2) return;

        double lo = pts.Min(p => p.y), hi = pts.Max(p => p.y);
        if (hi - lo < 1e-9) { hi = lo + 1; }
        double pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
        if (axis) Axis(c, lo, hi);

        double x0 = pts.Min(p => p.x), x1 = pts.Max(p => p.x);
        double w = c.ActualWidth - 38, h = c.ActualHeight;
        var poly = new Polyline { Stroke = stroke, StrokeThickness = 1.8 };
        foreach (var (x, y) in pts)
            poly.Points.Add(new Point(
                34 + (x1 > x0 ? (x - x0) / (x1 - x0) : 0) * w,
                h - (y - lo) / (hi - lo) * h));
        c.Children.Add(poly);
    }

    void DrawVal()
    {
        CVal.Children.Clear();
        ValLegend.Children.Clear();
        if (_val.Count < 2 || CVal.ActualWidth < 10) return;

        var names = _val[^1].vals.Keys.OrderBy(n => n).ToList();
        double lo = _val.Min(v => v.vals.Values.Min());
        double hi = _val.Max(v => v.vals.Values.Max());
        if (hi - lo < 1e-9) hi = lo + 1;
        double pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
        Axis(CVal, lo, hi);

        double x0 = _val[0].step, x1 = _val[^1].step;
        double w = CVal.ActualWidth - 38, h = CVal.ActualHeight;
        for (int i = 0; i < names.Count; i++)
        {
            var br = Palette[i % Palette.Length];
            var poly = new Polyline { Stroke = br, StrokeThickness = 1.5 };
            foreach (var (step, vals) in _val)
            {
                if (!vals.TryGetValue(names[i], out var y)) continue;
                poly.Points.Add(new Point(
                    34 + (x1 > x0 ? (step - x0) / (x1 - x0) : 0) * w,
                    h - (y - lo) / (hi - lo) * h));
            }
            CVal.Children.Add(poly);

            var sp = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 12, 0) };
            sp.Children.Add(new Border { Width = 10, Height = 3, Background = br, Margin = new Thickness(0, 7, 4, 0) });
            sp.Children.Add(new TextBlock { Text = names[i], FontSize = 11, Foreground = Brush("#8A90A0") });
            ValLegend.Children.Add(sp);
        }
    }

    // -------------------------------------------------------------- messages

    void LoadMessages()
    {
        var txt = ReadShared(P("messages.jsonl"));
        if (txt is null) return;
        var lines = txt.Split('\n').Where(l => !string.IsNullOrWhiteSpace(l)).ToList();
        if (lines.Count == _msgCount) return;
        _msgCount = lines.Count;

        var items = new List<object>();
        foreach (var line in lines)
        {
            JsonElement m;
            try { m = JsonDocument.Parse(line).RootElement; } catch { continue; }
            string who = Str(m, "from") ?? "?";
            items.Add(new
            {
                Who = $"{who}  {Str(m, "ts")}",
                Text = Str(m, "text") ?? "",
                Bg = who == "claude" ? Brush("#1E2A3D") : Brush("#272B38"),
            });
        }
        MsgList.ItemsSource = items;
        MsgScroll.ScrollToEnd();
    }

    void OnSend(object sender, RoutedEventArgs e) => Send();

    void OnMsgKey(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter) Send();
    }

    void Send()
    {
        var t = MsgBox.Text.Trim();
        if (t.Length == 0) return;
        var rec = JsonSerializer.Serialize(new
        {
            from = "user",
            ts = DateTime.Now.ToString("HH:mm:ss"),
            text = t,
        });
        try { File.AppendAllText(P("messages.jsonl"), rec + "\n"); MsgBox.Clear(); }
        catch (Exception ex) { MessageBox.Show(ex.Message); }
    }

    // -------------------------------------------------------------- controls

    void Command(string cmd)
    {
        try
        {
            File.WriteAllText(P("control.json"),
                JsonSerializer.Serialize(new { command = cmd }));
        }
        catch (Exception ex) { MessageBox.Show(ex.Message); }
    }

    void OnPause(object sender, RoutedEventArgs e) => Command("pause");

    void OnResume(object sender, RoutedEventArgs e) => Command("run");

    void OnStop(object sender, RoutedEventArgs e)
    {
        var r = MessageBox.Show(
            "Stop the run?\n\nThe trainer saves a checkpoint and exits. "
            + "Re-running the same command resumes from that checkpoint with no loss.",
            "Confirm stop", MessageBoxButton.YesNo, MessageBoxImage.Warning);
        if (r == MessageBoxResult.Yes) Command("stop");
    }
}
