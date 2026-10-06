using System.Runtime.InteropServices;
using System.Text;

namespace Kestrel.Engine;

/// <summary>One tensor's directory entry. <see cref="Shape"/> is numpy/PyTorch order
/// (slowest-varying first); GGUF itself stores dimensions fastest-first.</summary>
public sealed record GgufTensor(string Name, long[] Shape, uint Type, ulong Offset)
{
    public long Count => Shape.Aggregate(1L, (a, b) => a * b);
}

/// <summary>Minimal GGUF v2/v3 reader: metadata, tensor directory, and tensor data
/// dequantized to float32 (F32, F16, BF16 and Q8_0 are supported).</summary>
public sealed class GgufFile
{
    public const uint F32 = 0, F16 = 1, Q8_0 = 8, BF16 = 30;

    public string Path { get; }
    public uint Version { get; }
    public IReadOnlyDictionary<string, object> Metadata => _meta;
    public IReadOnlyDictionary<string, GgufTensor> Tensors => _tensors;

    readonly Dictionary<string, object> _meta = new();
    readonly Dictionary<string, GgufTensor> _tensors = new();
    readonly long _dataStart;

    GgufFile(string path)
    {
        Path = path;
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);
        if (Encoding.ASCII.GetString(br.ReadBytes(4)) != "GGUF")
            throw new InvalidDataException($"{path} is not a GGUF file");
        Version = br.ReadUInt32();
        if (Version < 2) throw new InvalidDataException($"GGUF v{Version} is not supported");
        ulong nTensors = br.ReadUInt64(), nKv = br.ReadUInt64();

        for (ulong i = 0; i < nKv; i++)
        {
            string key = ReadString(br);
            _meta[key] = ReadValue(br, br.ReadUInt32());
        }
        for (ulong i = 0; i < nTensors; i++)
        {
            string name = ReadString(br);
            uint nd = br.ReadUInt32();
            var dims = new long[nd];
            for (int d = 0; d < nd; d++) dims[d] = (long)br.ReadUInt64();
            Array.Reverse(dims);                       // -> numpy order
            uint type = br.ReadUInt32();
            ulong off = br.ReadUInt64();
            _tensors[name] = new GgufTensor(name, dims, type, off);
        }
        long align = Metadata.TryGetValue("general.alignment", out var a) ? Convert.ToInt64(a) : 32;
        _dataStart = (fs.Position + align - 1) / align * align;
    }

    public static GgufFile Open(string path) => new(path);

    // ---------------------------------------------------------------- metadata

    public string Architecture => GetString("general.architecture");
    public bool Has(string key) => _meta.ContainsKey(key);
    public string GetString(string key) => (string)_meta[key];
    public int GetInt(string key) => Convert.ToInt32(_meta[key]);
    public float GetFloat(string key) => Convert.ToSingle(_meta[key]);
    public bool GetBool(string key) => Convert.ToBoolean(_meta[key]);
    public string[] GetStrings(string key) => ((object[])_meta[key]).Cast<string>().ToArray();
    public int[] GetInts(string key) => ((object[])_meta[key]).Select(Convert.ToInt32).ToArray();

    public string GetString(string key, string fallback) =>
        _meta.TryGetValue(key, out var v) ? (string)v : fallback;
    public int GetInt(string key, int fallback) =>
        _meta.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;
    public float GetFloat(string key, float fallback) =>
        _meta.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;

    static string ReadString(BinaryReader br)
    {
        ulong n = br.ReadUInt64();
        return Encoding.UTF8.GetString(br.ReadBytes(checked((int)n)));
    }

    static object ReadValue(BinaryReader br, uint type) => type switch
    {
        0 => br.ReadByte(),
        1 => br.ReadSByte(),
        2 => br.ReadUInt16(),
        3 => br.ReadInt16(),
        4 => br.ReadUInt32(),
        5 => br.ReadInt32(),
        6 => br.ReadSingle(),
        7 => br.ReadByte() != 0,
        8 => ReadString(br),
        9 => ReadArray(br),
        10 => br.ReadUInt64(),
        11 => br.ReadInt64(),
        12 => br.ReadDouble(),
        _ => throw new InvalidDataException($"unknown GGUF metadata type {type}")
    };

    static object[] ReadArray(BinaryReader br)
    {
        uint et = br.ReadUInt32();
        ulong n = br.ReadUInt64();
        var arr = new object[n];
        for (ulong i = 0; i < n; i++) arr[i] = ReadValue(br, et);
        return arr;
    }

    // ---------------------------------------------------------------- tensors

    /// <summary>Read a tensor as float32 regardless of its storage type.</summary>
    public float[] ReadFloats(string name)
    {
        if (!_tensors.TryGetValue(name, out var t))
            throw new KeyNotFoundException($"tensor '{name}' not in {System.IO.Path.GetFileName(Path)}");
        long n = t.Count;
        long bytes = t.Type switch
        {
            F32 => n * 4,
            F16 or BF16 => n * 2,
            Q8_0 => n / 32 * 34,
            _ => throw new NotSupportedException($"tensor '{name}': GGUF type {t.Type} not supported")
        };
        var raw = new byte[bytes];
        using (var fs = new FileStream(Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20))
        {
            fs.Seek(_dataStart + (long)t.Offset, SeekOrigin.Begin);
            fs.ReadExactly(raw);
        }
        var dst = new float[n];
        switch (t.Type)
        {
            case F32:
                MemoryMarshal.Cast<byte, float>(raw).CopyTo(dst);
                break;
            case F16:
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    MemoryMarshal.Cast<byte, Half>(raw), dst);
                break;
            case BF16:
            {
                var src = MemoryMarshal.Cast<byte, ushort>(raw);
                var di = MemoryMarshal.Cast<float, uint>(dst.AsSpan());
                for (int i = 0; i < src.Length; i++) di[i] = (uint)src[i] << 16;
                break;
            }
            case Q8_0:
                for (long b = 0; b < n / 32; b++)
                {
                    int o = (int)(b * 34);
                    float d = (float)BitConverter.ToHalf(raw, o);
                    for (int j = 0; j < 32; j++) dst[b * 32 + j] = d * (sbyte)raw[o + 2 + j];
                }
                break;
        }
        return dst;
    }
}
