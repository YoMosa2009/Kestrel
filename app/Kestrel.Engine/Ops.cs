using System.Numerics.Tensors;

namespace Kestrel.Engine;

/// <summary>CPU kernels. Everything is float32; tensors are flat row-major arrays,
/// activations are (T, dim) with T tokens processed together (T = 1 when decoding).</summary>
internal static class Ops
{
    /// <summary>Y[t, o] = Σ_i W[o, i] X[t, i] (+ b[o]) — a PyTorch nn.Linear over T tokens.
    /// Parallel over output rows; each weight row is reused across all T tokens while it is
    /// hot in cache, which is what makes prompt prefill much faster per token than decode.</summary>
    public static void Linear(float[] w, float[]? bias, int outDim, int inDim,
                              float[] x, int xOff, float[] y, int yOff, int t)
    {
        const int Rows = 16;
        int blocks = (outDim + Rows - 1) / Rows;
        if ((long)outDim * inDim * t < 32_768)
        {
            for (int b = 0; b < blocks; b++) LinearRows(b * Rows, Math.Min(outDim, b * Rows + Rows));
            return;
        }
        Parallel.For(0, blocks, b => LinearRows(b * Rows, Math.Min(outDim, b * Rows + Rows)));

        void LinearRows(int r0, int r1)
        {
            for (int o = r0; o < r1; o++)
            {
                var row = new ReadOnlySpan<float>(w, o * inDim, inDim);
                float bo = bias?[o] ?? 0f;
                for (int tt = 0; tt < t; tt++)
                    y[yOff + tt * outDim + o] =
                        TensorPrimitives.Dot(row, new ReadOnlySpan<float>(x, xOff + tt * inDim, inDim)) + bo;
            }
        }
    }

    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> w, Span<float> y, float eps = 1e-6f)
    {
        float ms = TensorPrimitives.SumOfSquares(x) / x.Length;
        float inv = 1f / MathF.Sqrt(ms + eps);
        TensorPrimitives.Multiply(x, inv, y);
        if (!w.IsEmpty) TensorPrimitives.Multiply(y, w, y);
    }

    /// <summary>nn.LayerNorm(dim) with affine weight and bias, eps 1e-5.</summary>
    public static void LayerNorm(Span<float> x, ReadOnlySpan<float> w, ReadOnlySpan<float> b, float eps = 1e-5f)
    {
        float mean = TensorPrimitives.Sum(x) / x.Length;
        TensorPrimitives.Subtract(x, mean, x);
        float var = TensorPrimitives.SumOfSquares(x) / x.Length;
        TensorPrimitives.Multiply(x, 1f / MathF.Sqrt(var + eps), x);
        TensorPrimitives.Multiply(x, w, x);
        TensorPrimitives.Add(x, b, x);
    }

    /// <summary>F.normalize(x, dim=-1): x / max(‖x‖₂, 1e-12).</summary>
    public static void L2Normalize(Span<float> x)
    {
        float n = TensorPrimitives.Norm(x);
        TensorPrimitives.Multiply(x, 1f / MathF.Max(n, 1e-12f), x);
    }

    public static float Silu(float v) => v / (1f + MathF.Exp(-v));

    public static void SiluInPlace(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = Silu(x[i]);
    }

    /// <summary>torch softplus (beta 1, threshold 20).</summary>
    public static float Softplus(float v) => v > 20f ? v : MathF.Log(1f + MathF.Exp(v));

    public static void SoftmaxInPlace(Span<float> x)
    {
        float max = TensorPrimitives.Max(x);
        float sum = 0f;
        for (int i = 0; i < x.Length; i++) { x[i] = MathF.Exp(x[i] - max); sum += x[i]; }
        TensorPrimitives.Multiply(x, 1f / sum, x);
    }

    /// <summary>Indices of the k largest values, descending.</summary>
    public static void TopK(ReadOnlySpan<float> x, int k, Span<int> idx, Span<float> val)
    {
        int n = 0;
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            if (n == k && v <= val[k - 1]) continue;
            int p = n < k ? n++ : k - 1;
            while (p > 0 && val[p - 1] < v) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; }
            val[p] = v; idx[p] = i;
        }
    }
}
