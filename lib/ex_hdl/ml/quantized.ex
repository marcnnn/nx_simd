defmodule ExHdl.ML.Quantized do
  @moduledoc """
  Quantized (int8) neural network building blocks with TensorFlow Lite
  semantics, written as `defn` so they run on the Nx evaluator, EXLA or
  `ExHdl.SIMD.Compiler` with identical bits.

  Follows the TFLite int8 quantization spec: activations int8 per tensor
  with a zero point, weights int8 per output channel (zero point 0,
  values in -127..127), bias int32 with scale `input_scale * weight_scale`.
  Requantization uses TFLite's default double rounding (gemmlowp):
  `SaturatingRoundingDoublingHighMul` followed by `RoundingDivideByPOT`,
  both rounding ties away from zero.

  Sources: tensorflow/lite/kernels/internal/{quantization_util.cc,common.cc},
  reference/integer_ops/fully_connected.h, gemmlowp fixedpoint/fixedpoint.h.
  """
  import Nx.Defn
  import Bitwise

  # -- host side -------------------------------------------------------------------

  @doc """
  TFLite `QuantizeMultiplier`: a real multiplier as `{multiplier, shift}`
  with `real = multiplier * 2^(shift - 31)` and `multiplier` in
  `2^30..2^31-1` (or 0).
  """
  def quantize_multiplier(real) when real == 0, do: {0, 0}

  def quantize_multiplier(real) when is_float(real) and real > 0 do
    {f, shift} = frexp(real)
    # f * 2^31 is exact in a double; round/1 rounds half away from zero like TfLiteRound
    q = round(f * 2_147_483_648.0)
    {q, shift} = if q == 1 <<< 31, do: {div(q, 2), shift + 1}, else: {q, shift}
    if shift < -31, do: {0, 0}, else: {q, shift}
  end

  def quantize_multiplier(real) when is_integer(real), do: quantize_multiplier(real * 1.0)

  # frexp for positive normal doubles: real = f * 2^e with f in [0.5, 1).
  defp frexp(x) do
    <<0::1, exp::11, mantissa::52>> = <<x::float-64>>
    if exp == 0, do: raise(ArgumentError, "subnormal multipliers are not supported")
    f = Bitwise.bor(mantissa, 1 <<< 52) / (1 <<< 53)
    {f, exp - 1022}
  end

  @doc """
  Precomputes a fully connected layer from real scales (TFLite style).
  `weights` is `[[w]]` with one row per output channel, `weight_scales`
  one scale per output channel (or a single number).
  """
  def fully_connected_params(opts) do
    opts =
      Keyword.validate!(opts, [
        :weights,
        :bias,
        :input_scale,
        :input_zero_point,
        :weight_scales,
        :output_scale,
        :output_zero_point,
        activation: :none
      ])

    weights = opts[:weights]
    channels = length(weights)

    w_scales =
      List.wrap(opts[:weight_scales])
      |> then(&if length(&1) == 1, do: List.duplicate(hd(&1), channels), else: &1)

    {mults, shifts} =
      w_scales
      |> Enum.map(&quantize_multiplier(opts[:input_scale] * &1 / opts[:output_scale]))
      |> Enum.unzip()

    out_zp = opts[:output_zero_point]
    quant = fn real -> out_zp + round(real / opts[:output_scale]) end

    {act_min, act_max} =
      case opts[:activation] do
        :none -> {-128, 127}
        :relu -> {max(-128, quant.(0.0)), 127}
        :relu6 -> {max(-128, quant.(0.0)), min(127, quant.(6.0))}
      end

    # Parameters are tensors, not lists: Nx (and EXLA) key their compile
    # cache on the jitted closure, which captures these values, and hashing
    # a binary is far cheaper than hashing a 32k-element list on every call.
    zp = opts[:input_zero_point]
    bias = opts[:bias] || List.duplicate(0, channels)

    # sum_k w * (x - zp) = sum_k w * x - zp * sum_k w: folding the zero point
    # into the bias (exact, as in TFLite's optimised kernels) keeps the inner
    # product int8 x int8, which maps to widening dot-product instructions.
    folded = Enum.zip_with(bias, weights, fn b, row -> b - zp * Enum.sum(row) end)

    shift =
      case Enum.uniq(shifts) do
        [single] -> single
        _ -> Nx.tensor(shifts, type: :s32, backend: Nx.BinaryBackend)
      end

    [
      weights: Nx.tensor(weights, type: :s8, backend: Nx.BinaryBackend),
      bias: Nx.tensor(folded, type: :s32, backend: Nx.BinaryBackend),
      output_zero_point: out_zp,
      multiplier: Nx.tensor(mults, type: :s32, backend: Nx.BinaryBackend),
      shift: shift,
      activation_min: act_min,
      activation_max: act_max
    ]
  end

  # -- requantization (inside defn) ---------------------------------------------------

  @doc """
  gemmlowp `SaturatingRoundingDoublingHighMul(x, m)`: `x * m / 2^31` rounded
  half away from zero. `x` is int32, `m` a multiplier from
  `quantize_multiplier/1` (never -2^31, so the saturating case cannot occur).
  """
  deftransform saturating_rounding_doubling_high_mul(x, m) do
    prod = Nx.multiply(Nx.as_type(x, :s64), Nx.as_type(m, :s64))
    nudge = Nx.select(Nx.greater_equal(prod, 0), 1 <<< 30, 1 - (1 <<< 30))
    # Explicitly s64: Nx infers s32 for integer literals and silently wraps
    # 2^31 to -2^31 (Nx.tensor(2147483648) is -2147483648).
    two_31 = Nx.tensor(1 <<< 31, type: :s64, backend: Nx.BinaryBackend)
    prod |> Nx.add(nudge) |> Nx.quotient(two_31) |> Nx.as_type(:s32)
  end

  @doc "gemmlowp `RoundingDivideByPOT(x, e)`: `x / 2^e` rounded half away from zero, `e` in 0..31."
  deftransform rounding_divide_by_pot(x, e) do
    mask = Nx.subtract(Nx.left_shift(Nx.tensor(1, type: :s32), e), 1)
    remainder = Nx.bitwise_and(x, mask)
    threshold = Nx.add(Nx.right_shift(mask, 1), Nx.select(Nx.less(x, 0), 1, 0))
    Nx.add(Nx.right_shift(x, e), Nx.select(Nx.greater(remainder, threshold), 1, 0))
  end

  @doc """
  TFLite `MultiplyByQuantizedMultiplier` (default double rounding). Shifts
  above 0 shift the accumulator left first; `ExHdl.Passes.WidthCheck`
  rejects the graph if that could wrap.
  """
  deftransform multiply_by_quantized_multiplier(acc, multiplier, shift) do
    {left, right} = split_shift(shift)
    acc = if left == 0, do: acc, else: Nx.left_shift(acc, left)
    acc |> saturating_rounding_doubling_high_mul(multiplier) |> rounding_divide_by_pot(right)
  end

  defp split_shift(shift) when is_integer(shift), do: {max(shift, 0), max(-shift, 0)}

  defp split_shift(%Nx.Tensor{} = shifts) do
    if Nx.to_number(Nx.reduce_max(shifts)) > 0,
      do: raise(ArgumentError, "per-channel left shifts are not supported")

    {0, Nx.negate(shifts)}
  end

  # -- layers -------------------------------------------------------------------------

  @doc """
  int8 fully connected layer (TFLite `FullyConnectedPerChannel`):
  `x` is `s8[batch][in]`; the options come from `fully_connected_params/1`
  and are compile-time constants (weights `s8[out][in]`, the input zero
  point folded into the bias). Returns `s8[batch][out]`.
  """
  defn fully_connected(x, opts \\ []) do
    opts =
      keyword!(opts, [
        :weights,
        :bias,
        :output_zero_point,
        :multiplier,
        :shift,
        :activation_min,
        :activation_max
      ])

    fc(x, opts)
  end

  # The input zero point is already folded into the bias.
  deftransformp fc(x, opts) do
    x
    |> Nx.as_type(:s32)
    |> Nx.dot([1], Nx.as_type(opts[:weights], :s32), [1])
    |> Nx.add(opts[:bias])
    |> multiply_by_quantized_multiplier(opts[:multiplier], opts[:shift])
    |> Nx.add(opts[:output_zero_point])
    |> Nx.clip(opts[:activation_min], opts[:activation_max])
    |> Nx.as_type(:s8)
  end
end
