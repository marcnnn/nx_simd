defmodule ExHdl.SIMDCompilerTest do
  @moduledoc """
  ExHdl.SIMD.Compiler (generated C++) against the Nx evaluator: bit-identical
  for every supported operation, with broadcasting, views and reductions;
  refuses graphs that may overflow; falls back for floats.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Defn

  @moduletag timeout: 600_000

  defn elementwise(a, b) do
    # a: s16[3][4], b: s16[4] (broadcast over rows)
    x = Nx.as_type(a, :s32) * 3 - Nx.as_type(b, :s32)
    y = Nx.select(x > 0, Nx.right_shift(x, 2), Nx.negate(Nx.abs(x)))

    {Nx.clip(y, -1000, 1000), Nx.min(x, 7), Nx.max(x, -7), Nx.bitwise_xor(a, b),
     Nx.bitwise_and(a, 255)}
  end

  defn views(a) do
    # a: s16[3][4]
    t = Nx.transpose(a)
    r = Nx.reshape(a, {4, 3}) |> Nx.as_type(:s32)
    s = Nx.reshape(a, {1, 12}) |> Nx.squeeze()
    b = Nx.broadcast(Nx.reshape(Nx.slice_along_axis(a, 0, 1, axis: 0), {4}), {2, 4}, axes: [1])
    {t, r + 1, s, b}
  end

  defn reductions(a) do
    a = Nx.as_type(a, :s32)
    {Nx.sum(a), Nx.sum(a, axes: [0]), Nx.sum(a, axes: [1])}
  end

  defn dots(a, b, v) do
    # a: s8[3][5], b: s8[5][2], v: s8[5]
    a = Nx.as_type(a, :s32)

    {Nx.dot(a, Nx.as_type(b, :s32)), Nx.dot(a, [1], Nx.as_type(Nx.transpose(b), :s32), [1]),
     Nx.dot(a, Nx.as_type(v, :s32))}
  end

  defn lookup_and_division(i, x) do
    # i: u8[6] index, x: s32[6]
    table = Nx.tensor([5, -3, 100, 7, -128, 0, 42, 9], type: :s16)

    {Nx.take(table, Nx.bitwise_and(i, 7)), Nx.quotient(Nx.as_type(x, :s64), 7),
     Nx.remainder(Nx.as_type(x, :s64), -5)}
  end

  defn unsigned(a, k) do
    # a: u16[4], k: u8 shift amounts
    {Nx.left_shift(Nx.as_type(a, :u32), Nx.bitwise_and(k, 7)),
     Nx.right_shift(a, Nx.bitwise_and(k, 7)), a == 7}
  end

  defp both(fun, args) do
    ref = apply(fun, args)
    got = apply(Nx.Defn.jit(fun, compiler: ExHdl.SIMD.Compiler, fallback: false), args)
    assert flat(got) == flat(ref)
  end

  defp flat(container),
    do:
      [container]
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.map(&{Nx.type(&1), Nx.shape(&1), Nx.to_flat_list(&1)})

  defp ints(type, shape) do
    {_, bits} = type

    {lo, hi} =
      if elem(type, 0) == :s,
        do: {-Bitwise.bsl(1, bits - 1), Bitwise.bsl(1, bits - 1) - 1},
        else: {0, Bitwise.bsl(1, bits) - 1}

    n = shape |> Tuple.to_list() |> Enum.product()

    gen all values <- list_of(one_of([integer(lo..hi), member_of([lo, hi, 0])]), length: n) do
      values |> Nx.tensor(type: type) |> Nx.reshape(shape)
    end
  end

  property "every supported operation is bit-identical to the evaluator" do
    check all a <- ints({:s, 16}, {3, 4}),
              b <- ints({:s, 16}, {4}),
              a8 <- ints({:s, 8}, {3, 5}),
              b8 <- ints({:s, 8}, {5, 2}),
              v8 <- ints({:s, 8}, {5}),
              i <- ints({:u, 8}, {6}),
              x <- ints({:s, 32}, {6}),
              u <- ints({:u, 16}, {4}),
              k <- ints({:u, 8}, {4}),
              max_runs: 25 do
      both(&elementwise/2, [a, b])
      both(&views/1, [a])
      both(&reductions/1, [a])
      both(&dots/3, [a8, b8, v8])
      both(&lookup_and_division/2, [i, x])
      both(&unsigned/2, [u, k])
    end
  end

  defn overflowing(a, b), do: a * b

  test "graphs that may overflow are refused, not silently wrapped" do
    jit = Nx.Defn.jit(&overflowing/2, compiler: ExHdl.SIMD.Compiler)

    assert_raise ExHdl.DSP.RangeError, ~r/multiply needs/, fn ->
      jit.(Nx.s16(300), Nx.s16(300))
    end
  end

  defn float_graph(x), do: Nx.sin(x) * 2

  test "float graphs fall back to the evaluator, or raise with fallback: false" do
    x = Nx.tensor([0.0, 1.0], type: :f32)

    assert Nx.Defn.jit(&float_graph/1, compiler: ExHdl.SIMD.Compiler).(x) |> Nx.to_list() ==
             float_graph(x) |> Nx.to_list()

    assert_raise ExHdl.SIMD.Codegen.Unsupported, fn ->
      Nx.Defn.jit(&float_graph/1, compiler: ExHdl.SIMD.Compiler, fallback: false).(x)
    end
  end
end
