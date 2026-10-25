defmodule NxSimd.CompilerTest do
  @moduledoc """
  NxSimd.Compiler (generated C++) against the Nx evaluator: bit-identical
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

  defn gather_computed(a, i) do
    # a: s16[3][4], i: u8[6]; the table is computed and reshaped
    table = (Nx.as_type(a, :s32) * 3 - 1) |> Nx.reshape({12})
    Nx.take(table, Nx.clip(Nx.as_type(i, :s32), 0, 11))
  end

  defn unsigned(a, k) do
    # a: u16[4], k: u8 shift amounts
    {Nx.left_shift(Nx.as_type(a, :u32), Nx.bitwise_and(k, 7)),
     Nx.right_shift(a, Nx.bitwise_and(k, 7)), a == 7}
  end

  defn narrowing(a, b, p) do
    # a, b: s16[5], p: u32[5]. The s64 chain is masked to 32 bits (so the
    # generator may compute it in 32 bits), then used exactly (select, shifts).
    w = Nx.as_type(a, :s64) * Nx.as_type(p, :s64) + Nx.as_type(b, :s64)
    low = Nx.bitwise_and(w, Nx.s64(0xFFFFFFFF))
    hi = Nx.right_shift(low, 16)
    m = Nx.bitwise_and(w * 3 - 5, 0xFFF)
    sel = Nx.select(a > b, Nx.as_type(hi, :s32) - 7, Nx.negate(Nx.as_type(m, :s32)))

    {low, hi, m, sel, Nx.left_shift(Nx.as_type(m, :s32), 4), Nx.right_shift(w, 20),
     Nx.min(Nx.as_type(low, :s64), w)}
  end

  defn staged(a, b, table) do
    # a, b: s16[64][128], table: s16[1024]: dependent loops over buffers that
    # the generator splits across threads (concatenations are buffers), a
    # table lookup and a reduction
    x = Nx.as_type(a, :s32) + Nx.as_type(b, :s32) * 2
    c1 = Nx.concatenate([x, x + 1], axis: 0)
    c2 = Nx.concatenate([c1 * 3, c1 - 5], axis: 1)
    c3 = Nx.concatenate([Nx.right_shift(c2, 1), c2], axis: 0)
    g = Nx.take(table, Nx.bitwise_and(Nx.as_type(c3, :s32), 1023))
    {c3, Nx.sum(c3, axes: [0]), Nx.transpose(g)}
  end

  defp both(fun, args, opts \\ []) do
    ref = apply(fun, args)
    jit = Nx.Defn.jit(fun, [compiler: NxSimd.Compiler, fallback: false] ++ opts)
    assert flat(apply(jit, args)) == flat(ref)
  end

  defp pseudo_random(shape, type, seed) do
    {lo, hi} = NxFixed.WidthCheck.type_range(type)
    n = Tuple.product(shape)

    Nx.iota({n}, type: :s64)
    |> Nx.multiply(7919)
    |> Nx.add(seed)
    |> Nx.remainder(hi - lo + 1)
    |> Nx.add(lo)
    |> Nx.as_type(type)
    |> Nx.reshape(shape)
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
      both(&gather_computed/2, [a, i])
      both(&unsigned/2, [u, k])
    end
  end

  property "ring chains computed at narrower widths are exact" do
    check all a <- ints({:s, 16}, {5}),
              b <- ints({:s, 16}, {5}),
              p <- ints({:u, 32}, {5}),
              max_runs: 40 do
      both(&narrowing/3, [a, b, p])
    end
  end

  test "threaded kernels give the same result as single-threaded ones" do
    a = pseudo_random({64, 128}, {:s, 16}, 1)
    b = pseudo_random({64, 128}, {:s, 16}, 5)

    table = pseudo_random({1024}, {:s, 16}, 9)

    for threads <- [1, 2, 3, 4], do: both(&staged/3, [a, b, table], threads: threads)

    expr = Nx.Defn.debug_expr(&staged/3).(a, b, table)
    refute NxSimd.Compiler.source(expr, threads: 1) =~ "omp parallel"
    assert NxSimd.Compiler.source(expr, threads: 4) =~ "omp parallel"
  end

  defn overflowing(a, b), do: a * b

  test "graphs that may overflow are refused, not silently wrapped" do
    jit = Nx.Defn.jit(&overflowing/2, compiler: NxSimd.Compiler)

    assert_raise NxFixed.RangeError, ~r/multiply needs/, fn ->
      jit.(Nx.s16(300), Nx.s16(300))
    end
  end

  defn float_graph(x), do: Nx.sin(x) * 2

  test "float graphs fall back to the evaluator, or raise with fallback: false" do
    x = Nx.tensor([0.0, 1.0], type: :f32)

    assert Nx.Defn.jit(&float_graph/1, compiler: NxSimd.Compiler).(x) |> Nx.to_list() ==
             float_graph(x) |> Nx.to_list()

    assert_raise NxSimd.Codegen.Unsupported, fn ->
      Nx.Defn.jit(&float_graph/1, compiler: NxSimd.Compiler, fallback: false).(x)
    end
  end
end
