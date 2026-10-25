defmodule NxSimd.IndexTest do
  @moduledoc """
  Simplified index forms (division and modulo removed where the loop ranges
  allow) evaluate like the unsimplified C expressions at every point.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias NxSimd.Index

  defp envs(dims) do
    ranges = Enum.map(dims, &Enum.to_list(0..(&1 - 1)))

    Enum.reduce(Enum.with_index(ranges), [%{}], fn {r, i}, acc ->
      for env <- acc, v <- r, do: Map.put(env, "v#{i}", v)
    end)
  end

  defp form(dims, coefs, const) do
    dims
    |> Enum.zip(coefs)
    |> Enum.with_index()
    |> Enum.reduce(Index.const(const), fn {{dim, k}, i}, acc ->
      Index.add(acc, Index.mul(Index.var("v#{i}", dim), k))
    end)
  end

  property "div and mod match C semantics" do
    check all(
            dims <- list_of(integer(1..9), min_length: 1, max_length: 3),
            coefs <- list_of(integer(0..40), length: length(dims)),
            const <- integer(0..30),
            c <- integer(1..24)
          ) do
      f = form(dims, coefs, const)

      for env <- envs(dims) do
        v = Index.eval(f, env)
        assert Index.eval(Index.div(f, c), env) == Kernel.div(v, c)
        assert Index.eval(Index.mod(f, c), env) == Kernel.rem(v, c)
      end
    end
  end

  property "reshape chains (flat, split, flat again) evaluate like the original" do
    check all(
            dims <- list_of(integer(1..8), min_length: 1, max_length: 3),
            a <- integer(1..6),
            b <- integer(1..6)
          ) do
      # flat index of the loop nest, re-split as [a, b, rest] and flattened again
      strides =
        Enum.scan(Enum.reverse(dims), 1, &(&1 * &2))
        |> then(&[1 | &1])
        |> Enum.take(length(dims))
        |> Enum.reverse()

      flat = form(dims, strides, 0)
      total = Enum.product(dims)
      d1 = Index.mod(Index.div(flat, a * b), max(div(total, a * b), 1) + 1)
      d2 = Index.mod(Index.div(flat, b), a)
      d3 = Index.mod(flat, b)
      back = d1 |> Index.mul(a * b) |> Index.add(Index.mul(d2, b)) |> Index.add(d3)

      for env <- envs(dims), do: assert(Index.eval(back, env) == Index.eval(flat, env))
    end
  end

  test "reshape of an aligned loop nest becomes affine" do
    flat = form([16, 1, 1024], [2048, 2048, 1], 1024)
    refute Index.render(flat) =~ "/"
    inner = flat |> Index.div(2048) |> Index.mod(16)
    assert Index.render(inner) == "v0"
    assert Index.render(Index.mod(flat, 2048)) == "v2 + 1024"
  end
end
