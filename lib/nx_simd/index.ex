defmodule NxSimd.Index do
  @moduledoc """
  Symbolic integer index expressions for generated loops.

  A form is `{const, terms}` with `terms` a map from an atom term to an
  integer coefficient. Atom terms are loop variables `{:var, name, dim}`
  (ranging over `0..dim-1`), and `{:div, form, c}` / `{:mod, form, c}` of a
  form by a positive constant (C semantics, which equal floor division and
  modulo for the non-negative forms the generator produces).

  Reshapes turn a flat index into `(flat / stride) % dim` per axis. With the
  loop variable ranges known, `div/2` and `mod/2` remove whatever cannot
  matter (terms that are multiples of the divisor, divisions of values below
  the divisor), so the common reshape chains of FFT stages and framing
  collapse to plain affine indices that the C++ compiler can vectorise.
  """

  @type form :: {integer(), %{optional(term()) => integer()}}

  def const(n) when is_integer(n), do: {n, %{}}
  def var(_name, 1), do: const(0)
  def var(name, dim), do: {0, %{{:var, name, dim} => 1}}

  def zero?({0, t}) when map_size(t) == 0, do: true
  def zero?(_), do: false

  def add({c1, t1}, {c2, t2}),
    do: {c1 + c2, Map.merge(t1, t2, fn _, a, b -> a + b end) |> drop_zero()} |> recombine()

  def add(a, n) when is_integer(n), do: add(a, const(n))

  # k * c * (f / c) + k * (f % c) is k * f: splits and re-joins of a flat
  # index (reshape of a reshape) cancel.
  defp recombine({c, t} = form) do
    Enum.find_value(t, form, fn
      {{:div, f, d} = atom, kd} ->
        km = Kernel.div(kd, d)

        if rem(kd, d) == 0 and Map.get(t, {:mod, f, d}) == km do
          rest = t |> Map.delete(atom) |> Map.delete({:mod, f, d})
          add({c, rest}, mul(f, km))
        end

      _ ->
        nil
    end)
  end

  def mul(_, 0), do: const(0)
  def mul({c, t}, k) when is_integer(k), do: {c * k, Map.new(t, fn {a, v} -> {a, v * k} end)}

  defp drop_zero(t), do: t |> Enum.reject(fn {_, v} -> v == 0 end) |> Map.new()

  @doc "Smallest and largest value of a form."
  def bounds({c, t}) do
    Enum.reduce(t, {c, c}, fn {atom, k}, {lo, hi} ->
      {alo, ahi} = atom_bounds(atom)
      {x, y} = if k >= 0, do: {k * alo, k * ahi}, else: {k * ahi, k * alo}
      {lo + x, hi + y}
    end)
  end

  defp atom_bounds({:var, _, dim}), do: {0, dim - 1}

  defp atom_bounds({:div, f, c}) do
    {lo, hi} = bounds(f)
    {Kernel.div(lo, c), Kernel.div(hi, c)}
  end

  defp atom_bounds({:mod, f, c}) do
    case bounds(f) do
      {lo, hi} when lo >= 0 and hi < c -> {lo, hi}
      {lo, _} when lo >= 0 -> {0, c - 1}
      _ -> {-(c - 1), c - 1}
    end
  end

  @doc "`form / c` (C division by a positive constant), simplified."
  def div(form, 1), do: form

  def div(form, c) when c > 1 do
    {lo, _} = bounds(form)

    with true <- lo >= 0,
         {q, r} <- split(form, c),
         {rlo, rhi} = bounds(r),
         true <- rlo >= 0 do
      if rhi < c, do: q, else: add(q, atom({:div, r, c}))
    else
      _ -> atom({:div, form, c})
    end
  end

  @doc "`form % c` (C remainder by a positive constant), simplified."
  def mod(form, c) when c >= 1 do
    {lo, hi} = bounds(form)

    cond do
      lo >= 0 and hi < c ->
        form

      lo >= 0 ->
        {_, r} = split(form, c)
        {rlo, rhi} = bounds(r)

        cond do
          rlo >= 0 and rhi < c -> r
          rlo >= 0 -> atom({:mod, r, c})
          true -> atom({:mod, form, c})
        end

      true ->
        atom({:mod, form, c})
    end
  end

  # form = c * quotient + rest, taking out every multiple of c.
  defp split({k, t}, c) do
    {q, r} =
      Enum.reduce(t, {%{}, %{}}, fn {a, v}, {q, r} ->
        if rem(v, c) == 0, do: {Map.put(q, a, Kernel.div(v, c)), r}, else: {q, Map.put(r, a, v)}
      end)

    {{Integer.floor_div(k, c), q}, {Integer.mod(k, c), r}}
  end

  defp atom(a), do: {0, %{a => 1}}

  @doc "C expression for a form (all values `int64_t`)."
  def render({c, t}) do
    terms =
      t
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map(fn
        {a, 1} -> render_atom(a)
        {a, k} -> "#{render_atom(a)} * #{k}"
      end)

    case {terms, c} do
      {[], c} -> Integer.to_string(c)
      {terms, 0} -> Enum.join(terms, " + ")
      {terms, c} -> Enum.join(terms, " + ") <> " + #{c}"
    end
    |> String.replace("+ -", "- ")
  end

  defp render_atom({:var, name, _}), do: name
  defp render_atom({:div, f, c}), do: "((#{render(f)}) / #{c})"
  defp render_atom({:mod, f, c}), do: "((#{render(f)}) % #{c})"

  @doc "Value of a form for `env` (`%{name => integer}`), with C semantics."
  def eval({c, t}, env) do
    Enum.reduce(t, c, fn {a, k}, acc -> acc + k * eval_atom(a, env) end)
  end

  defp eval_atom({:var, name, _}, env), do: Map.fetch!(env, name)
  defp eval_atom({:div, f, c}, env), do: Kernel.div(eval(f, env), c)
  defp eval_atom({:mod, f, c}, env), do: Kernel.rem(eval(f, env), c)
end
