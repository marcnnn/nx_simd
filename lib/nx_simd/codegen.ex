defmodule NxSimd.Codegen do
  @moduledoc """
  Generates C++ for an integer `NxFixed.Dataflow` graph.

  The generated library exports

      extern "C" void nx_simd_kernel(const void* const* in, void* const* out);

  with all shapes and types compiled in. `in` holds the parameters in
  order, `out` the outputs (flattened).

  ## Structure

    * Parameters, constant tensors, `dot`/`sum` results and outputs are
      buffers. Everything else (elementwise ops, `as_type`, views such as
      `broadcast`/`reshape`/`transpose`, table lookups) is fused into the
      loop that computes the next buffer, so a chain like
      `dot -> + bias -> requantize -> clamp -> as_type` is one loop.
    * Loops are plain C++ with `#pragma omp simd` on the innermost,
      contiguous dimension; the C++ compiler vectorises them (the same split
      as TVM/LLVM: generated loops plus compiler vectorisation).
    * Index expressions are symbolic (`NxSimd.Index`): reshapes of aligned
      loop nests reduce to affine indices instead of `/` and `%`.
    * Values are computed in the narrowest type their proven range allows,
      and chains that only feed ring operations (e.g. a 64-bit phase
      accumulator masked to 32 bits) at the width their consumers look at.
    * Local buffers share one scratch allocation (lifetime based reuse).
    * With `threads: n` (n > 1) all loops sit in one OpenMP parallel region:
      large loops are split statically across threads, barriers appear only
      between loops that depend on each other. Each element is computed by
      one thread, so the result does not depend on the thread count.
    * `dot` uses i-k-j order (vectorised over the output row) for
      `[m, k] x [k, n]`, and an inner reduction otherwise.

  ## Exactness

  Every node's values fit its type (`NxFixed.WidthCheck` must have
  accepted the graph), so elementwise arithmetic in the node type cannot
  overflow. Accumulations (`dot`, `sum`) run in the unsigned type of the
  result: partial sums may wrap, but the final value equals the exact sum
  modulo 2^bits and the exact sum fits, so the result is exact and the C++
  has no undefined behaviour. Compiled as C++20 (defined shifts of negative
  values and modular unsigned-to-signed conversion).
  """

  alias NxFixed.Dataflow
  alias NxSimd.Index

  defmodule Unsupported do
    defexception [:message]
  end

  @binary ~w(add subtract multiply min max bitwise_and bitwise_or bitwise_xor left_shift right_shift quotient remainder)a
  @ring ~w(add subtract multiply bitwise_and bitwise_or bitwise_xor)a
  @plan_key {__MODULE__, :plan}
  @idx_key {__MODULE__, :index_type}
  @compare ~w(equal not_equal less less_equal greater greater_equal)a
  @logical ~w(logical_and logical_or)a
  @unary ~w(negate abs as_type logical_not bitwise_not)a
  @views ~w(broadcast reshape squeeze transpose slice)a

  @doc """
  Returns `%{source: cpp, inputs: [{type, shape}], outputs: [{type, shape}]}`
  or raises `Unsupported`. Option: `:threads` (default 1).
  """
  def generate(%Dataflow{} = g, opts \\ []) do
    threads = Keyword.get(opts, :threads, 1)
    Enum.each(g.order, fn id -> check_node!(g, g.nodes[id]) end)

    outputs = Dataflow.output_ids(g)
    params = for id <- g.order, g.nodes[id].op == :parameter, do: id
    params = Enum.sort_by(params, &g.nodes[&1].attrs)

    buffers = buffer_nodes(g, outputs)
    Process.put(@plan_key, plan(g, buffers))

    # Loop variables are 32-bit when every buffer is addressable with them.
    big = g.nodes |> Map.values() |> Enum.any?(&(size(&1.shape) >= 1_073_741_824))
    Process.put(@idx_key, if(big, do: "int64_t", else: "int32_t"))

    # Where each buffer lives: an input pointer, an output pointer, a static
    # constant array, or a local array.
    homes =
      Enum.reduce(Enum.with_index(params), %{}, fn {id, i}, acc ->
        Map.put(acc, id, {:input, i})
      end)

    {homes, _} =
      outputs
      |> Enum.with_index()
      |> Enum.reduce({homes, MapSet.new()}, fn {id, i}, {acc, seen} ->
        cond do
          # each output gets its own pointer; repeated or input outputs are copied
          MapSet.member?(seen, id) or Map.has_key?(acc, id) -> {acc, seen}
          true -> {Map.put(acc, id, {:output, i}), MapSet.put(seen, id)}
        end
      end)

    homes =
      Enum.reduce(buffers, homes, fn id, acc ->
        case {Map.has_key?(acc, id), g.nodes[id].op} do
          {true, _} -> acc
          {false, :tensor} -> Map.put(acc, id, :constant)
          {false, _} -> Map.put(acc, id, :local)
        end
      end)

    # Compute every buffer that is not an input/constant, in topological order.
    computed =
      for id <- g.order,
          id in buffers,
          homes[id] not in [:constant] and not match?({:input, _}, homes[id]),
          do: id

    {offsets, arena_size} = arena(g, computed, buffers, homes)
    decls = Enum.map(buffers, &declare(g, &1, homes[&1], offsets[&1]))

    arena_decl =
      if arena_size > 0,
        do: """
          // scratch space for the local buffers, kept per thread between calls
          static thread_local char* nxs_scratch = nullptr;
          static thread_local size_t nxs_scratch_size = 0;
          if (nxs_scratch_size < #{arena_size}) {
            std::free(nxs_scratch);
            nxs_scratch = static_cast<char*>(std::aligned_alloc(64, #{arena_size}));
            nxs_scratch_size = #{arena_size};
          }
          char* nxs_arena = nxs_scratch;
        """,
        else: ""

    body =
      if threads > 1,
        do: parallel_body(g, computed, buffers, offsets, threads),
        else: Enum.map(computed, &compute(g, &1, buffers, false))

    # Outputs that are inputs, constants or repeated: copy.
    copies =
      outputs
      |> Enum.with_index()
      |> Enum.reject(fn {id, i} -> homes[id] == {:output, i} end)
      |> Enum.map(fn {id, i} ->
        n = g.nodes[id]
        "  std::memcpy(out[#{i}], #{buf(id)}, #{size(n.shape)} * sizeof(#{ctype(n.type)}));\n"
      end)

    Process.delete(@plan_key)
    Process.delete(@idx_key)

    source = """
    // Generated by NxSimd.Codegen. Do not edit.
    #include <algorithm>
    #{if threads > 1, do: "#include <omp.h>\n", else: ""}#include <cstdint>
    #include <cstdlib>
    #include <cstring>

    // By value, so that loops with clamps if-convert (std::min/max return
    // references, which stops GCC vectorising under `omp simd`).
    template <class T> static inline T nxs_min(T a, T b) { return b < a ? b : a; }
    template <class T> static inline T nxs_max(T a, T b) { return a < b ? b : a; }

    extern "C" __attribute__((visibility("default")))
    void nx_simd_kernel(const void* const* in, void* const* out) {
    #{arena_decl}#{decls}#{body}#{copies}}
    """

    %{
      source: source,
      inputs: Enum.map(params, &{g.nodes[&1].type, g.nodes[&1].shape}),
      outputs: Enum.map(outputs, &{g.nodes[&1].type, g.nodes[&1].shape})
    }
  end

  # -- which nodes are buffers ------------------------------------------------------

  defp buffer_nodes(g, outputs) do
    base =
      for id <- g.order, g.nodes[id].op in [:parameter, :tensor, :dot, :sum], do: id

    # dot/sum operands that are more than casts/views of a buffer get their
    # own loop, so they are not recomputed for every product.
    operands =
      for id <- g.order,
          g.nodes[id].op in [:dot, :sum],
          input <- g.nodes[id].inputs,
          not cheap?(g, input),
          do: input

    # gathers read their table at arbitrary positions: computed tables
    # are materialised first
    tables = for id <- g.order, g.nodes[id].op == :take, do: hd(g.nodes[id].inputs)

    # concatenations are filled by one loop per input
    concats = for id <- g.order, g.nodes[id].op == :concatenate, do: id

    MapSet.new(base ++ operands ++ tables ++ concats ++ outputs) |> bound_fusion(g)
  end

  # Fused expressions are recomputed by every consumer. A node that several
  # consumers read, and that costs more than @share_cost operations, is
  # computed once into a buffer instead (e.g. the NCO rotation feeding the
  # four first FFT loops).
  @share_cost 12

  # Number of reads of each node, looking through views.
  defp uses(g) do
    Enum.reduce(Enum.reverse(g.order), %{}, fn id, uses ->
      node = g.nodes[id]
      mine = Map.get(uses, id, 0)
      weight = if node.op in @views, do: max(mine, 1), else: 1

      Enum.reduce(node.inputs, uses, fn input, uses ->
        Map.update(uses, input, weight, &(&1 + weight))
      end)
    end)
  end

  # Fusing inlines a node's whole expression into each consumer, once per
  # distinct index it is read at, so chains that read a value at several
  # indices (e.g. butterflies) grow exponentially. Nodes whose inlined
  # expression would exceed @max_fused operations become buffers.
  @max_fused 256

  defp bound_fusion(buffers, g) do
    uses = uses(g)

    {buffers, _cost} =
      Enum.reduce(g.order, {buffers, %{}}, fn id, {buffers, cost} ->
        node = g.nodes[id]
        inputs = Enum.map(node.inputs, &if(MapSet.member?(buffers, &1), do: 0, else: cost[&1]))

        own =
          case node.op do
            op when op in [:parameter, :tensor, :constant] -> 0
            op when op in @views -> Enum.sum(inputs)
            _ -> 1 + Enum.sum(inputs)
          end

        shared? = own > @share_cost and Map.get(uses, id, 0) > 1 and node.op not in @views

        if (own > @max_fused or shared?) and node.op not in [:parameter, :tensor, :constant],
          do: {MapSet.put(buffers, id), Map.put(cost, id, own)},
          else: {buffers, Map.put(cost, id, own)}
      end)

    buffers
  end

  defp cheap?(g, id) do
    case g.nodes[id] do
      %{op: op} when op in [:parameter, :tensor, :constant] -> true
      %{op: op, inputs: [x]} when op in [:as_type | @views] -> cheap?(g, x)
      _ -> false
    end
  end

  defp check_node!(g, node) do
    case node.type do
      {k, _} when k in [:s, :u] -> :ok
      t -> raise Unsupported, "#{node.op}: type #{inspect(t)} (only integer types are generated)"
    end

    supported =
      node.op in [:parameter, :tensor, :constant, :select, :clip, :take, :dot, :sum, :concatenate] or
        node.op in @binary or node.op in @compare or node.op in @logical or node.op in @unary or
        node.op in @views

    unless supported, do: raise(Unsupported, "operation #{inspect(node.op)} is not generated yet")

    case node do
      %{op: :dot, attrs: {_, batch, _, _}} when batch != [] ->
        raise Unsupported, "batched dot"

      %{op: :take, inputs: [table | _]} ->
        if tuple_size(g.nodes[table].shape) != 1,
          do: raise(Unsupported, "take on a non-vector table")

      %{op: :constant, attrs: v} when not is_integer(v) ->
        raise Unsupported, "non-integer constant"

      _ ->
        :ok
    end
  end

  # -- declarations ---------------------------------------------------------------------

  defp declare(g, id, home, offset) do
    n = g.nodes[id]
    t = ctype(n.type)

    case home do
      {:input, i} ->
        "  const #{t}* __restrict__ #{buf(id)} = static_cast<const #{t}*>(in[#{i}]);\n"

      {:output, i} ->
        "  #{t}* __restrict__ #{buf(id)} = static_cast<#{t}*>(out[#{i}]);\n"

      :constant ->
        # Stored in the narrowest type that holds the values (Nx folds
        # as_type into constants, so int8 weights often arrive as s32).
        values = n.attrs |> Nx.to_flat_list()
        st = ctype(storage_type(n))

        "  static const #{st} #{buf(id)}[#{max(size(n.shape), 1)}] = {#{Enum.map_join(values, ", ", &Integer.to_string/1)}};\n"

      :local ->
        lt = ctype(natural(n))

        "  #{lt}* __restrict__ #{buf(id)} = reinterpret_cast<#{lt}*>(nxs_arena + #{offset});\n"
    end
  end

  # Local buffers share one allocation: a buffer lives from the loop that
  # computes it to the last loop that reads it (through fused nodes), and
  # later buffers reuse the freed space (smaller working set, no zeroing).
  defp arena(g, computed, buffers, homes) do
    pos = computed |> Enum.with_index() |> Map.new()

    last =
      Enum.reduce(computed, %{}, fn c, last ->
        c
        |> reads(g, buffers)
        |> Enum.reduce(last, fn b, last -> Map.update(last, b, pos[c], &max(&1, pos[c])) end)
      end)

    {offsets, _live, top} =
      for id <- computed, homes[id] == :local, reduce: {%{}, [], 0} do
        {offsets, live, top} ->
          p = pos[id]
          live = Enum.filter(live, fn {_, _, l} -> l >= p end)
          bytes = align64(size(g.nodes[id].shape) * div(elem(natural(g.nodes[id]), 1), 8))
          off = first_fit(Enum.sort(live), bytes)
          entry = {off, bytes, Map.get(last, id, p)}
          {Map.put(offsets, id, off), [entry | live], max(top, off + bytes)}
      end

    {offsets, top}
  end

  defp align64(n), do: div(n + 63, 64) * 64

  defp first_fit(live, bytes) do
    Enum.reduce_while(live, 0, fn {off, size, _}, cursor ->
      if off - cursor >= bytes, do: {:halt, cursor}, else: {:cont, max(cursor, off + size)}
    end)
  end

  # Buffers read by the loop computing buffer `id`.
  defp reads(id, g, buffers) do
    g.nodes[id].inputs |> Enum.flat_map(&reached(g, &1, buffers)) |> Enum.uniq()
  end

  defp reached(g, id, buffers) do
    if MapSet.member?(buffers, id),
      do: [id],
      else: g.nodes[id].inputs |> Enum.flat_map(&reached(g, &1, buffers))
  end

  # -- computing a buffer ------------------------------------------------------------------

  defp compute(g, id, buffers, par?) do
    node = g.nodes[id]

    case node.op do
      :dot -> dot(g, node, buffers)
      :sum -> sum(g, node, buffers)
      :concatenate -> concatenate(g, node, buffers, par?)
      _ -> elementwise_loop(g, node, buffers, par?)
    end
  end

  # One loop nest over the buffer's shape; the innermost loop is contiguous.
  defp elementwise_loop(g, node, buffers, par?) do
    rank = tuple_size(node.shape)
    vars = for d <- 0..(rank - 1)//1, do: "i#{d}"
    idx = index_vars(vars, node.shape)
    {value, lines} = expr(g, node.id, idx, buffers, %{lines: [], memo: %{}}, true)

    store =
      "#{buf(node.id)}[#{flat(idx, node.shape)}] = static_cast<#{elem_ctype(node)}>(#{value});"

    loops(node.shape, vars, Enum.reverse(lines.lines) ++ [store], "  ", par?)
  end

  # One loop per input, writing its range of the concatenation axis.
  defp concatenate(g, %{inputs: inputs, attrs: axis} = node, buffers, par?) do
    rank = tuple_size(node.shape)
    vars = for d <- 0..(rank - 1)//1, do: "i#{d}"

    {code, _} =
      Enum.reduce(inputs, {"", 0}, fn input, {code, offset} ->
        in_node = g.nodes[input]
        idx = index_vars(vars, in_node.shape)
        {value, st} = expr(g, input, idx, buffers, %{lines: [], memo: %{}})
        out = if offset == 0, do: idx, else: List.update_at(idx, axis, &Index.add(&1, offset))

        store =
          "#{buf(node.id)}[#{flat(out, node.shape)}] = static_cast<#{elem_ctype(node)}>(#{value});"

        {code <> loops(in_node.shape, vars, Enum.reverse(st.lines) ++ [store], "  ", par?),
         offset + elem(in_node.shape, axis)}
      end)

    code
  end

  # With `par?` (inside the parallel region of `parallel_body/5`) each thread
  # runs its static share of one dimension of the nest.
  defp loops(shape, vars, body, indent, par? \\ false) do
    dims = Tuple.to_list(shape)

    case dims do
      [] ->
        "  {\n" <> Enum.map_join(body, "", &"    #{&1}\n") <> "  }\n"

      _ ->
        pd = if par?, do: partition_dim(dims)

        {open, close, depth} =
          dims
          |> Enum.zip(vars)
          |> Enum.with_index()
          |> Enum.reduce({"", "", indent}, fn {{dim, v}, d}, {open, close, ind} ->
            pragma = if d == length(dims) - 1, do: "#{ind}#pragma omp simd\n", else: ""
            {lo, hi} = if d == pd, do: {"nxs_lo", "nxs_hi"}, else: {"0", "#{dim}"}

            {open <> pragma <> "#{ind}for (#{itype()} #{v} = #{lo}; #{v} < #{hi}; ++#{v}) {\n",
             "#{ind}}\n" <> close, ind <> "  "}
          end)

        nest = open <> Enum.map_join(body, "", &"#{depth}#{&1}\n") <> close

        if pd == nil,
          do: nest,
          else:
            "  {\n" <>
              partition_bounds(Enum.at(dims, pd), pd == length(dims) - 1) <> nest <> "  }\n"
    end
  end

  # The outermost dimension with room for every thread (else the largest);
  # the innermost is split in multiples of 16 so that threads keep whole
  # vectors.
  defp partition_dim(dims) do
    Enum.find_index(dims, &(&1 >= 32)) ||
      dims |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)
  end

  defp partition_bounds(dim, innermost?) do
    unit = if innermost?, do: 16, else: 1
    blocks = div(dim, unit)

    """
      const int64_t nxs_lo = (int64_t)#{blocks} * nxs_tid / nxs_nt * #{unit};
      const int64_t nxs_hi = nxs_tid == nxs_nt - 1 ? #{dim} : (int64_t)#{blocks} * (nxs_tid + 1) / nxs_nt * #{unit};
    """
  end

  # -- threads ---------------------------------------------------------------------------------
  #
  # One `omp parallel` region around all loops. Large loops are split
  # statically (each output element is still computed by exactly one
  # thread, so results do not depend on the thread count); small loops and
  # dot/sum run on thread 0. A barrier is placed only where a loop reads,
  # overwrites or reuses (arena) storage that another loop since the last
  # barrier wrote or read.
  @par_min 4096

  defp parallel_body(g, computed, buffers, offsets, threads) do
    work = computed |> Enum.map(&size(g.nodes[&1].shape)) |> Enum.sum()

    {chunks, _} =
      Enum.map_reduce(computed, {[], []}, fn id, {writes, reads} ->
        node = g.nodes[id]
        par? = size(node.shape) >= @par_min and node.op not in [:dot, :sum]
        code = compute(g, id, buffers, par?)
        code = if par?, do: code, else: "  if (nxs_tid == 0) {\n#{code}  }\n"

        my_reads = reads(id, g, buffers)

        conflict? =
          overlaps?(my_reads, writes, g, offsets) or overlaps?([id], reads ++ writes, g, offsets)

        {writes, reads} = if conflict?, do: {[], []}, else: {writes, reads}
        barrier = if conflict?, do: "  #pragma omp barrier\n", else: ""
        {barrier <> code, {[id | writes], my_reads ++ reads}}
      end)

    """
      #pragma omp parallel num_threads(#{threads}) if(#{work} >= #{@par_min})
      {
        const int nxs_tid = omp_get_thread_num();
        const int nxs_nt = omp_get_num_threads();
    #{chunks}  }
    """
  end

  defp overlaps?(as, bs, g, offsets) do
    Enum.any?(as, fn a -> Enum.any?(bs, &same_storage?(a, &1, g, offsets)) end)
  end

  defp same_storage?(a, b, _g, _offsets) when a == b, do: true

  defp same_storage?(a, b, g, offsets) do
    with {oa, sa} <- extent(a, g, offsets), {ob, sb} <- extent(b, g, offsets) do
      oa < ob + sb and ob < oa + sa
    else
      _ -> false
    end
  end

  defp extent(id, g, offsets) do
    case offsets do
      %{^id => off} ->
        n = g.nodes[id]
        {off, align64(size(n.shape) * div(elem(natural(n), 1), 8))}

      _ ->
        nil
    end
  end

  # C expression for `id` at multi-index `idx` (list of C index expressions
  # for the node's own shape). Intermediate values become local constants,
  # memoised per {node, index}.
  defp expr(g, id, idx, buffers, st, root? \\ false) do
    node = g.nodes[id]
    key = {id, idx}

    cond do
      not root? and id in buffers ->
        {"#{buf(id)}[#{flat(idx, node.shape)}]", st}

      Map.has_key?(st.memo, key) ->
        {st.memo[key], st}

      true ->
        {code, st} = op_expr(g, node, idx, buffers, st)

        case node.op do
          op when op in [:constant | @views] ->
            {code, st}

          _ ->
            name = "v#{id}_#{map_size(st.memo)}"
            vt = value_ctype(id)
            line = "const #{vt} #{name} = (#{vt})(#{code});"
            {name, %{st | lines: [line | st.lines], memo: Map.put(st.memo, key, name)}}
        end
    end
  end

  defp op_expr(_g, %{op: :constant, attrs: v, type: t}, _idx, _buffers, st),
    do: {literal(v, t), st}

  # Ring operations (modular in the low bits) run in the unsigned type of the
  # node's width: operands are truncated, the result is exact modulo 2^width
  # and the proven range makes that the value (see `plan/2`).
  defp op_expr(g, %{op: op, inputs: [a, b]} = n, idx, buffers, st) when op in @ring do
    {ca, st} = operand(g, a, n, idx, buffers, st)
    {cb, st} = operand(g, b, n, idx, buffers, st)
    {u, c} = ring_types(n)

    sym =
      %{
        add: "+",
        subtract: "-",
        multiply: "*",
        bitwise_and: "&",
        bitwise_or: "|",
        bitwise_xor: "^"
      }[op]

    {"(#{u})((#{c})#{ca} #{sym} (#{c})#{cb})", st}
  end

  defp op_expr(g, %{op: :left_shift, inputs: [a, b]} = n, idx, buffers, st) do
    {ca, st} = operand(g, a, n, idx, buffers, st)
    {cb, st} = operand(g, b, n, idx, buffers, st)

    case constant_value(g, b) do
      nil ->
        ct = ctype(n.type)
        {"(#{ct})((#{ct})#{ca} << (int)(#{cb}))", st}

      k ->
        {u, c} = ring_types(n)
        {if(k >= width(n.id), do: "(#{u})0", else: "(#{u})((#{c})#{ca} << #{k})"), st}
    end
  end

  # The remaining binary operations need exact operands: they run in the
  # narrowest type holding both operands and the result.
  defp op_expr(g, %{op: op, inputs: [a, b]} = n, idx, buffers, st) when op in @binary do
    {ca, st} = operand(g, a, n, idx, buffers, st)
    {cb, st} = operand(g, b, n, idx, buffers, st)
    ct = ctype(exact_type(g, [a, b, n.id]))

    code =
      case op do
        :quotient ->
          "(#{ct})((#{ct})#{ca} / (#{ct})#{cb})"

        :remainder ->
          "(#{ct})((#{ct})#{ca} % (#{ct})#{cb})"

        :min ->
          "nxs_min<#{ct}>((#{ct})#{ca}, (#{ct})#{cb})"

        :max ->
          "nxs_max<#{ct}>((#{ct})#{ca}, (#{ct})#{cb})"

        :right_shift ->
          st = shift_type(g, a, b, n)
          "(#{st})((#{st})#{ca} >> (int)(#{cb}))"
      end

    {code, st}
  end

  defp op_expr(g, %{op: op, inputs: [a, b]} = n, idx, buffers, st) when op in @compare do
    {ca, st} = operand(g, a, n, idx, buffers, st)
    {cb, st} = operand(g, b, n, idx, buffers, st)
    m = ctype(exact_type(g, [a, b]))

    c =
      %{
        equal: "==",
        not_equal: "!=",
        less: "<",
        less_equal: "<=",
        greater: ">",
        greater_equal: ">="
      }[op]

    {"(uint8_t)((#{m})#{ca} #{c} (#{m})#{cb})", st}
  end

  defp op_expr(g, %{op: op, inputs: [a, b]} = n, idx, buffers, st) when op in @logical do
    {ca, st} = operand(g, a, n, idx, buffers, st)
    {cb, st} = operand(g, b, n, idx, buffers, st)
    c = if op == :logical_and, do: "&&", else: "||"
    {"(uint8_t)((#{ca} != 0) #{c} (#{cb} != 0))", st}
  end

  defp op_expr(g, %{op: op, inputs: [a]} = n, idx, buffers, st) when op in @unary do
    {ca, st} = operand(g, a, n, idx, buffers, st)

    code =
      case op do
        :negate ->
          {u, c} = ring_types(n)
          "(#{u})(-(#{c})#{ca})"

        :bitwise_not ->
          {u, c} = ring_types(n)
          "(#{u})(~(#{c})#{ca})"

        :abs ->
          ct = ctype(exact_type(g, [a, n.id]))
          "(#{ct})((#{ct})#{ca} < 0 ? -(#{ct})#{ca} : (#{ct})#{ca})"

        :as_type ->
          # the node's conversion is applied by the cast to its value type
          ca

        :logical_not ->
          "(uint8_t)(#{ca} == 0)"
      end

    {code, st}
  end

  defp op_expr(g, %{op: :select, inputs: [p, a, b]} = n, idx, buffers, st) do
    {cp, st} = operand(g, p, n, idx, buffers, st)
    {ca, st} = operand(g, a, n, idx, buffers, st)
    {cb, st} = operand(g, b, n, idx, buffers, st)
    {u, _} = ring_types(n)
    {"(#{cp} ? (#{u})#{ca} : (#{u})#{cb})", st}
  end

  defp op_expr(g, %{op: :clip, inputs: [x, lo, hi]} = n, idx, buffers, st) do
    {cx, st} = operand(g, x, n, idx, buffers, st)
    {clo, st} = operand(g, lo, n, idx, buffers, st)
    {chi, st} = operand(g, hi, n, idx, buffers, st)
    ct = ctype(exact_type(g, [x, lo, hi, n.id]))
    {"nxs_min<#{ct}>(nxs_max<#{ct}>((#{ct})#{cx}, (#{ct})#{clo}), (#{ct})#{chi})", st}
  end

  defp op_expr(g, %{op: :take, inputs: [table, indices]} = n, idx, buffers, st) do
    {ci, st} = expr(g, indices, idx, buffers, st)
    {"#{buf(table)}[(int64_t)(#{ci})]", ensure_buffer!(n, table, buffers, st)}
  end

  # Views only remap the index.
  defp op_expr(g, %{op: :broadcast, inputs: [x], attrs: axes} = n, idx, buffers, st) do
    xs = Tuple.to_list(g.nodes[x].shape)

    inner =
      for {dim, d} <- Enum.with_index(xs),
          do: if(dim == 1, do: Index.const(0), else: Enum.at(idx, Enum.at(axes, d)))

    _ = n
    expr(g, x, inner, buffers, st)
  end

  defp op_expr(g, %{op: :transpose, inputs: [x], attrs: perm}, idx, buffers, st) do
    # output dim k is input dim perm[k]
    inner =
      perm
      |> Enum.with_index()
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map(fn {_, k} -> Enum.at(idx, k) end)

    expr(g, x, inner, buffers, st)
  end

  defp op_expr(g, %{op: op, inputs: [x], shape: shape}, idx, buffers, st)
       when op in [:reshape, :squeeze] do
    xs = g.nodes[x].shape

    inner =
      if xs == shape do
        idx
      else
        flat = flat_form(idx, shape)

        for {dim, s} <- Enum.zip(Tuple.to_list(xs), strides(xs)),
            do: if(dim == 1, do: Index.const(0), else: flat |> Index.div(s) |> Index.mod(dim))
      end

    expr(g, x, inner, buffers, st)
  end

  # slice: input index = start + index * stride (static starts).
  defp op_expr(
         g,
         %{op: :slice, inputs: [x], attrs: {starts, _lengths, strides}},
         idx,
         buffers,
         st
       ) do
    inner =
      [idx, starts, strides]
      |> Enum.zip()
      |> Enum.map(fn
        {i, s, k} -> i |> Index.mul(k) |> Index.add(s)
      end)

    expr(g, x, inner, buffers, st)
  end

  # concatenate along `axis`: pick the input whose range holds the index.
  defp op_expr(g, %{op: :concatenate, inputs: inputs, attrs: axis, type: t}, idx, buffers, st) do
    i = Enum.at(idx, axis)

    {parts, st, _offset} =
      Enum.reduce(inputs, {[], st, 0}, fn input, {acc, st, offset} ->
        size = elem(g.nodes[input].shape, axis)
        local = if offset == 0, do: idx, else: List.replace_at(idx, axis, Index.add(i, -offset))
        {code, st} = expr(g, input, local, buffers, st)
        {[{offset + size, code} | acc], st, offset + size}
      end)

    [{_, last} | rest] = parts
    ct = ctype(t)

    code =
      Enum.reduce(rest, "(#{ct})#{last}", fn {limit, code}, else_code ->
        "(#{Index.render(i)} < #{limit} ? (#{ct})#{code} : #{else_code})"
      end)

    {code, st}
  end

  defp ensure_buffer!(_n, table, buffers, st) do
    unless table in buffers, do: raise(Unsupported, "take from a computed table")
    st
  end

  # An input of an elementwise node, with implicit right-aligned broadcasting.
  defp operand(g, input, node, idx, buffers, st) do
    in_shape = Tuple.to_list(g.nodes[input].shape)
    out_rank = tuple_size(node.shape)
    offset = out_rank - length(in_shape)

    inner =
      for {dim, d} <- Enum.with_index(in_shape),
          do:
            if(dim == 1 and elem(node.shape, d + offset) != 1,
              do: Index.const(0),
              else: Enum.at(idx, d + offset)
            )

    expr(g, input, inner, buffers, st)
  end

  # -- dot and sum --------------------------------------------------------------------------

  # Accumulation in the unsigned type: exact modulo 2^bits, and the final
  # sum is proven to fit.
  defp dot(g, %{inputs: [a, b], attrs: {[ca], [], [cb], []}} = n, buffers) do
    as = g.nodes[a].shape
    bs = g.nodes[b].shape
    {_, bits} = n.type
    t = ctype(n.type)
    u = "uint#{bits}_t"
    k = elem(as, ca)

    if tuple_size(as) == 2 and ca == 1 and tuple_size(bs) == 2 and cb == 0 do
      # [m, k] x [k, n]: i-k-j, vectorised over the output row.
      {m, nn} = {elem(as, 0), elem(bs, 1)}

      {va, st} =
        expr(g, a, [Index.var("i", m), Index.var("kk", k)], buffers, %{lines: [], memo: %{}})

      {vb, st2} =
        expr(g, b, [Index.var("kk", k), Index.var("j", nn)], buffers, %{lines: [], memo: %{}})

      """
        for (#{itype()} i = 0; i < #{m}; ++i) {
          #{u} acc[#{nn}] = {0};
          for (#{itype()} kk = 0; kk < #{k}; ++kk) {
      #{Enum.map_join(Enum.reverse(st.lines), "", &"      #{&1}\n")}      const #{u} av = (#{u})(#{t})#{va};
      #pragma omp simd
            for (#{itype()} j = 0; j < #{nn}; ++j) {
      #{Enum.map_join(Enum.reverse(st2.lines), "", &"        #{&1}\n")}        acc[j] += av * (#{u})(#{t})#{vb};
            }
          }
          for (#{itype()} j = 0; j < #{nn}; ++j) #{buf(n.id)}[i * #{nn} + j] = (#{t})acc[j];
        }
      """
    else
      # General: loops over the free dims, inner reduction over k.
      a_free = for d <- 0..(tuple_size(as) - 1)//1, d != ca, do: d
      b_free = for d <- 0..(tuple_size(bs) - 1)//1, d != cb, do: d
      vars = for d <- 0..(tuple_size(n.shape) - 1)//1, do: "o#{d}"
      {a_vars, b_vars} = Enum.split(vars, length(a_free))
      kv = Index.var("kk", k)
      a_idx = index_with(tuple_size(as), a_free, index_vars(a_vars, a_free, as), ca, kv)
      b_idx = index_with(tuple_size(bs), b_free, index_vars(b_vars, b_free, bs), cb, kv)
      {va, st} = expr(g, a, a_idx, buffers, %{lines: [], memo: %{}})
      {vb, st} = expr(g, b, b_idx, buffers, st)

      # A signed int32 accumulator when even the worst partial sum is bounded
      # by the operand types (no overflow possible, so no undefined
      # behaviour): compilers then recognise the dot-product idiom (e.g. sdot).
      acc_type = if small_products?(g, a, b, k, u), do: "int32_t", else: u

      prod =
        if acc_type == "int32_t",
          do: "(int32_t)#{va} * (int32_t)#{vb}",
          else: product(g, a, b, va, vb, t, u)

      # No `omp simd reduction` here: integer addition is associative, so the
      # compiler vectorises the reduction itself, and the OpenMP rewrite
      # (private copies + memset) hides the dot-product idiom (sdot).
      body = [
        "#{acc_type} acc = 0;",
        "for (#{itype()} kk = 0; kk < #{k}; ++kk) { #{Enum.join(Enum.reverse(st.lines), " ")} acc += #{prod}; }",
        "#{buf(n.id)}[#{flat(index_vars(vars, n.shape), n.shape)}] = (#{t})acc;"
      ]

      loops(n.shape, vars, body, "  ") |> String.replace("#pragma omp simd\n", "", global: false)
    end
  end

  defp dot(_g, _n, _buffers), do: raise(Unsupported, "dot with more than one contraction axis")

  # One product of a dot. When the operands' source types guarantee that the
  # product fits int32 (e.g. int8 x int8 or int16 x int8), it is computed
  # signed, which compilers turn into widening multiply-accumulate / dot
  # product instructions; otherwise in the unsigned accumulator type.
  defp product(g, a, b, va, vb, t, u) do
    {sa, ba} = source_type(g, a)
    {sb, bb} = source_type(g, b)

    if sa == :s and sb == :s and ba + bb <= 31 and u == "uint32_t" do
      "(#{u})((int32_t)#{va} * (int32_t)#{vb})"
    else
      "(#{u})(#{t})#{va} * (#{u})(#{t})#{vb}"
    end
  end

  defp small_products?(g, a, b, k, u) do
    {sa, ba} = source_type(g, a)
    {sb, bb} = source_type(g, b)

    u == "uint32_t" and sa == :s and sb == :s and
      k * Bitwise.bsl(1, ba - 1) * Bitwise.bsl(1, bb - 1) < Bitwise.bsl(1, 31)
  end

  # The narrowest type the operand's values come from (looking through
  # widening casts and views).
  defp source_type(g, id) do
    case g.nodes[id] do
      %{op: op, inputs: [x]} when op in [:as_type | @views] -> source_type(g, x)
      %{op: :tensor} = n -> storage_type(n)
      %{type: type} -> type
    end
  end

  # Smallest standard integer type holding a constant tensor's values.
  defp storage_type(%{attrs: t, type: type}) do
    values = Nx.to_flat_list(t)
    {lo, hi} = {Enum.min(values, fn -> 0 end), Enum.max(values, fn -> 0 end)}

    Enum.find([{:s, 8}, {:u, 8}, {:s, 16}, {:u, 16}, {:s, 32}, {:u, 32}], type, fn {k, b} ->
      {tlo, thi} =
        if k == :s,
          do: {-Bitwise.bsl(1, b - 1), Bitwise.bsl(1, b - 1) - 1},
          else: {0, Bitwise.bsl(1, b) - 1}

      lo >= tlo and hi <= thi and elem(type, 1) >= b
    end)
  end

  defp index_with(rank, free, free_vars, contract, kvar) do
    map = Map.new(Enum.zip(free, free_vars)) |> Map.put(contract, kvar)
    for d <- 0..(rank - 1)//1, do: map[d]
  end

  defp sum(g, %{inputs: [x], attrs: opts} = n, buffers) do
    xs = g.nodes[x].shape
    rank = tuple_size(xs)
    axes = opts[:axes] || Enum.to_list(0..(rank - 1)//1)
    if opts[:keep_axes], do: raise(Unsupported, "sum with keep_axes")
    {_, bits} = n.type
    t = ctype(n.type)
    u = "uint#{bits}_t"

    free = for d <- 0..(rank - 1)//1, d not in axes, do: d
    out_vars = for d <- 0..(length(free) - 1)//1, do: "o#{d}"
    red_vars = for d <- axes, do: "r#{d}"

    idx =
      for d <- 0..(rank - 1)//1,
          do:
            if(d in axes,
              do: Index.var("r#{d}", elem(xs, d)),
              else: Index.var("o#{Enum.find_index(free, &(&1 == d))}", elem(xs, d))
            )

    {v, st} = expr(g, x, idx, buffers, %{lines: [], memo: %{}})

    inner =
      Enum.reduce(
        Enum.reverse(Enum.zip(axes, red_vars)),
        "{ #{Enum.join(Enum.reverse(st.lines), " ")} acc += (#{u})(#{t})#{v}; }",
        fn {d, rv}, body ->
          "for (#{itype()} #{rv} = 0; #{rv} < #{elem(xs, d)}; ++#{rv}) #{body}"
        end
      )

    loops(
      n.shape,
      out_vars,
      [
        "#{u} acc = 0;",
        inner,
        "#{buf(n.id)}[#{flat(index_vars(out_vars, n.shape), n.shape)}] = (#{t})acc;"
      ],
      "  "
    )
    |> String.replace("#pragma omp simd\n", "")
  end

  # -- value types ------------------------------------------------------------------------
  #
  # WidthCheck proves a range for every node. Values are computed in the
  # narrowest standard type that holds the range (the node's "natural"
  # type), which is what makes 16-bit data vectorise 8 lanes wide even when
  # the graph says s64 or s32. Beyond that, a node that only feeds ring
  # operations (add, subtract, multiply, and/or/xor, negate, not, casts,
  # constant left shifts, select) needs just the low bits its consumers
  # look at, so it is computed at that width in an unsigned type: the
  # sum is exact modulo 2^width, and wherever an exact value is needed the
  # width is the natural one. A 64-bit phase accumulator that is masked to
  # 32 bits becomes a 32-bit multiply-add.

  # Natural type of a node: smallest standard type holding its proven range
  # (never wider than the declared type).
  defp natural(%{type: {_, bits} = type} = node) do
    {lo, hi} = node_bounds(node)
    found = std_type(lo, hi)
    if found != nil and elem(found, 1) <= bits, do: found, else: type
  end

  defp node_bounds(%{range: range, type: type}) do
    with [_ | _] <- range,
         {lo, hi} when is_integer(lo) and is_integer(hi) <- NxFixed.IntSet.bounds(range) do
      {lo, hi}
    else
      _ -> NxFixed.WidthCheck.type_range(type)
    end
  end

  defp std_type(lo, hi) do
    candidates =
      if lo >= 0,
        do: [{:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}],
        else: [{:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}]

    Enum.find(candidates, fn t ->
      {tlo, thi} = NxFixed.WidthCheck.type_range(t)
      lo >= tlo and hi <= thi
    end)
  end

  # Type for exact arithmetic on nodes `ids`: holds all their ranges.
  defp exact_type(g, ids) do
    bounds = Enum.map(ids, &node_bounds(g.nodes[&1]))
    lo = bounds |> Enum.map(&elem(&1, 0)) |> Enum.min()
    hi = bounds |> Enum.map(&elem(&1, 1)) |> Enum.max()
    std_type(lo, hi) || {:s, 64}
  end

  # Right shifts must stay below the type width: widen when the amount may not.
  defp shift_type(g, a, b, n) do
    {_, amount_hi} = node_bounds(g.nodes[b])
    {kind, bits} = exact_type(g, [a, n.id])
    ctype({kind, if(amount_hi >= bits, do: max(bits, 64), else: bits)})
  end

  defp constant_value(g, id) do
    case g.nodes[id] do
      %{op: :constant, attrs: v} when is_integer(v) and v >= 0 -> v
      _ -> nil
    end
  end

  defp itype, do: Process.get(@idx_key)

  defp plan_of(id), do: Map.fetch!(Process.get(@plan_key), id)

  # Compute width of a node's value (bits).
  defp width(id), do: elem(plan_of(id), 1)

  # C type of the value expression of a node: the natural type, or an
  # unsigned type of the narrower demanded width.
  defp value_ctype(id) do
    {nt, w} = plan_of(id)
    if w == elem(nt, 1), do: ctype(nt), else: "uint#{w}_t"
  end

  # {result type, arithmetic type}: unsigned of the width; below 32 bits
  # arithmetic is done in uint32_t to keep C++ integer promotion signed-free.
  defp ring_types(%{id: id}) do
    w = width(id)
    {"uint#{w}_t", if(w < 32, do: "uint32_t", else: "uint#{w}_t")}
  end

  # Element type of a buffer node's storage (locals use the natural type).
  defp elem_ctype(node), do: ctype(natural(node))

  @ring_ops [
    :add,
    :subtract,
    :multiply,
    :bitwise_and,
    :bitwise_or,
    :bitwise_xor,
    :negate,
    :bitwise_not,
    :as_type
  ]

  # Demanded widths: walk the graph backwards, each node asking its inputs
  # for the width it needs. Buffers (stored exactly) and non-ring consumers
  # ask for the natural width.
  defp plan(g, buffers) do
    {_, plan} =
      g.order
      |> Enum.reverse()
      |> Enum.reduce({%{}, %{}}, fn id, {dem, plan} ->
        node = g.nodes[id]
        nt = natural(node)
        full = elem(nt, 1)

        w =
          case Map.get(dem, id) do
            d when d != nil ->
              if(MapSet.member?(buffers, id), do: full, else: min(full, round_width(d)))

            _ ->
              full
          end

        dem =
          Enum.reduce(demands(g, node, w), dem, fn {input, d}, dem ->
            Map.update(dem, input, d, &max(&1, d))
          end)

        {dem, Map.put(plan, id, {nt, w})}
      end)

    plan
  end

  defp demands(g, node, w) do
    full = fn id -> elem(natural(g.nodes[id]), 1) end

    case node do
      %{op: op, inputs: inputs} when op in @ring_ops ->
        Enum.map(inputs, &{&1, w})

      %{op: :left_shift, inputs: [a, b]} ->
        if constant_value(g, b), do: [{a, w}, {b, full.(b)}], else: [{a, full.(a)}, {b, full.(b)}]

      %{op: :select, inputs: [p, a, b]} ->
        [{p, full.(p)}, {a, w}, {b, w}]

      %{op: op, inputs: [x]} when op in @views ->
        [{x, w}]

      %{inputs: inputs} ->
        Enum.map(inputs, &{&1, full.(&1)})
    end
  end

  defp round_width(d), do: Enum.find([8, 16, 32, 64], &(&1 >= d))

  # -- helpers ---------------------------------------------------------------------------------

  defp buf(id), do: "b#{id}"

  # Loop variable names to index forms (one per dimension of `shape`).
  defp index_vars(vars, shape),
    do: vars |> Enum.zip(Tuple.to_list(shape)) |> Enum.map(fn {v, d} -> Index.var(v, d) end)

  # Variables of the output dims `dims` of an operand of shape `shape`.
  defp index_vars(vars, dims, shape),
    do: vars |> Enum.zip(dims) |> Enum.map(fn {v, d} -> Index.var(v, elem(shape, d)) end)

  defp flat_form(idx, shape) do
    idx
    |> Enum.zip(strides(shape))
    |> Enum.reduce(Index.const(0), fn {i, s}, acc -> Index.add(acc, Index.mul(i, s)) end)
  end

  defp flat(idx, shape), do: idx |> flat_form(shape) |> Index.render()

  defp strides(shape) do
    dims = Tuple.to_list(shape)

    dims
    |> Enum.reverse()
    |> Enum.scan(1, &(&1 * &2))
    |> then(fn s -> [1 | s] end)
    |> Enum.take(length(dims))
    |> Enum.reverse()
  end

  defp size(shape), do: shape |> Tuple.to_list() |> Enum.product()

  defp ctype({:s, b}), do: "int#{b}_t"
  defp ctype({:u, b}), do: "uint#{b}_t"

  defp literal(v, {:s, 64}) when v < 0, do: "(int64_t)(#{v + 1}LL - 1)"
  defp literal(v, t), do: "(#{ctype(t)})#{v}LL"
end
