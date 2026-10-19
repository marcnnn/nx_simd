defmodule ExHdl.SIMD.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` that turns integer `defn` graphs into native code.

      Nx.Defn.jit(&MyModel.layer/2, compiler: ExHdl.SIMD.Compiler).(x)

  Steps per new combination of function and argument shapes/types:

    1. Build the `ExHdl.IR.Dataflow` graph and run `ExHdl.Passes.WidthCheck`.
       Possible overflow raises `ExHdl.DSP.RangeError`: the generated code
       is only exact (and free of undefined behaviour) for proven graphs.
    2. Generate C++ (`ExHdl.SIMD.Codegen`), compile it with the host C++
       compiler (`-O3 -march=native -fopenmp-simd`) into a shared library,
       cached by content hash, and load it into the SIMD NIF.
    3. Run: inputs are passed as binaries, outputs come back as binaries.

  Graphs the generator does not cover (floats, unsupported operations) run
  on `Nx.Defn.Evaluator` instead, unless `fallback: false` is given.

  Options: `:fallback` (default `true`), `:cxx` (default `$CXX` or `c++`),
  `:cxxflags` (replaces the default flags), plus the width-check options
  (`:param_ranges`, ...).
  """
  @behaviour Nx.Defn.Compiler

  require Logger
  alias ExHdl.IR.Dataflow
  alias ExHdl.Passes.WidthCheck
  alias ExHdl.SIMD.{Codegen, NIF}

  @own [:param_ranges, :case_split, :split_threshold, :split_budget, :fallback, :cxx, :cxxflags]
  @default_flags ~w(-O3 -march=native -std=c++20 -fopenmp-simd -fPIC -shared -fno-math-errno)

  @impl true
  def __jit__(key, vars, fun, args_list, opts), do: __compile__(key, vars, fun, opts).(args_list)

  # Nx calls __jit__ on every invocation. The finished runner is cached per
  # function (`key` captures static arguments such as weights), argument
  # types/shapes and options, so tracing, analysis and code generation run
  # once, like EXLA's executable cache.
  @impl true
  def __compile__(key, vars, fun, opts) do
    signature =
      vars |> List.wrap() |> Nx.Defn.Composite.flatten_list() |> Enum.map(&{&1.type, &1.shape})

    cache_key = {__MODULE__, :runner, key, signature, opts}

    case :persistent_term.get(cache_key, nil) do
      nil ->
        runner = do_compile(key, vars, fun, opts)
        :persistent_term.put(cache_key, runner)
        runner

      runner ->
        runner
    end
  end

  defp do_compile(key, vars, fun, opts) do
    {own, rest} = Keyword.split(opts, @own)
    output = fun.(vars)

    case build(output, own) do
      {:ok, kernel, meta} ->
        runner(kernel, meta, output)

      {:fallback, reason} ->
        if Keyword.get(own, :fallback, true) == false, do: raise(Codegen.Unsupported, reason)
        Logger.debug("ExHdl.SIMD.Compiler: falling back to Nx.Defn.Evaluator (#{reason})")
        Nx.Defn.Evaluator.__compile__(key, vars, fun, rest)
    end
  end

  @impl true
  def __partitions_options__(opts), do: [opts]

  @impl true
  def __to_backend__(_opts), do: {Nx.BinaryBackend, []}

  @impl true
  def __shard_jit__(_key, _mesh, _vars, _fun, _args_list, _opts),
    do: raise("sharding is not supported by ExHdl.SIMD.Compiler")

  @doc "The generated C++ for a traced output (for inspection and tests)."
  def source(output, opts \\ []) do
    {:ok, _graph, meta} = generate(output, opts)
    meta.source
  end

  defp build(output, own) do
    with {:ok, _graph, meta} <- generate(output, own),
         true <- ExHdl.SIMD.available?() || {:fallback, "the SIMD NIF is not loaded"} do
      {:ok, load(meta.source, own), meta}
    end
  end

  defp generate(output, own) do
    graph =
      try do
        Dataflow.from_expr(output)
      rescue
        e in ArgumentError -> throw({:fallback, Exception.message(e)})
      end

    {graph, violations} = WidthCheck.run(graph, own)

    cond do
      Enum.any?(violations, &(&1.kind == :float)) ->
        {:fallback, "floating point (the generator handles integer graphs)"}

      violations != [] ->
        raise ExHdl.DSP.RangeError,
          message: WidthCheck.format(graph, violations, "ExHdl.SIMD.Compiler"),
          graph: graph,
          violations: violations

      true ->
        try do
          meta = Codegen.generate(graph)

          params =
            for id <- graph.order, graph.nodes[id].op == :parameter, do: graph.nodes[id].attrs

          {:ok, graph, Map.put(meta, :params, Enum.sort(params))}
        rescue
          e in Codegen.Unsupported -> {:fallback, Exception.message(e)}
        end
    end
  catch
    {:fallback, _} = fallback -> fallback
  end

  # Compiles (once per source, flags and compiler) and loads (once per VM).
  defp load(source, own) do
    cxx = Keyword.get_lazy(own, :cxx, fn -> System.get_env("CXX") || "c++" end)
    flags = Keyword.get(own, :cxxflags, @default_flags)

    hash =
      :crypto.hash(:sha256, [source, 0, cxx, 0, Enum.join(flags, " ")])
      |> Base.encode16(case: :lower)
      |> binary_part(0, 24)

    case :persistent_term.get({__MODULE__, hash}, nil) do
      nil ->
        kernel = NIF.kernel_load(compile(source, cxx, flags, hash))
        :persistent_term.put({__MODULE__, hash}, kernel)
        kernel

      kernel ->
        kernel
    end
  end

  defp compile(source, cxx, flags, hash) do
    dir = cache_dir()
    so = Path.join(dir, "#{hash}.so")

    unless File.exists?(so) do
      File.mkdir_p!(dir)
      tmp = Path.join(dir, "#{hash}.#{System.unique_integer([:positive])}")
      File.write!(tmp <> ".cc", source)

      case System.cmd(cxx, flags ++ ["-o", tmp <> ".so", tmp <> ".cc"], stderr_to_stdout: true) do
        {_, 0} ->
          File.rename!(tmp <> ".so", so)
          File.rename!(tmp <> ".cc", Path.join(dir, "#{hash}.cc"))

        {log, status} ->
          raise "ExHdl.SIMD.Compiler: #{cxx} exited with #{status}:\n#{log}\nsource: #{tmp}.cc"
      end
    end

    so
  end

  defp cache_dir do
    if Code.ensure_loaded?(Mix.Project) and Mix.Project.get(),
      do: Path.join(Mix.Project.build_path(), "ex_hdl_jit"),
      else: Path.join(System.tmp_dir!(), "ex_hdl_jit")
  end

  # Nx calls the compiled function with a list of lazy parameter lists.
  defp runner(kernel, meta, output) do
    out_bytes = Enum.map(meta.outputs, fn {{_, bits}, shape} -> div(bits, 8) * Nx.size(shape) end)

    fn [params] ->
      inputs =
        Enum.map(meta.params, fn i ->
          params |> Enum.fetch!(i) |> then(& &1.()) |> Nx.to_binary()
        end)

      binaries = NIF.kernel_run(kernel, inputs, out_bytes)

      tensors =
        Enum.zip_with(binaries, meta.outputs, fn bin, {type, shape} ->
          bin |> Nx.from_binary(type) |> Nx.reshape(shape)
        end)

      {result, []} =
        Nx.Defn.Composite.traverse(output, tensors, fn _expr, [t | rest] -> {t, rest} end)

      [result]
    end
  end
end
