# int8 fully connected layer per backend.
#   MIX_ENV=test mix run scripts/bench_ml.exs [batch] [in] [out]
alias ExHdl.ML.Quantized, as: Q
[batch, inputs, outputs] =
  case System.argv() do
    [b, i, o] -> Enum.map([b, i, o], &String.to_integer/1)
    _ -> [64, 256, 128]
  end

:rand.seed(:exsss, {1, 2, 3})
weights = for _ <- 1..outputs, do: for(_ <- 1..inputs, do: :rand.uniform(255) - 128)
params =
  Q.fully_connected_params(
    weights: weights, bias: for(_ <- 1..outputs, do: :rand.uniform(20_001) - 10_001),
    input_scale: 0.05, input_zero_point: -3, weight_scales: for(_ <- 1..outputs, do: 0.002 + :rand.uniform() / 100),
    output_scale: 0.2, output_zero_point: 5, activation: :relu
  )
x = Nx.tensor(for(_ <- 1..batch, do: for(_ <- 1..inputs, do: :rand.uniform(256) - 129)), type: :s8)
macs = batch * inputs * outputs

time = fn label, fun ->
  ref = fun.()
  reps = 20
  {us, _} = :timer.tc(fn -> for _ <- 1..reps, do: fun.() end)
  per = us / reps
  IO.puts("#{String.pad_trailing(label, 30)} #{:erlang.float_to_binary(per / 1000, decimals: 3)} ms  #{:erlang.float_to_binary(macs / per / 1000, decimals: 2)} GMAC/s")
  ref
end

IO.puts("fully connected #{batch}x#{inputs} -> #{outputs} (#{macs} MACs)")

# Nx.Defn.compile fixes the parameters at compile time: no per-call cache
# lookup (Nx.Defn.jit hashes the captured parameters on every call).
template = Nx.template({batch, inputs}, :s8)
simd = Nx.Defn.compile(&Q.fully_connected/2, [template, params], compiler: ExHdl.SIMD.Compiler, fallback: false)
exla = Nx.Defn.compile(&Q.fully_connected/2, [template, params], compiler: EXLA)
xe = Nx.backend_transfer(x, EXLA.Backend)
a = time.("ExHdl.SIMD.Compiler", fn -> simd.(x) end)
b = time.("EXLA (inputs on EXLA)", fn -> exla.(xe) end)
c = if macs <= 300_000, do: time.("Nx evaluator", fn -> Q.fully_connected(x, params) end), else: Q.fully_connected(x, params)
IO.puts("identical: #{Nx.to_list(a) == Nx.to_list(Nx.backend_copy(b)) and Nx.to_list(a) == Nx.to_list(c)}")
