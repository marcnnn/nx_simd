# NxSimd

An `Nx.Defn.Compiler` that turns integer `defn` graphs into native code:
it generates C++ for the traced graph, builds it with the host C++
compiler (`-O3 -march=native`, auto-vectorised) and runs it through a
small NIF. On top of it, `NxSimd.Quantized` provides int8 layers with
TensorFlow Lite semantics.

```elixir
Nx.Defn.jit(&MyModel.layer/3, compiler: NxSimd.Compiler).(x, w, b)
```

Every graph must pass [nx_fixed](../nx_fixed)'s width check first: the
generated code is exact (and free of undefined behaviour) only for graphs
whose values provably fit their types. The same range information picks
accumulator types and narrows constant storage. Float graphs fall back to
the Nx evaluator (or raise with `fallback: false`).

## When to use it, and when not

| Situation | Use |
|---|---|
| Float models, training, GPU | EXLA (or Torchx, EMLX on Apple silicon) |
| Running exported ONNX models | Ortex |
| Integer / quantized inference that must be bit-exact across machines (and with an FPGA implementation) | NxSimd |
| Proven accumulator widths, no silent overflow | NxSimd |
| Small int8 layers on a CPU where XLA's per-call overhead dominates | NxSimd (see the measurements below) |

Not supported: floats, autodiff, GPU, batched `dot`. Cross-compiling the
generated code for a target without a compiler (e.g. a Zynq's Cortex-A9)
is not implemented, and the generated code has not been verified on
32-bit ARM yet (the hand-written kernels in ex_sdr have).

## How it works

`Nx.Defn.jit(fun, compiler: NxSimd.Compiler)` turns integer `defn`
graphs into native code instead of hand-written kernels:

1. `NxFixed.WidthCheck` must accept the graph; possible overflow
   raises (the generated C++ is only exact, and free of undefined
   behaviour, for proven graphs). Floats fall back to the evaluator.
2. `NxSimd.Codegen` emits C++: parameters, constants, `dot`/`sum`
   results and outputs are buffers; elementwise operations, casts, views
   (`broadcast`, `reshape`, `transpose`, `slice`, `concatenate`) and table
   lookups are fused into the loop that computes the next buffer.
3. The host compiler builds it (`-O3 -march=native`), cached by content
   hash, and a small NIF loads it with `dlopen`. The finished runner is
   cached per function, argument types and options.

The split follows established practice (TVM/LLVM style): generated loops,
vectorised by the C++ compiler. What made the difference in practice:

- **Storage narrowing from ranges:** Nx folds `as_type` into constants, so
  int8 weights arrive as s32; constants are stored in the narrowest type
  their values fit (4x less memory traffic, and the int8 pattern stays
  visible to the compiler).
- **Signed accumulators where the types prove no overflow** (e.g. K int8 x
  int8 products for K < 2^17), unsigned modular accumulation otherwise.
- **No `#pragma omp simd reduction` on integer dot products:** the OpenMP
  rewrite hides the dot-product idiom; without it GCC emits `sdot`
  (16 int8 multiply-accumulates per instruction on Armv8.2+).
- **Parameters as tensors, compiled with `Nx.Defn.compile`:** `Nx.Defn.jit`
  keys its cache on the closure, i.e. hashes captured parameters on every
  call (900 us for a 32k-element weight list; EXLA pays the same).

## Quantized ML: `NxSimd.Quantized`

int8 fully connected layer with TensorFlow Lite semantics (per-channel
weights, `QuantizeMultiplier`, gemmlowp double rounding with ties away
from zero, zero point folded into the bias as in TFLite's optimised
kernels). Verified against TFLite reference values worked out from the
sources, bit-identical across evaluator, EXLA and generated code, and
within 1 LSB of a float reference.

Neoverse-N1, `MIX_ENV=test mix run scripts/bench_ml.exs batch in out`:

| Layer (int8) | Generated code | EXLA | Nx evaluator |
|---|---|---|---|
| 64x256 -> 128 | 0.23-0.70 ms | 0.97-1.81 ms | |
| 1x1024 -> 1024 | 0.24 ms | 0.90 ms | |
| 128x512 -> 256 | 2.1 ms (7.9 GMAC/s) | 3.0 ms | |
| 16x64 -> 32 | 0.033 ms | 0.46 ms | 11.3 ms |

Pitfall found on the way: `Nx.tensor(2147483648)` is `-2147483648`
(integer literals default to s32 and wrap silently); constants of 2^31 and
above must be created with an explicit `:s64` type.



## Building

`mix compile` builds the loader NIF (no third-party C++ library); kernels
are compiled at run time with `$CXX` or `c++` and cached in `_build`.
