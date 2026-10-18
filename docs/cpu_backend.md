# CPU backend: design notes and measurements

ExHdl runs `ExHdl.DSP.Blocks` on the CPU in three ways, all bit-identical
for fixed-point formats:

| Path | Entry point | Use |
|---|---|---|
| Per-sample Nx evaluator | `ExHdl.DSP.run_scan/5` | reference, debugging |
| Compiled block loop (EXLA) | `ExHdl.DSP.run_scan_stacked/5` | any block and format |
| Highway SIMD NIF | `ExHdl.CPU.run_block/5`, `ExHdl.SIMD` | Q15 DDC, FIR, one-pole |

`ExHdl.CPU.run_block/5` picks the SIMD kernels where they cover the block
and format and falls back to the compiled loop otherwise.

## Measurements

Neoverse-N1 (aarch64, NEON), block size 65536, Msamples/s, from
`MIX_ENV=test mix run scripts/bench_cpu.exs`. The machine is shared, so
numbers vary by about +-30 % between runs.

| Block (Q15) | SIMD NIF | Nx/EXLA loop | GNU Radio 3.10 (float, same machine) |
|---|---|---|---|
| DDC (NCO, mixer, gain) | 80-93 | 14-24 | 33.5 (`rotator_cc` + `multiply_const_cc`) |
| Complex FIR, 31 taps | 50-52 | 1 | 32.5 (`fir_filter_ccf`) |
| One-pole IIR | 94-109 | 21-34 | 69.3 (`single_pole_iir_filter_ff`) |

The per-sample evaluator runs at about 0.01 Msamples/s.

## Decisions and why

**Exact int32 arithmetic instead of Q15 fixed-point multiply instructions.**
Highway's `MulFixedPoint15` (NEON `vqrdmulh`, x86 `pmulhrsw`) rounds each
product separately and differs between targets at -32768 x -32768. The
reference rounds the *sum* of the complex products once, so the kernels
widen to int32, add exactly and then apply `(v + 2^14) >> 15` and a
saturating `DemoteTo`. `ExHdl.Passes.WidthCheck` proves the int32 ranges
(e.g. the mixer stays below 2^31 - 2^16 with a sine table of amplitude
32767); the NIF validates the inputs those proofs rely on.

**FIR: vectorised over outputs with widening multiply-accumulate.**
Each tap is broadcast and multiplied with a sliding window of int16 inputs
into int32 accumulators (`WidenMulAccumulate`, `vmlal_s16` on NEON),
keeping lane order. Constant taps let the NIF check the accumulator bound
(32768 * sum |taps| < 2^31). History and new block are joined inside the
NIF: doing it with Elixir binaries cost as much as the filter itself.

**DDC: the table lookup limits SIMD gains.** NEON has no gather, so
`GatherIndex` is emulated per lane and the SIMD kernel is only somewhat
faster than scalar code. Faster NCOs (recursive oscillators, CORDIC) change
the bits and belong to the planned implementation selection, not to a
bit-exact port of the table NCO.

**IIR: scalar.** The one-pole recurrence cannot be vectorised over time
without changing the arithmetic (look-ahead forms are not bit-identical in
fixed point). The established approach is one independent channel per SIMD
lane; not implemented yet.

**Compiled loop for everything else.** Running the unchanged step function
in one EXLA `while` loop is bit-identical and needs no per-kernel work. Two
details matter: arguments must already live on `EXLA.Backend` (jit with
BinaryBackend arguments was about 5x slower) and the width check is cached
per kernel and argument types.

## Building

The Makefile fetches Google Highway 1.4.0 (pinned SHA-256) and compiles the
few library sources needed for runtime dispatch into the NIF; no system
package is needed. Highway picks the best target at runtime on x86_64 and
aarch64.

**Zynq-7000 (Cortex-A9, ARMv7, NEON + VFPv3).** Highway disables NEON on
ARMv7 cores without VFPv4 unless `HWY_BROKEN_ARM7_WITHOUT_VFP4=0` is set.
The Makefile adds `-march=armv7-a -mfpu=neon -mfloat-abi=hard` and that
override for 32-bit ARM compilers. Verified under `qemu-arm -cpu cortex-a9`:
target `NEON_WITHOUT_AES`, all 1812 random cases bit-exact against the
scalar reference, FIR about 3x faster than scalar. Do not use
`-mfpu=neon-vfpv4` (Highway's README suggestion): it permits VFPv4
instructions, which a real Cortex-A9 lacks. The override is only safe for
integer kernels. `test/simd_test.exs` fails if the build falls back to
`SCALAR`/`EMU128`.

EXLA has no precompiled XLA for 32-bit ARM, so on the Zynq the SIMD NIF
(or the slow Nx evaluator) is the CPU path.

## Sources

- Google Highway 1.4.0: `g3doc/quick_reference.md`, `hwy/ops/arm_neon-inl.h`,
  `hwy/detect_targets.h` (https://github.com/google/highway)
- ARM CMSIS-DSP `arm_fir_q15`, `arm_cmplx_mult_cmplx_q15`,
  `arm_biquad_cascade_df1_q15` (https://github.com/ARM-software/CMSIS-DSP)
- VOLK `volk_16ic_x2_multiply_16ic` (https://github.com/gnuradio/volk)
- erl_nif: dirty schedulers, binaries (https://www.erlang.org/doc/apps/erts/erl_nif.html)
- fine 0.1.6 (https://github.com/elixir-nx/fine)
- XLA precompiled targets (https://github.com/elixir-nx/xla, `lib/xla.ex`)
