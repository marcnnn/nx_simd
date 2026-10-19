defmodule NxSimd do
  @moduledoc """
  Native CPU execution for integer Nx graphs.

  `NxSimd.Compiler` is an `Nx.Defn.Compiler`: it generates C++ for a traced
  integer `defn` (fused element-wise loops, dot products with the
  accumulator type chosen from `NxFixed`'s range analysis), compiles it
  with the system C++ compiler (`-O3 -march=native`, auto-vectorised) and
  runs it through a small NIF. `NxSimd.Quantized` builds int8 layers with
  TFLite semantics on top of it.

  Use it where EXLA does not fit: 32-bit ARM, bit-exact integer inference,
  proven accumulator widths. For float models EXLA is the better choice.
  """

  @doc "Whether the kernel loader NIF is loaded."
  def available? do
    Code.ensure_loaded?(NxSimd.NIF) and
      :persistent_term.get({NxSimd.NIF, :loaded}, false) == true
  end
end
