defmodule NxSimd.NIF do
  @moduledoc false
  # Raw bindings to c_src/nx_simd: loading and running generated kernels.
  @on_load :load

  def load do
    path = :filename.join(:code.priv_dir(:nx_simd), ~c"native/libnx_simd")

    # Keep the module loadable without the NIF; NxSimd.available?/0 tells.
    case :erlang.load_nif(path, 0) do
      :ok -> :persistent_term.put({__MODULE__, :loaded}, true)
      {:error, reason} -> :persistent_term.put({__MODULE__, :loaded}, {:error, reason})
    end

    :ok
  end

  def kernel_load(_path), do: :erlang.nif_error(:not_loaded)
  def kernel_run(_kernel, _inputs, _output_bytes), do: :erlang.nif_error(:not_loaded)
end
