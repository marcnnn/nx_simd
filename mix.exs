defmodule NxSimd.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_simd,
      version: "0.1.0",
      elixir: "~> 1.17",
      description:
        "Nx compiler to native C++ for integer and quantized (int8) graphs, including 32-bit ARM",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_env: fn -> %{"FINE_INCLUDE_DIR" => Fine.include_dir()} end,
      make_clean: ["clean"],
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 1.0"},
      sibling(:nx_fixed),
      {:fine, "~> 0.1.6", runtime: false},
      {:elixir_make, "~> 0.9", runtime: false},
      {:stream_data, "~> 1.1", only: [:dev, :test]}
    ]
  end

  # A sibling checkout (by default ../<app>) when there is one, otherwise the
  # GitHub repository. Inside deps/ always GitHub, so that every project in a
  # dependency tree picks the same source.
  defp sibling(app, opts \\ []) do
    {path, opts} = Keyword.pop(opts, :path, "../#{app}")
    nested? = Path.basename(Path.dirname(__DIR__)) == "deps"

    if not nested? and File.dir?(Path.expand(path, __DIR__)),
      do: {app, [path: path, override: true] ++ opts},
      else: {app, [github: "marcnnn/#{app}"] ++ opts}
  end
end
