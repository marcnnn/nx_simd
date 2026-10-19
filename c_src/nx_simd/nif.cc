// Erlang NIF for NxSimd: loads generated kernel libraries (dlopen) and runs
// them on binaries. Inputs are read in place; outputs are new binaries.

#include <dlfcn.h>
#include <erl_nif.h>

#include <fine.hpp>
#include <stdexcept>
#include <string>
#include <vector>

// -- generated kernels (NxSimd.Compiler) --------------------------------
// A generated shared library exports
//   extern "C" void nx_simd_kernel(const void* const* in, void* const* out);
// with all shapes and types compiled in.

using KernelFn = void (*)(const void* const*, void* const*);

struct Kernel {
  void* handle = nullptr;
  KernelFn fn = nullptr;

  explicit Kernel(const std::string& path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) throw std::runtime_error(std::string("kernel_load: ") + dlerror());
    fn = reinterpret_cast<KernelFn>(dlsym(handle, "nx_simd_kernel"));
    if (fn == nullptr) throw std::runtime_error("kernel_load: nx_simd_kernel not found");
  }

  void destructor(ErlNifEnv*) {
    if (handle != nullptr) dlclose(handle);
    handle = nullptr;
  }
};
FINE_RESOURCE(Kernel);

fine::ResourcePtr<Kernel> kernel_load(ErlNifEnv*, std::string path) { return fine::make_resource<Kernel>(path); }
FINE_NIF(kernel_load, ERL_NIF_DIRTY_JOB_IO_BOUND);

// Inputs are read in place; outputs are allocated as binaries of the given
// byte sizes and filled by the kernel.
std::vector<fine::Term> kernel_run(ErlNifEnv* env, fine::ResourcePtr<Kernel> kernel, std::vector<ErlNifBinary> inputs,
                                   std::vector<int64_t> output_bytes) {
  std::vector<const void*> in;
  in.reserve(inputs.size());
  for (const auto& bin : inputs) in.push_back(bin.data);

  std::vector<fine::Term> terms;
  std::vector<void*> out;
  for (int64_t bytes : output_bytes) {
    ERL_NIF_TERM term;
    auto* data = enif_make_new_binary(env, static_cast<size_t>(bytes), &term);
    if (data == nullptr) throw std::runtime_error("kernel_run: cannot allocate output");
    terms.push_back(fine::Term(term));
    out.push_back(data);
  }

  kernel->fn(in.data(), out.data());
  return terms;
}
FINE_NIF(kernel_run, ERL_NIF_DIRTY_JOB_CPU_BOUND);

FINE_INIT("Elixir.NxSimd.NIF");
