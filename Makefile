# Builds the kernel loader NIF (c_src/nx_simd) for NxSimd. Driven by
# elixir_make, which sets MIX_APP_PATH, ERTS_INCLUDE_DIR and (via mix.exs)
# FINE_INCLUDE_DIR. The kernels themselves are compiled at run time by
# NxSimd.Compiler.

MIX_APP_PATH ?= _build/dev/lib/nx_simd
BUILD_DIR    := $(abspath $(MIX_APP_PATH)/../..)/nx_simd_native
OBJ_DIR      := $(BUILD_DIR)/obj
PRIV_DIR     := $(MIX_APP_PATH)/priv/native
NIF          := $(PRIV_DIR)/libnx_simd.so

CXX      ?= g++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter
CPPFLAGS += -I$(ERTS_INCLUDE_DIR) -I$(FINE_INCLUDE_DIR)
LDFLAGS  += -shared -ldl

all: $(NIF)

$(NIF): $(OBJ_DIR)/nif.o
	@mkdir -p $(PRIV_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/nif.o: c_src/nx_simd/nif.cc
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(NIF)

.PHONY: all clean
