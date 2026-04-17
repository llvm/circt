# ESI Runtime development guide

## Use an existing CIRCT build for local debugging

The ESI runtime can be built independently, but for local agent work and for debugging changes that span both the runtime and PyCDE, prefer an existing CIRCT CMake build under `build/` or some subdirectory. If they exist, prefer 'build/debug' or 'build/default'.

### Setup

```bash
# Create a venv with matching Python version and install test deps.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools nanobind pytest numpy psutil executing
```

If an existing CIRCT build is not available, use something like the following to create one. The important part is to have the Python bindings and the runtime targets built, which are required for the integration tests. If clang is available, prefer that as the host compiler for faster builds and better diagnostics; otherwise, the default system compiler will work. LLD is _highly_ recommended for faster linking.

```
cmake -G Ninja llvm/llvm -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=circt \
  -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
  -DCIRCT_ENABLE_FRONTENDS=PyCDE \
  -DESI_RUNTIME=ON \
  -DESI_RUNTIME_TRACE=ON \
  -DESI_COSIM=ON
```

Now that a valid build directory exists, continue as below.

```
# Reuse a configured CIRCT build directory under build/.
# In many local setups this is build/default.
export CIRCT_BUILD=<absolute-path-to-build-tree>
test -f "$CIRCT_BUILD/CMakeCache.txt"

# System prerequisites for cosim: gRPC and protobuf C++ libraries must be
# installed (e.g., apt install libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
# on Debian/Ubuntu, or via vcpkg/brew). These are required by CMake's
# find_package(gRPC) and find_package(Protobuf) when ESI_COSIM=ON.

# Refresh the PyCDE and ESI runtime pieces inside the CIRCT build.
ninja -C "$CIRCT_BUILD" PyCDE ESIRuntime ESIRuntimeCppTests
```

This reuses the CIRCT build's `pycde` package under `tools/circt/python_packages/pycde/`, the ESI runtime Python package under `tools/circt/lib/Dialect/ESI/runtime/python/`, and the runtime shared libraries under `lib/`. The `ESIRuntime` target still builds `ESICppRuntime`, `CosimBackend`, `CosimRpc`, `EsiCosimDpiServer`, `esiquery`, and the Python runtime pieces.

When running as a local agent, DO NOT use docker. Check for a working virtual environment first, which is usually at the repo root. Then look for a working CMake build directory under `build/`; `build/default` is the common case.

### Running the integration pytests

The cosim pytests should use the CIRCT build tree for both PyCDE and the ESI runtime:

```bash
cd <repo-root>
source .venv/bin/activate

# Reuse a configured CIRCT build under build/.
# In many local setups this is build/default.
export CIRCT_BUILD=$PWD/build/default

export LD_LIBRARY_PATH=$CIRCT_BUILD/lib:$CIRCT_BUILD/tools/circt/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CIRCT_BUILD/lib:$CIRCT_BUILD/tools/circt/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$CIRCT_BUILD/tools/circt/python_packages/pycde:$CIRCT_BUILD/tools/circt/lib/Dialect/ESI/runtime/python
export PATH=$CIRCT_BUILD/bin:$PWD/ext/bin:$PATH

python3 -m pytest lib/Dialect/ESI/runtime/tests/ -v
```

If `build/default` does not exist, use whichever `build/<name>` contains `CMakeCache.txt` and the `tools/circt` tree. This keeps `pycde` and `esiaccel` coming from the same CIRCT build, which is the preferred setup when co-developing across both projects.

Additionally, the verilator linker needs `libEsiCosimDpiServer.so`. The cosim pytest framework looks for it at `<esiaccel-package>/lib/`. If the build tree does not already provide that directory, create symlinks:
```bash
mkdir -p "$CIRCT_BUILD/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/lib"
ln -sf "$CIRCT_BUILD/lib/libEsiCosimDpiServer.so" \
  "$CIRCT_BUILD/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/lib/libEsiCosimDpiServer.so"
ln -sf "$CIRCT_BUILD/lib/libESICppRuntime.so" \
  "$CIRCT_BUILD/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/lib/libESICppRuntime.so"
ln -sf "$CIRCT_BUILD/lib/libCosimBackend.so" \
  "$CIRCT_BUILD/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/lib/libCosimBackend.so"
ln -sf "$CIRCT_BUILD/lib/libCosimRpc.so" \
  "$CIRCT_BUILD/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/lib/libCosimRpc.so"
```

### Debugging the test environment

Do **not** modify test infrastructure, package import logic, or simulator lookup paths just because the runtime/test invocation is unclear. In particular, do not patch files like `tests/conftest.py`, `esiaccel/cosim/simulator.py`, or other harness code as a substitute for figuring out the correct local setup.

Before changing any infrastructure, first verify what Python and pytest are actually using:

```bash
cd <repo-root>
source .venv/bin/activate

export CIRCT_BUILD=$PWD/build/default
export LD_LIBRARY_PATH=$CIRCT_BUILD/lib:$CIRCT_BUILD/tools/circt/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CIRCT_BUILD/lib:$CIRCT_BUILD/tools/circt/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$CIRCT_BUILD/tools/circt/python_packages/pycde:$CIRCT_BUILD/tools/circt/lib/Dialect/ESI/runtime/python
export PATH=$CIRCT_BUILD/bin:$PWD/ext/bin:$PATH

python3 -c 'import sys, pycde, esiaccel, esiaccel.codegen; print(sys.executable); print(pycde.__file__); print(esiaccel.__file__); print(esiaccel.codegen.__file__)'
python3 -m pytest --version
python3 -m pytest lib/Dialect/ESI/runtime/tests/unit/test_types.py -q
```

If the wrong `pycde` or `esiaccel` package is being imported, fix the environment outside the repo first:
- point `CIRCT_BUILD` at the correct `build/<name>` directory
- or set `PYTHONPATH`/`LD_LIBRARY_PATH`/`PATH` correctly in the shell running pytest
- confirm the imported module paths again before rerunning tests

Only change infrastructure or test harness files when there is a demonstrated product bug in that infrastructure and not merely uncertainty about how to run the tests locally.

### GTest unit tests

The GTest-based unit tests live in `lib/Dialect/ESI/runtime/tests/cpp` and they are built with the ESIRuntimeCppTests target. They can be run independently or they are run by pytest if the binary is present.

## Key architecture notes

### ESI Type system (`Types.h`)
- `Type` is the root class; `getID()` returns a unique string -- the MLIR type string in the case of pycde generation of the manifest, `toString()` returns a human-readable form.
- `TypeAliasType` wraps another type with a name — always unwrap (possibly recursively) before type-checking.
- `BitVectorType` is the parent of `BitsType` (signless) and `IntegerType`; `IntegerType` is the parent of `SIntType` and `UIntType`. Use `getWidth()` for bit width.
- `VoidType` serializes as 1 byte (zero) by convention since all cosim messages must have data and that's persisted to include DMA engines. This may change in the future.
- Wire byte count for integral types: `(bitWidth + 7) / 8`.

### Port classes (`Ports.h`)
- `WriteChannelPort` / `ReadChannelPort` are the untyped byte-stream ports.
- `BundlePort` aggregates named channels; `getRawWrite()`/`getRawRead()` return channel ports.
- `BundlePort::getAs<T>()` dynamic-casts to service port types like `FuncService::Function` or `CallService::Callback`.

### Service ports (`Services.h`)
- `FuncService::Function`: call-style port with `call(MessageData) → future<MessageData>`. Has `getArgType()`/`getResultType()`.
- `CallService::Callback`: reverse direction — accelerator calls host. `connect(std::function<MessageData(const MessageData&)>)`.

### MessageData serialization
- `MessageData::from(T&)` copies `sizeof(T)` bytes — this does NOT match the ESI wire format for non-power-of-two bit widths (e.g., si24 needs 3 bytes but int32_t is 4).
- `MessageData::as<T>()` requires exact size match or throws.
- For correct framing, use the port type's bit width to compute wire bytes: `(bitWidth+7)/8`.
- Sign extension for signed types with non-byte-aligned widths (e.g., si4, si22): the sign bit is at position `(bitWidth-1)`, not necessarily at bit 7 of the last byte. Must memcpy first, then OR in the high bits: `val |= (~T(0)) << bitWidth`.

### C++ codegen
- `esiaccel.codegen` generates C++ headers from the ESI manifest.
- Generated structs have `static constexpr std::string_view _ESI_ID` matching the MLIR type ID of the inner type (not the TypeAlias wrapper).
- Generated headers include `types.h` (sibling) and a per-module header (e.g., `LoopbackIP.h`).

### PyCDE version compatibility
- When testing locally, install PyCDE from a compatible build or ask the maintainer for a wheel.
- The `@cosim_test` decorator in `esiaccel.cosim.pytest` handles: running the PyCDE script → generating HW + manifest → compiling the simulator → launching cosim → injecting `host`/`port`/`sources_dir` into the test function.

### Test infrastructure
- Integration tests: `lib/Dialect/ESI/runtime/tests/integration/`
  - `hw/`: PyCDE hardware scripts (e.g., `loopback.py`)
  - `sw/`: C++ test binaries and their CMakeLists.txt
  - `test_*.py`: pytest files using `@cosim_test`
- The `sw/CMakeLists.txt` builds C++ test binaries against the ESI runtime. It searches for the runtime headers/libs using `ESI_RUNTIME_ROOT`.
- `check_lines(stdout, expected)` asserts substrings appear in order.
