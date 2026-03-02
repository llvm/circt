# ESI Runtime development guide

## Standalone builds (no LLVM/CIRCT build required)

The ESI runtime can be built independently using `lib/Dialect/ESI/runtime/CMakeLists.txt` as the project root. This is **much** faster than a full CIRCT build and is the preferred workflow for runtime-only changes.

### Setup

```bash
# Create a venv with matching Python version and install build deps.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools nanobind pytest

# Install PyCDE from PyPI (prereleases needed to match HEAD).
pip install --pre pycde

# System prerequisites for cosim: gRPC and protobuf C++ libraries must be
# installed (e.g., apt install libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
# on Debian/Ubuntu, or via vcpkg/brew). These are required by CMake's
# find_package(gRPC) and find_package(Protobuf) when ESI_COSIM=ON.

# Configure and build the ESI runtime (with cosim).
cd lib/Dialect/ESI/runtime
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DESI_COSIM=ON
ninja -C build ESIRuntime
```

This builds: `ESICppRuntime` (shared lib), `CosimBackend`, `CosimRpc`, `EsiCosimDpiServer`, `esiquery`, the Python native extension (`esiCppAccel`), and the Python package under `build/python/esiaccel/`. The `esitester` tool is not part of the `ESIRuntime` target; build it separately with `ninja -C build esitester` if you need it.

### Running the integration pytests

The cosim pytests need several environment variables:

```bash
cd <repo-root>
source .venv/bin/activate
export LD_LIBRARY_PATH=$PWD/lib/Dialect/ESI/runtime/build:$PWD/lib/Dialect/ESI/runtime/build/lib
export LIBRARY_PATH=$LD_LIBRARY_PATH   # for verilator linking
export PATH=$PWD/lib/Dialect/ESI/runtime/build:$PATH  # for esiquery

python3 -m pytest lib/Dialect/ESI/runtime/tests/ -v
```

**Important:** The `esiaccel` package must be importable by subprocess children (forked by the cosim pytest framework). Two options:
1. Install editable: `pip install -e lib/Dialect/ESI/runtime --no-build-isolation` — but note this uses the *source* tree's Python files, which lack the cosim `.sv` files that only exist in the build tree.
2. Use a `.pth` file pointing to the build output:
   ```bash
   SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
   echo "$PWD/lib/Dialect/ESI/runtime/build/python" > "$SITE/esiaccel-build.pth"
   ```

Option 2 is recommended because the build tree's `esiaccel/` has the correct cosim SystemVerilog files, driver, and native extension.

Additionally, the verilator linker needs `libEsiCosimDpiServer.so`. The cosim pytest framework looks for it at `<esiaccel-package>/lib/`. If using the build tree via `.pth`, create symlinks:
```bash
mkdir -p lib/Dialect/ESI/runtime/build/python/esiaccel/lib
cd lib/Dialect/ESI/runtime/build/python/esiaccel/lib
ln -sf ../../lib/libEsiCosimDpiServer.so .
ln -sf ../../libESICppRuntime.so .
ln -sf ../../libCosimBackend.so .
ln -sf ../../libCosimRpc.so .
```

### GTest unit tests

The GTest-based unit tests live in `unittests/Dialect/ESI/runtime/` and require a full CIRCT build (they link against `esiaccel::ESICppRuntime` via the CIRCT cmake infrastructure). They can't be built from the standalone runtime cmake.

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
