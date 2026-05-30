# CIRCT Standalone Example

This is a minimal out-of-tree CIRCT project, modeled after MLIR's
`examples/standalone`, for bootstrapping a dialect, pass library, optimizer
driver, and plugin.

Configure it against a CIRCT build or install tree:

```sh
cd examples/circt-standalone
mkdir build-circt-standalone
CIRCT_BUILD_DIR=<path-to-circt-build-or-install>
# This the same as CIRCT_BUILD_DIR when unified build is used
MLIR_BUILD_DIR=<path-to-mlir-build-or-install>
cmake -G Ninja -S . -B build-circt-standalone \
  -DCIRCT_DIR=$CIRCT_BUILD_DIR/lib/cmake/circt \
  -DMLIR_DIR=$MLIR_BUILD_DIR/lib/cmake/mlir
```

Build the optimizer, plugin, and tests:

```sh
ninja -C build-circt-standalone check-circt-standalone
```

Like MLIR's `examples/standalone`, this example is not built by the in-tree
examples target.  Configure it separately as shown above.

The plugin can be loaded into `circt-opt`:

```sh
circt-opt input.mlir \
  --load-dialect-plugin=build-circt-standalone/lib/CIRCTStandalonePlugin.so \
  --pass-pipeline='builtin.module(hw.module(circt-standalone-rename-wires))'
```
