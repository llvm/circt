# Yosys Integration

Yosys is de-facto standard as an opensource RTL synthesis and verification tool.
The purpose of the yosys integration is to serve as a baseline for circt-based synthesis flow.

## Build
Yosys integration is disabled by default. To enable Yosys integration, you need to build CIRCT with `-DCIRCT_YOSYS_INTEGRATION_ENABLED=ON`.

## Tranlation between RTLIL

`circt-translate` provides subcommand for translation between CIRCT IR and RTLIL.

* CIRCT IR -> RTLIL

```bash
circt-tranlsate --export-rtlil
```
Currently only subest of core dialects are translated during translation to RTLIL.

* RTLIL -> CIRCT IR

```bash
circt-tranlsate --import-rtlil
```

## Run Yosys passes on CIRCT IR

To run Yosys passes on CIRCT there are two passes `yosys-optimizer` and `yosys-optimizer-parallel`.

```bash
# Run synth (canonicalizer is currently required to get "clean" core dialect operations)
circt-opt  -pass-pipeline='builtin.module(yosys-optimizer{passes=synth},canonicalize)'

# Pipe Yosys log to the stderr via `redirect-log=true`.
circt-opt  -pass-pipeline='builtin.module(yosys-optimizer{passes=synth redirect-log=true})'

# Run synth_xilinx (symbol-dce is recommended to remove unused external modules)
circt-opt  -pass-pipeline='builtin.module(yosys-optimizer{passes=synth_xilinx},canonicalize,symbol-dce)'

# Run multiple yosys pass (comma separated).
circt-opt  -pass-pipeline='builtin.module(yosys-optimizer{passes=write_rtlil,write_verilog,synth},canonicalize)'

# Emit Verilog using yosys backend. It's recommended to run `opt` for the fair comparison
circt-opt  -pass-pipeline='builtin.module(yosys-optimizer{passes=opt,write_verilog})'
```

### CIRCT as a parallel Yosys driver

Yosys has a globally context which is a not thread-safe so we cannot parallelly run `yosys-optimizer` on each HW module. As a workaround CIRCT provides a `yosys-optimizer-parallel` pass that parallelly invokes yosys in child processes. `yosys-optimizer-parallel` cannot be used for transformation that requires module hierarchly (e.g. inlining/flattening etc)

## Testing

Testing Yosys integration is tricky since RTLIL textual format could differ between Yosys versions. Currently we test the correctness of RTLIL translation by running LEC on the CIRCT IR after import with the original CIRCT IR after export.
