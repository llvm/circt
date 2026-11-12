// RUN: circt-opt %s --export-verilog --verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false -o %t.mlir > /dev/null

// These widths are legal as parameters, but not as DPI function results.
sv.func private @signed_vector(out result: si7 {sv.func.explicitly_returned})
// expected-error @+1 {{DPI integer return must have width 1, 8, 16, 32, or 64}}
sv.func.dpi.import @signed_vector

// -----

sv.func private @unsigned_vector(out result: ui65 {sv.func.explicitly_returned})
// expected-error @+1 {{DPI integer return must have width 1, 8, 16, 32, or 64}}
sv.func.dpi.import @unsigned_vector

// -----

sv.func private @signless_vector(out result: i1024 {sv.func.explicitly_returned})
// expected-error @+1 {{DPI integer return must have width 1, 8, 16, 32, or 64}}
sv.func.dpi.import @signless_vector
