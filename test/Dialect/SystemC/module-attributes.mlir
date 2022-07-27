// RUN: circt-opt %s --mlir-print-op-generic | circt-opt --mlir-print-op-generic | FileCheck %s

// CHECK-LABEL: "systemc.module"
// CHECK: ^bb0(%arg0: i4, %arg1: i32, %arg2: i4, %arg3: i8):
// CHECK: function_type = (i4, i32, i4, i8) -> (),
// CHECK: portDirections = #systemc.port_directions<[sc_out, sc_in, sc_out, sc_inout]>,
// CHECK: portNames = ["port0", "port1", "port2", "port3"],
systemc.module @mixedPorts (sc_out %port0: i4, sc_in %port1: i32, sc_out %port2: i4, sc_inout %port3: i8) {}
