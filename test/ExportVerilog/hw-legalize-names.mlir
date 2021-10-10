// RUN: circt-opt -export-verilog  %s | FileCheck %s

// CHECK-LABEL: sv.interface @output_0
sv.interface @output {
  // CHECK-NEXT: sv.interface.signal @input_0 : i1
  sv.interface.signal @input : i1
  // CHECK-NEXT: sv.interface.signal @output_1 : i1
  sv.interface.signal @output : i1
  // CHECK-NEXT: sv.interface.modport @always_2
  // CHECK-SAME: ("input" @input_0, "output" @output_1)
  sv.interface.modport @always ("input" @input, "output" @output)
}

// TODO: Renaming the above interface declarations currently does not rename
// their use in the following types.

// hw.module @InterfaceAsInstance () {
//   %0 = sv.interface.instance : !sv.interface<@output>
// }
// hw.module @InterfaceInPort (%m: !sv.modport<@output::@always>) {
// }
