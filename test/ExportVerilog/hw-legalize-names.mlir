// RUN: circt-opt -export-verilog  %s | FileCheck %s

hw.module @B(%a: i1) -> () {
}

// CHECK-LABEL: hw.module @TestDupInstanceName
hw.module @TestDupInstanceName(%a: i1) {
  // CHECK: hw.instance "name"
  hw.instance "name" @B(a: %a: i1) -> ()

  // CHECK: hw.instance "name_0"
  hw.instance "name" @B(a: %a: i1) -> ()
}

// CHECK-LABEL: TestEmptyInstanceName
hw.module @TestEmptyInstanceName(%a: i1) {
  // CHECK: hw.instance "_T"
  hw.instance "" @B(a: %a: i1) -> ()

  // CHECK: hw.instance "_T_0"
  hw.instance "" @B(a: %a: i1) -> ()
}

// CHECK-LABEL: hw.module @TestInstanceNameValueConflict
hw.module @TestInstanceNameValueConflict(%a: i1) {
  // CHECK:  %name = sv.wire
  %name = sv.wire : !hw.inout<i1>
  // CHECK:  %output_0 = sv.wire
  %output = sv.wire : !hw.inout<i1>
  // CHECK:  %input_1 = sv.reg
  %input = sv.reg : !hw.inout<i1>
  // CHECK: hw.instance "name_2"
  hw.instance "name" @B(a: %a: i1) -> ()
}

// https://github.com/llvm/circt/issues/855
// CHECK-LABEL: hw.module @nameless_reg
// CHECK-NEXT: %_T = sv.reg : !hw.inout<i4>
hw.module @nameless_reg(%a: i1) -> () {
  %661 = sv.reg : !hw.inout<i4>
}

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

// This is made collide with the first renaming attempt of the `@inout` module
// above.
hw.module.extern @inout_0 () -> ()
hw.module.extern @inout_1 () -> ()
hw.module.extern @inout_2 () -> ()
