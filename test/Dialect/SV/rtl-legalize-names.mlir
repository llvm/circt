// RUN: circt-opt -rtl-legalize-names %s | FileCheck %s

rtl.module @B(%a: i1) -> () {
}

// CHECK-LABEL: rtl.module @TestDupInstanceName
rtl.module @TestDupInstanceName(%a: i1) {
  // CHECK: rtl.instance "name"
  rtl.instance "name" @B(%a) : (i1) -> ()

  // CHECK: rtl.instance "name_0"
  rtl.instance "name" @B(%a) : (i1) -> ()
}

// CHECK-LABEL: TestEmptyInstanceName
rtl.module @TestEmptyInstanceName(%a: i1) {
  // CHECK: rtl.instance "_T"
  rtl.instance "" @B(%a) : (i1) -> ()

  // CHECK: rtl.instance "_T_0"
  rtl.instance "" @B(%a) : (i1) -> ()
}

// CHECK-LABEL: rtl.module @TestInstanceNameValueConflict
rtl.module @TestInstanceNameValueConflict(%a: i1) {
  // CHECK:  %name = sv.wire
  %name = sv.wire : !rtl.inout<i1>
  // CHECK:  %output_0 = sv.wire
  %output = sv.wire : !rtl.inout<i1>
  // CHECK:  %input_1 = sv.reg
  %input = sv.reg : !rtl.inout<i1>
  // CHECK: rtl.instance "name_2"
  rtl.instance "name" @B(%a) : (i1) -> ()
}

// https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// CHECK-LABEL: rtl.module @inout_3
// CHECK-SAME: (%inout_0: i1) -> (%output_1: i1)
// CHECK-NEXT: rtl.output %inout_0 : i1
rtl.module @inout(%inout: i1) -> (%output: i1) {
  rtl.output %inout : i1
}

// https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// CHECK-LABEL: rtl.module @reg_4
// CHECK-SAME: (%inout_0: i1) -> (%output_1: i1)
// CHECK-NEXT: rtl.output %inout_0 : i1
rtl.module @reg(%inout: i1) -> (%output: i1) {
  rtl.output %inout : i1
}

// CHECK-LABEL: rtl.module @inout_inst
// CHECK-NEXT: rtl.instance "foo" @inout_3
rtl.module @inout_inst(%a: i1) -> () {
  %0 = rtl.instance "foo" @inout (%a) : (i1) -> (i1)
}

// CHECK-LABEL: sv.interface @output_5
sv.interface @output {
  // CHECK-NEXT: sv.interface.signal @input_0 : i1
  sv.interface.signal @input : i1
  // CHECK-NEXT: sv.interface.signal @output_1 : i1
  sv.interface.signal @output : i1
  // CHECK-NEXT: sv.interface.modport @always_2
  // CHECK-SAME: ("input" @input_0, "output" @output_1)
  sv.interface.modport @always ("input" @input, "output" @output)
}

// Instantiate a module which has had its ports renamed.
// CHECK-LABEL: rtl.module @ModuleWithCollision(
// CHECK-SAME:    %reg_0: i1) -> (%wire_1: i1)
rtl.module @ModuleWithCollision(%reg: i1) -> (%wire: i1) {
  rtl.output %reg : i1
}
rtl.module @InstanceWithCollisions(%a: i1) {
  rtl.instance "parameter" @ModuleWithCollision(%a) : (i1) -> (i1)
}



// TODO: Renaming the above interface declarations currently does not rename
// their use in the following types.

// rtl.module @InterfaceAsInstance () {
//   %0 = sv.interface.instance : !sv.interface<@output>
// }
// rtl.module @InterfaceInPort (%m: !sv.modport<@output::@always>) {
// }

// This is made collide with the first renaming attempt of the `@inout` module
// above.
rtl.module.extern @inout_0 () -> ()
rtl.module.extern @inout_1 () -> ()
rtl.module.extern @inout_2 () -> ()


