// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-signatures))' %s | FileCheck --check-prefixes=CHECK %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-signatures))' --mlir-print-debuginfo %s | FileCheck --check-prefixes=CHECK-LOC %s

firrtl.circuit "Prop" {
  // CHECK-LABEL @Prop(out %y: !firrtl.string)
  firrtl.module @Prop(out %y: !firrtl.string) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.string "test"
    // CHECK: firrtl.propassign
    firrtl.propassign %y, %0 : !firrtl.string
  }

  firrtl.module private @emptyVec(in %vi : !firrtl.vector<uint<4>, 0>, out %vo : !firrtl.vector<uint<4>, 0>) attributes {convention = #firrtl<convention scalarized>} {
    firrtl.matchingconnect %vo, %vi : !firrtl.vector<uint<4>, 0>
  }

  // CHECK-LABEL: @Annos
  // CHECK-SAME: in %x: !firrtl.uint<1> [{class = "circt.test", pin = "pin0"}],
  // CHECK-SAME: in %y_a: !firrtl.uint<1> [{class = "circt.test", pin = "pin1"}],
  // CHECK-SAME: in %y_b: !firrtl.uint<2> [{class = "circt.test", pin = "pin2"}])
  firrtl.module private @Annos(
    in %x: !firrtl.uint<1> [{circt.fieldID = 0 : i64, class = "circt.test", pin = "pin0"}],
    in %y: !firrtl.bundle<a: uint<1>, b: uint<2>> [{circt.fieldID = 2 : i64, class = "circt.test", pin = "pin2"}, {circt.fieldID = 1 : i64, class = "circt.test", pin = "pin1"}]
  )  attributes {convention = #firrtl<convention scalarized>} {
  }

  // CHECK-LABEL: @AnalogBlackBox
  firrtl.extmodule private @AnalogBlackBox<index: ui32 = 0>(out bus: !firrtl.analog<32>) attributes {convention = #firrtl<convention scalarized>, defname = "AnalogBlackBox"}
  firrtl.module @AnalogBlackBoxModule(out %io: !firrtl.bundle<bus: analog<32>>) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: %io = firrtl.wire interesting_name : !firrtl.bundle<bus: analog<32>>
    // CHECK: %0 = firrtl.subfield %io[bus] : !firrtl.bundle<bus: analog<32>>
    // CHECK: firrtl.attach %0, %io_bus : !firrtl.analog<32>, !firrtl.analog<32>
    %0 = firrtl.subfield %io[bus] : !firrtl.bundle<bus: analog<32>>
    %impl_bus = firrtl.instance impl interesting_name @AnalogBlackBox(out bus: !firrtl.analog<32>)
    firrtl.attach %0, %impl_bus : !firrtl.analog<32>, !firrtl.analog<32>
  }

  // CHECK-LABEL: firrtl.module private @Bar
  firrtl.module private @Bar(out %in1: !firrtl.bundle<a flip: uint<1>, b flip: uint<1>>, in %in2: !firrtl.bundle<c: uint<1>>, in %out: !firrtl.bundle<d flip: uint<1>, e flip: uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK-NEXT: %in1 = firrtl.wire
    // CHECK-NEXT: %in2 = firrtl.wire
    // CHECK-NEXT: %out = firrtl.wire
  }

  // CHECK-LABEL: firrtl.module private @Foo
  firrtl.module private @Foo() attributes {convention = #firrtl<convention scalarized>} {
    %bar_in1, %bar_in2, %bar_out = firrtl.instance bar interesting_name @Bar(out in1: !firrtl.bundle<a flip: uint<1>, b flip: uint<1>>, in in2: !firrtl.bundle<c: uint<1>>, in out: !firrtl.bundle<d flip: uint<1>, e flip: uint<1>>)
    // CHECK: %bar.in1 = firrtl.wire
    // CHECK: %bar.in2 = firrtl.wire 
    // CHECK: %bar.out = firrtl.wire
  }

}

// Instances should preserve their location.
// See https://github.com/llvm/circt/issues/6535
firrtl.circuit "PreserveLocation" {
  firrtl.extmodule @Foo()
  // CHECK-LOC-LABEL: firrtl.module @PreserveLocation
  firrtl.module @PreserveLocation() {
    // CHECK-LOC: firrtl.instance foo @Foo() loc([[LOC:#.+]])
    firrtl.instance foo @Foo() loc(#instLoc)
  } loc(#moduleLoc)
}
// CHECK-LOC: [[LOC]] = loc("someLoc":9001:1)
#moduleLoc = loc("wrongLoc":42:1)
#instLoc = loc("someLoc":9001:1)

// Internal paths should be expanded
firrtl.circuit "InternalPaths"  {
  // CHECK: firrtl.extmodule private @BlackBox
  // CHECK-SAME: out bundle_a: !firrtl.uint<32>, out bundle_b: !firrtl.uint<23>
  // CHECK-SAME: out array_0: !firrtl.uint<1>, out array_1: !firrtl.uint<1>
  // CHECK-SAME: out probe: !firrtl.probe<uint<32>>
  // CHECK-SAME: internalPaths = [
  // CHECK-SAME: #firrtl.internalpath
  // CHECK-SAME: #firrtl.internalpath
  // CHECK-SAME: #firrtl.internalpath
  // CHECK-SAME: #firrtl.internalpath
  // CHECK-SAME: #firrtl.internalpath<"some_probe">
  // CHECK-SAME: ]
  firrtl.extmodule private @BlackBox(
    out bundle : !firrtl.bundle<a: uint<32>, b: uint<23>>,
    out array : !firrtl.vector<uint<1>, 2>,
    out probe : !firrtl.probe<uint<32>>
  ) attributes {
    convention = #firrtl<convention scalarized>,
    internalPaths = [
      #firrtl.internalpath,
      #firrtl.internalpath,
      #firrtl.internalpath<"some_probe">
    ]
  }

  // CHECK-LABEL: @InternalPaths
  firrtl.module @InternalPaths() {
    %bundle, %array, %probe = firrtl.instance blackbox @BlackBox(
      out bundle : !firrtl.bundle<a: uint<32>, b: uint<23>>,
      out array : !firrtl.vector<uint<1>, 2>,
      out probe : !firrtl.probe<uint<32>>
    )
  }
}
