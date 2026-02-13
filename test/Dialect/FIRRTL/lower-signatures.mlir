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

//CHECK-LABEL: firrtl.circuit "Domains"
firrtl.circuit "Domains" {
  firrtl.domain @ClockDomain
  // CHECK:      firrtl.module @Foo
  // CHECK-SAME:   in %A: !firrtl.domain of @ClockDomain
  // CHECK-SAME:   in %a_b: !firrtl.uint<1> domains [%A]
  // CHECK-SAME:   in %a_c: !firrtl.uint<1> domains [%A]
  // CHECK-SAME:   in %d_0: !firrtl.uint<1> domains [%D]
  // CHECK-SAME:   in %d_1: !firrtl.uint<1> domains [%D]
  // CHECK-SAME:   in %D: !firrtl.domain of @ClockDomain
  firrtl.module @Foo(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.bundle<b: uint<1>, c: uint<1>> domains [%A],
    in %d: !firrtl.vector<uint<1>, 2> domains [%D],
    in %D: !firrtl.domain of @ClockDomain
  ) attributes {
    convention = #firrtl<convention scalarized>
  } {
  }
  // CHECK: firrtl.module @Domains
  firrtl.module @Domains() {
    // CHECK:      firrtl.instance foo @Foo
    // CHECK-SAME:   in A: !firrtl.domain of @ClockDomain
    // CHECK-SAME:   in a_b: !firrtl.uint<1> domains [A]
    // CHECK-SAME:   in a_c: !firrtl.uint<1> domains [A]
    // CHECK-SAME:   in d_0: !firrtl.uint<1> domains [D]
    // CHECK-SAME:   in d_1: !firrtl.uint<1> domains [D]
    // CHECK-SAME:   in D: !firrtl.domain of @ClockDomain
    firrtl.instance foo @Foo(
      in A: !firrtl.domain of @ClockDomain,
      in a: !firrtl.bundle<b: uint<1>, c: uint<1>> domains [A],
      in d: !firrtl.vector<uint<1>, 2> domains [D],
      in D: !firrtl.domain of @ClockDomain
    )
  }
}

// CHECK-LABEL: firrtl.circuit "InstanceChoice"
firrtl.circuit "InstanceChoice" {
  firrtl.domain @ClockDomain
  firrtl.option @Platform {
    firrtl.option_case @FPGA
    firrtl.option_case @ASIC
  }

  // CHECK: firrtl.module @Target
  // CHECK-SAME: in %a_x: !firrtl.uint<1>
  // CHECK-SAME: in %a_y: !firrtl.uint<2>
  firrtl.module @Target(in %a: !firrtl.bundle<x: uint<1>, y: uint<2>>)
    attributes {convention = #firrtl<convention scalarized>} {
  }

  firrtl.module @FPGATarget(in %a: !firrtl.bundle<x: uint<1>, y: uint<2>>)
    attributes {convention = #firrtl<convention scalarized>} {
  }

  firrtl.module @ASICTarget(in %a: !firrtl.bundle<x: uint<1>, y: uint<2>>)
    attributes {convention = #firrtl<convention scalarized>} {
  }

  // CHECK: firrtl.module @TargetWithDomain
  // CHECK-SAME: in %D: !firrtl.domain of @ClockDomain
  // CHECK-SAME: in %b_x: !firrtl.uint<1> domains [%D]
  // CHECK-SAME: in %b_y: !firrtl.uint<2> domains [%D]
  firrtl.module @TargetWithDomain(
    in %D: !firrtl.domain of @ClockDomain,
    in %b: !firrtl.bundle<x: uint<1>, y: uint<2>> domains [%D]
  ) attributes {convention = #firrtl<convention scalarized>} {
  }

  firrtl.module @FPGATargetWithDomain(
    in %D: !firrtl.domain of @ClockDomain,
    in %b: !firrtl.bundle<x: uint<1>, y: uint<2>> domains [%D]
  ) attributes {convention = #firrtl<convention scalarized>} {
  }

  firrtl.module @ASICTargetWithDomain(
    in %D: !firrtl.domain of @ClockDomain,
    in %b: !firrtl.bundle<x: uint<1>, y: uint<2>> domains [%D]
  ) attributes {convention = #firrtl<convention scalarized>} {
  }

  // CHECK: firrtl.module @InstanceChoice
  firrtl.module @InstanceChoice() {
    // CHECK: firrtl.instance_choice inst @Target alternatives @Platform
    // CHECK-SAME: in a_x: !firrtl.uint<1>
    // CHECK-SAME: in a_y: !firrtl.uint<2>
    %inst_a = firrtl.instance_choice inst @Target alternatives @Platform
      { @FPGA -> @FPGATarget, @ASIC -> @ASICTarget }
      (in a: !firrtl.bundle<x: uint<1>, y: uint<2>>)
    // CHECK: firrtl.instance_choice inst2 @TargetWithDomain alternatives @Platform
    // CHECK-SAME: in D: !firrtl.domain of @ClockDomain
    // CHECK-SAME: in b_x: !firrtl.uint<1> domains [D]
    // CHECK-SAME: in b_y: !firrtl.uint<2> domains [D]
    %inst2_D, %inst2_b = firrtl.instance_choice inst2 @TargetWithDomain alternatives @Platform
      { @FPGA -> @FPGATargetWithDomain, @ASIC -> @ASICTargetWithDomain }
      (in D: !firrtl.domain of @ClockDomain, in b: !firrtl.bundle<x: uint<1>, y: uint<2>> domains [D])
  }
}
