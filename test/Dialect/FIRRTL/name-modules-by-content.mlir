// RUN: circt-opt -split-input-file -pass-pipeline='builtin.module(firrtl.circuit(firrtl-name-modules-by-content))' %s | FileCheck %s
// RUN: circt-opt -split-input-file -pass-pipeline='builtin.module(firrtl.circuit(firrtl-name-modules-by-content{hash-length=12}))' %s | FileCheck %s --check-prefix=LONG

// Two different queues, numbered in the order they were created, are named by
// their contents. The suffix is gone, and the parent's instances follow. A
// module whose name nothing else shares keeps it.

// CHECK-LABEL: firrtl.circuit "Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module private @[[$WIDE:Queue_[0-9a-f]{8}]](
  // LONG: firrtl.module private @{{Queue_[0-9a-f]{12}[(]}}
  firrtl.module private @Queue_3(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }
  // CHECK: firrtl.module private @[[$NARROW:Queue_[0-9a-f]{8}]](
  firrtl.module private @Queue(in %in: !firrtl.uint<4>, out %out: !firrtl.uint<4>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<4>
  }
  // CHECK: firrtl.module private @Lonely_4(
  firrtl.module private @Lonely_4() {}
  // CHECK: firrtl.module @Top(
  firrtl.module @Top(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<4>) {
    // CHECK: firrtl.instance q8 @[[$WIDE]](
    %q8_in, %q8_out = firrtl.instance q8 @Queue_3(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    // CHECK: firrtl.instance q4 @[[$NARROW]](
    %q4_in, %q4_out = firrtl.instance q4 @Queue(in in: !firrtl.uint<4>, out out: !firrtl.uint<4>)
    // CHECK: firrtl.instance l @Lonely_4()
    firrtl.instance l @Lonely_4()
    firrtl.matchingconnect %q8_in, %a : !firrtl.uint<8>
    firrtl.matchingconnect %q4_in, %b : !firrtl.uint<4>
  }
}

// -----

// The same two queues created in the other order, under another parent, with
// an annotation on one: the same names as above.

// CHECK-LABEL: firrtl.circuit "Other"
firrtl.circuit "Other" {
  // CHECK: firrtl.module private @[[$NARROW]](
  firrtl.module private @Queue_1(in %in: !firrtl.uint<4>, out %out: !firrtl.uint<4>) attributes {annotations = [{class = "Unrelated"}]} {
    firrtl.matchingconnect %out, %in : !firrtl.uint<4>
  }
  // CHECK: firrtl.module private @[[$WIDE]](
  firrtl.module private @Queue_2(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }
  firrtl.module private @Wrapper(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<4>) {
    // CHECK: firrtl.instance x @[[$NARROW]](
    %x_in, %x_out = firrtl.instance x @Queue_1(in in: !firrtl.uint<4>, out out: !firrtl.uint<4>)
    // CHECK: firrtl.instance y @[[$WIDE]](
    %y_in, %y_out = firrtl.instance y @Queue_2(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %x_in, %b : !firrtl.uint<4>
    firrtl.matchingconnect %y_in, %a : !firrtl.uint<8>
  }
  firrtl.module @Other(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<4>) {
    %w_a, %w_b = firrtl.instance w @Wrapper(in a: !firrtl.uint<8>, in b: !firrtl.uint<4>)
    firrtl.matchingconnect %w_a, %a : !firrtl.uint<8>
    firrtl.matchingconnect %w_b, %b : !firrtl.uint<4>
  }
}

// -----

// The modules a module instantiates are black boxes: when the queue inside a
// wrapper is renumbered and changed, the wrapper keeps its name. This circuit
// and the next differ only in the 8-bit queue's suffix and contents.

// CHECK-LABEL: firrtl.circuit "Inside"
firrtl.circuit "Inside" {
  firrtl.module private @Queue(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }
  firrtl.module private @Queue_1(in %in: !firrtl.uint<4>, out %out: !firrtl.uint<4>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<4>
  }
  // CHECK: firrtl.module private @[[$WRAPPER:Wrapper_[0-9a-f]{8}]](
  firrtl.module private @Wrapper(in %a: !firrtl.uint<8>) {
    %q_in, %q_out = firrtl.instance q @Queue(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %q_in, %a : !firrtl.uint<8>
  }
  firrtl.module private @Wrapper_1(in %a: !firrtl.uint<4>) {
    %q_in, %q_out = firrtl.instance q @Queue_1(in in: !firrtl.uint<4>, out out: !firrtl.uint<4>)
    firrtl.matchingconnect %q_in, %a : !firrtl.uint<4>
  }
  firrtl.module @Inside(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<4>) {
    %w_a = firrtl.instance w @Wrapper(in a: !firrtl.uint<8>)
    %v_a = firrtl.instance v @Wrapper_1(in a: !firrtl.uint<4>)
    firrtl.matchingconnect %w_a, %a : !firrtl.uint<8>
    firrtl.matchingconnect %v_a, %b : !firrtl.uint<4>
  }
}

// -----

// CHECK-LABEL: firrtl.circuit "Inside"
firrtl.circuit "Inside" {
  // CHECK-NOT: @[[$WIDE]](
  firrtl.module private @Queue_2(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %out, %c0_ui8 : !firrtl.uint<8>
  }
  firrtl.module private @Queue_1(in %in: !firrtl.uint<4>, out %out: !firrtl.uint<4>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<4>
  }
  // CHECK: firrtl.module private @[[$WRAPPER]](
  firrtl.module private @Wrapper(in %a: !firrtl.uint<8>) {
    %q_in, %q_out = firrtl.instance q @Queue_2(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %q_in, %a : !firrtl.uint<8>
  }
  firrtl.module private @Wrapper_1(in %a: !firrtl.uint<4>) {
    %q_in, %q_out = firrtl.instance q @Queue_1(in in: !firrtl.uint<4>, out out: !firrtl.uint<4>)
    firrtl.matchingconnect %q_in, %a : !firrtl.uint<4>
  }
  firrtl.module @Inside(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<4>) {
    %w_a = firrtl.instance w @Wrapper(in a: !firrtl.uint<8>)
    %v_a = firrtl.instance v @Wrapper_1(in a: !firrtl.uint<4>)
    firrtl.matchingconnect %w_a, %a : !firrtl.uint<8>
    firrtl.matchingconnect %v_a, %b : !firrtl.uint<4>
  }
}

// -----

// Two wrappers that differ only in which of two queues with the same ports
// they instantiate are told apart by the queues' new names.

// CHECK-LABEL: firrtl.circuit "Tie"
firrtl.circuit "Tie" {
  // CHECK: firrtl.module private @[[$PASS:Queue_[0-9a-f]{8}]](
  firrtl.module private @Queue(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }
  // CHECK: firrtl.module private @[[$ZERO:Queue_[0-9a-f]{8}]](
  firrtl.module private @Queue_1(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %out, %c0_ui8 : !firrtl.uint<8>
  }
  // CHECK: firrtl.module private @[[$W0:Wrapper_[0-9a-f]{8}]](
  // CHECK: firrtl.instance q @[[$PASS]](
  firrtl.module private @Wrapper(in %a: !firrtl.uint<8>) {
    %q_in, %q_out = firrtl.instance q @Queue(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %q_in, %a : !firrtl.uint<8>
  }
  // CHECK-NOT: @[[$W0]](
  // CHECK: firrtl.module private @{{Wrapper_[0-9a-f]{8}[(]}}
  // CHECK: firrtl.instance q @[[$ZERO]](
  firrtl.module private @Wrapper_1(in %a: !firrtl.uint<8>) {
    %q_in, %q_out = firrtl.instance q @Queue_1(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %q_in, %a : !firrtl.uint<8>
  }
  firrtl.module @Tie(in %a: !firrtl.uint<8>) {
    %w_a = firrtl.instance w @Wrapper(in a: !firrtl.uint<8>)
    %v_a = firrtl.instance v @Wrapper_1(in a: !firrtl.uint<8>)
    firrtl.matchingconnect %w_a, %a : !firrtl.uint<8>
    firrtl.matchingconnect %v_a, %a : !firrtl.uint<8>
  }
}

// -----

// Public modules keep the name the user gave them, external modules and
// modules marked NoDedup keep theirs; the rest of their families is renamed.

// CHECK-LABEL: firrtl.circuit "Kept"
firrtl.circuit "Kept" {
  // CHECK: firrtl.extmodule private @Black_1(
  firrtl.extmodule private @Black_1(in in: !firrtl.uint<1>)
  // CHECK: firrtl.module private @{{Black_[0-9a-f]{8}[(]}}
  firrtl.module private @Black() {}
  // CHECK: firrtl.module private @Pinned_2(
  firrtl.module private @Pinned_2() attributes {annotations = [{class = "firrtl.transforms.NoDedupAnnotation"}]} {}
  // CHECK: firrtl.module private @{{Pinned_[0-9a-f]{8}[(]}}
  firrtl.module private @Pinned() {}
  // CHECK: firrtl.module @Kept(
  firrtl.module @Kept() {
    %b_in = firrtl.instance b @Black_1(in in: !firrtl.uint<1>)
    firrtl.instance c @Black()
    firrtl.instance p @Pinned_2()
    firrtl.instance q @Pinned()
    firrtl.instance k @Kept_1()
  }
  // CHECK: firrtl.module private @{{Kept_[0-9a-f]{8}[(]}}
  firrtl.module private @Kept_1() {}
}

// -----

// Hierarchical paths through a renamed module, and the alternatives of an
// instance choice, follow the new names.

// CHECK-LABEL: firrtl.circuit "Paths"
firrtl.circuit "Paths" {
  // CHECK: hw.hierpath private @nla [@Paths::@child, @[[$CHILD:Child_[0-9a-f]{8}]]::@w]
  hw.hierpath private @nla [@Paths::@child, @Child_5::@w]
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }
  // CHECK: firrtl.module private @[[$CHILD]](
  firrtl.module private @Child_5() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "Tagged"}]} : !firrtl.uint<1>
  }
  firrtl.module private @Child() {}
  // CHECK: firrtl.module private @[[$DEFAULT:Impl_[0-9a-f]{8}]](
  firrtl.module private @Impl_1(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }
  // CHECK: firrtl.module private @[[$FPGA:Impl_[0-9a-f]{8}]](
  firrtl.module private @Impl_2(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %out, %c0_ui8 : !firrtl.uint<8>
  }
  firrtl.module @Paths(in %x: !firrtl.uint<8>) {
    // CHECK: firrtl.instance child sym @child @[[$CHILD]]()
    firrtl.instance child sym @child @Child_5()
    firrtl.instance other @Child()
    // CHECK: firrtl.instance_choice inst @[[$DEFAULT]] alternatives @Platform {
    // CHECK-SAME: @FPGA -> @[[$FPGA]]
    %inst_in, %inst_out = firrtl.instance_choice inst @Impl_1 alternatives @Platform {
      @FPGA -> @Impl_2
    } (in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %inst_in, %x : !firrtl.uint<8>
  }
}
