// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(firrtl-sfc-compat))' --verify-diagnostics --split-input-file %s | FileCheck %s

firrtl.circuit "SFCCompatTests" {

  firrtl.module @SFCCompatTests() {}

  // An invalidated regreset should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidValue
  firrtl.module @InvalidValue(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %invalid_ui1  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated through a wire should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidThroughWire
  firrtl.module @InvalidThroughWire(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %inv  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated via an output port should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidPort
  firrtl.module @InvalidPort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>, out %x: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %x  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidate via an instance input port should be converted to a
  // reg.
  //
  // CHECK-LABEL: @InvalidInstancePort
  firrtl.module @InvalidInstancePort_Submodule(in %inv: !firrtl.uint<1>) {}
  firrtl.module @InvalidInstancePort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %submodule_inv = firrtl.instance submodule  @InvalidInstancePort_Submodule(in inv: !firrtl.uint<1>)
    firrtl.connect %submodule_inv, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %submodule_inv  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A primitive operation should block invalid propagation.
  firrtl.module @InvalidPrimop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %0 = firrtl.not %invalid_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: firrtl.regreset %clock
    %r = firrtl.regreset %clock, %reset, %0  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalid value should NOT propagate through a node.
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>) {
    %inv = firrtl.wire  : !firrtl.uint<8>
    %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    firrtl.connect %inv, %invalid_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
    %_T = firrtl.node %inv  : !firrtl.uint<8>
    // CHECK: firrtl.regreset %clock
    %r = firrtl.regreset %clock, %reset, %_T  : !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %r, %d : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %q, %r : !firrtl.uint<8>, !firrtl.uint<8>
  }

  firrtl.module @AggregateInvalid(out %q: !firrtl.bundle<a:uint<1>>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.bundle<a:uint<1>>
    firrtl.connect %q, %invalid_ui1 : !firrtl.bundle<a:uint<1>>, !firrtl.bundle<a:uint<1>>
    // CHECK: %c0_ui1 = firrtl.constant 0
    // CHECK-NEXT: %[[CAST:.+]] = firrtl.bitcast %c0_ui1
    // CHECK-NEXT: %q, %[[CAST]]
  }

}
