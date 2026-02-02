// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include connect-forwarder | FileCheck %s

firrtl.circuit "DontRemoveSyms" {
  // CHECK-LABEL: firrtl.module @DontRemoveSyms
  firrtl.module @DontRemoveSyms(in %input: !firrtl.uint<42>) {
    // CHECK-NEXT: %wireWithSym = firrtl.wire
    // CHECK-NOT: %wireWithoutSym = firrtl.wire
    %wireWithSym = firrtl.wire sym @sym : !firrtl.uint<42>
    %wireWithoutSym = firrtl.wire : !firrtl.uint<42>

    // CHECK-NOT: firrtl.connect
    firrtl.connect %wireWithSym, %input : !firrtl.uint<42>, !firrtl.uint<42>
    firrtl.connect %wireWithoutSym, %input : !firrtl.uint<42>, !firrtl.uint<42>

    // CHECK-NEXT: dbg.variable "wireWithSym", %input
    // CHECK-NEXT: dbg.variable "wireWithoutSym", %input
    dbg.variable "wireWithSym", %wireWithSym : !firrtl.uint<42>
    dbg.variable "wireWithoutSym", %wireWithoutSym : !firrtl.uint<42>
  }
}

firrtl.circuit "ForwardThroughRegs" {
  // CHECK-LABEL: firrtl.module @ForwardThroughRegs
  firrtl.module @ForwardThroughRegs(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %input: !firrtl.uint<42>) {
    // CHECK-NOT: %reg0 = firrtl.reg
    // CHECK-NOT: %reg1 = firrtl.regreset
    %reg0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<42>
    %reg1 = firrtl.regreset %clock, %reset, %input : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
    // CHECK-NOT: firrtl.connect
    firrtl.connect %reg0, %input : !firrtl.uint<42>, !firrtl.uint<42>
    firrtl.connect %reg1, %input : !firrtl.uint<42>, !firrtl.uint<42>
    // CHECK-NEXT: dbg.variable "reg0", %input
    // CHECK-NEXT: dbg.variable "reg1", %input
    dbg.variable "reg0", %reg0 : !firrtl.uint<42>
    dbg.variable "reg1", %reg1 : !firrtl.uint<42>
  }
}

firrtl.circuit "LayerblockCrossBlock" {
  firrtl.layer @A bind {
  }
  // CHECK-LABEL: firrtl.module @LayerblockCrossBlock
  firrtl.module @LayerblockCrossBlock() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %wire = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %wire_0 = firrtl.wire {name = "wire"} : !firrtl.uint<1>
      // This should not crash when connect-forwarder checks cross-block ordering
      firrtl.matchingconnect %wire_0, %c0_ui1 : !firrtl.uint<1>
      %0 = firrtl.ref.send %wire_0 : !firrtl.uint<1>
      %1 = firrtl.ref.cast %0 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %wire, %1 : !firrtl.probe<uint<1>, @A>
    }
  }
}
