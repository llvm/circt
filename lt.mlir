firrtl.circuit "TopLevel"  {
  firrtl.module @TopLevel(out %sink_a : !firrtl.uint<1>, out %sink_b : !firrtl.uint<2>, out %sink_c : !firrtl.uint<3>) {
    %a = firrtl.wire : !firrtl.bundle<a: bundle<a: uint<1>>, b: uint<2>>
    %b = firrtl.wire : !firrtl.vector<uint<3>, 4>
    %0 = firrtl.subfield %a("a") : (!firrtl.bundle<a: bundle<a: uint<1>>, b: uint<2>>) -> !firrtl.bundle<a: uint<1>>
    %1 = firrtl.subfield %0("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %a("b") : (!firrtl.bundle<a: bundle<a: uint<1>>, b: uint<2>>) -> !firrtl.uint<2>
    %3 = firrtl.subindex %b[3] : !firrtl.vector<uint<3>, 4>
    firrtl.connect %sink_a, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %sink_b, %2 : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.connect %sink_c, %3 : !firrtl.uint<3>, !firrtl.uint<3>
  }

  firrtl.module @WireBundlesWithBulkConnect(in %source: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>,
                             out %sink: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) {
    // CHECK: %w_valid = firrtl.wire  : !firrtl.uint<1>
    // CHECK: %w_ready = firrtl.wire  : !firrtl.uint<1>
    // CHECK: %w_data = firrtl.wire  : !firrtl.uint<64>
    %w = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
    // CHECK: firrtl.connect %w_valid, %source_valid : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %source_ready, %w_ready : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %w_data, %source_data : !firrtl.uint<64>, !firrtl.uint<64>
    firrtl.connect %w, %source : !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
    // CHECK: firrtl.connect %sink_valid, %w_valid : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %w_ready, %sink_ready : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %sink_data, %w_data : !firrtl.uint<64>, !firrtl.uint<64>
    firrtl.connect %sink, %w : !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  }

  firrtl.module @LowerVectors(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    firrtl.connect %b, %a: !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }

  firrtl.module @LowerVectorsIndRead(in %a: !firrtl.vector<uint<1>, 2>, in %i : !firrtl.uint<2>, out %b: !firrtl.uint<1>) {
    %0 = firrtl.subaccess %a[%i] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<2>
    firrtl.connect %b, %0: !firrtl.uint<1>, !firrtl.uint<1>
  }

  firrtl.module @LowerVectorsIndWrite(out %a: !firrtl.vector<uint<1>, 2>, in %i : !firrtl.uint<2>, in %b: !firrtl.uint<1>) {
    %0 = firrtl.subaccess %a[%i] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<2>
    firrtl.connect %0, %b: !firrtl.uint<1>, !firrtl.uint<1>
  }

  
// Check that a non-bundled mux ops are untouched.
    // check-label: firrtl.module @Mux
    firrtl.module @Mux(in %p: !firrtl.uint<1>, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
      // check-next: %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // check-next: firrtl.connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: firrtl.module @MuxBundle
    firrtl.module @MuxBundle(in %p: !firrtl.uint<1>, in %a: !firrtl.bundle<a: uint<1>>, in %b: !firrtl.bundle<a: uint<1>>, out %c: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %0 = firrtl.mux(%p, %a_a, %b_a) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %c_a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>) -> !firrtl.bundle<a: uint<1>>
      firrtl.connect %c, %0 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    }

    // CHECK-LABEL: firrtl.module @NodeBundle
    firrtl.module @NodeBundle(in %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.uint<1>) {
      // CHECK-NEXT: %n_a = firrtl.node %a_a  : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b, %n_a : !firrtl.uint<1>, !firrtl.uint<1>
      %n = firrtl.node %a : !firrtl.bundle<a: uint<1>>
      %n_a = firrtl.subfield %n("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %b, %n_a : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: firrtl.module @RegBundle(in %a_a: !firrtl.uint<1>, in %clk: !firrtl.clock, out %b_a: !firrtl.uint<1>) {
    firrtl.module @RegBundle(in %a: !firrtl.bundle<a: uint<1>>, in %clk: !firrtl.clock, out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = firrtl.reg %clk : (!firrtl.clock) -> !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %x_a, %a_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b_a, %x_a : !firrtl.uint<1>, !firrtl.uint<1>
      %x = firrtl.reg %clk {name = "x"} : (!firrtl.clock) -> !firrtl.bundle<a: uint<1>>
      %0 = firrtl.subfield %x("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      %1 = firrtl.subfield %a("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %b("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      %3 = firrtl.subfield %x("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: firrtl.module @RegBundleWithBulkConnect(in %a_a: !firrtl.uint<1>, in %clk: !firrtl.clock, out %b_a: !firrtl.uint<1>) {
    firrtl.module @RegBundleWithBulkConnect(in %a: !firrtl.bundle<a: uint<1>>, in %clk: !firrtl.clock, out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = firrtl.reg %clk : (!firrtl.clock) -> !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %x_a, %a_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b_a, %x_a : !firrtl.uint<1>, !firrtl.uint<1>
      %x = firrtl.reg %clk {name = "x"} : (!firrtl.clock) -> !firrtl.bundle<a: uint<1>>
      firrtl.connect %x, %a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
      firrtl.connect %b, %x : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    }


  // CHECK-LABEL: firrtl.module @LowerRegResetOp
  firrtl.module @LowerRegResetOp(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %init = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset, %init {name = "r"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
    firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %init_0 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   %init_1 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %r_0 = firrtl.regreset %clock, %reset, %init_0 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK:   %r_1 = firrtl.regreset %clock, %reset, %init_1 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK:   firrtl.connect %r_0, %a_d_0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %r_1, %a_d_1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %a_q_0, %r_0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %a_q_1, %r_1 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK-LABEL: firrtl.module @LowerRegResetOpNoName
  firrtl.module @LowerRegResetOpNoName(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %init = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset, %init {name = ""} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
    firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %init_0 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   %init_1 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %0 = firrtl.regreset %clock, %reset, %init_0 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK:   %1 = firrtl.regreset %clock, %reset, %init_1 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK:   firrtl.connect %0, %a_d_0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %1, %a_d_1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %a_q_0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %a_q_1, %1 : !firrtl.uint<1>, !firrtl.uint<1>


// Test RegOp lowering without name attribute
// https://github.com/llvm/circt/issues/795
  // CHECK-LABEL: firrtl.module @lowerRegOpNoName
  firrtl.module @lowerRegOpNoName(in %clock: !firrtl.clock, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %r = firrtl.reg %clock {name = ""} : (!firrtl.clock) -> !firrtl.vector<uint<1>, 2>
      firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
      firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
 // CHECK:    %0 = firrtl.reg %clock : (!firrtl.clock) -> !firrtl.uint<1>
 // CHECK:    %1 = firrtl.reg %clock : (!firrtl.clock) -> !firrtl.uint<1>
 // CHECK:    firrtl.connect %0, %a_d_0 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK:    firrtl.connect %1, %a_d_1 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK:    firrtl.connect %a_q_0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK:    firrtl.connect %a_q_1, %1 : !firrtl.uint<1>, !firrtl.uint<1>


  // CHECK-LABEL: firrtl.module @ArgTest
  // CHECK-SAME: in %[[SOURCE_VALID_NAME:source_valid]]: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[SOURCE_READY_NAME:source_ready]]: [[SOURCE_READY_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: in %[[SOURCE_DATA_NAME:source_data]]: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // CHECK-SAME: out %[[SINK_VALID_NAME:sink_valid]]: [[SINK_VALID_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: in %[[SINK_READY_NAME:sink_ready]]: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[SINK_DATA_NAME:sink_data]]: [[SINK_DATA_TYPE:!firrtl.uint<64>]]
  firrtl.module @ArgTest(in %source: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>,
                        out %sink: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) {

    // CHECK-NEXT: firrtl.when %[[SOURCE_VALID_NAME]]
    // CHECK-NEXT:   firrtl.connect %[[SINK_DATA_NAME]], %[[SOURCE_DATA_NAME]] : [[SINK_DATA_TYPE]], [[SOURCE_DATA_TYPE]]
    // CHECK-NEXT:   firrtl.connect %[[SINK_VALID_NAME]], %[[SOURCE_VALID_NAME]] : [[SINK_VALID_TYPE]], [[SOURCE_VALID_TYPE]]
    // CHECK-NEXT:   firrtl.connect %[[SOURCE_READY_NAME]], %[[SINK_READY_NAME]] : [[SOURCE_READY_TYPE]], [[SINK_READY_TYPE]]

    %0 = firrtl.subfield %source("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %source("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %source("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
    %3 = firrtl.subfield %sink("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %4 = firrtl.subfield %sink("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %5 = firrtl.subfield %sink("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
    firrtl.when %0 {
      firrtl.connect %5, %2 : !firrtl.uint<64>, !firrtl.uint<64>
      firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %1, %4 : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}
