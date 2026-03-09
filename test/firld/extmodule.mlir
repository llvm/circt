// RUN: split-file %s %t
// RUN: firld %t/Top.mlir %t/Extmodule.mlir --base-circuit Top | FileCheck %s

//--- Top.mlir
module {
  // CHECK: firrtl.circuit "Top"
  firrtl.circuit "Top" {
    firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %i: !firrtl.bundle<ready flip: uint<1>, valid: uint<1>, bits: uint<16>>, out %o: !firrtl.bundle<ready flip: uint<1>, valid: uint<1>, bits: uint<16>>) attributes {convention = #firrtl<convention scalarized>} {
      %io = firrtl.wire : !firrtl.bundle<clock flip: clock, reset flip: uint<1>, i flip: bundle<ready flip: uint<1>, valid: uint<1>, bits: uint<16>>, o: bundle<ready flip: uint<1>, valid: uint<1>, bits: uint<16>>>
      %fifo_clk, %fifo_rst_n, %fifo_push_req_n, %fifo_pop_req_n, %fifo_diag_n, %fifo_data_in, %fifo_empty, %fifo_almost_empty, %fifo_half_full, %fifo_almost_full, %fifo_full, %fifo_error, %fifo_data_out = firrtl.instance fifo interesting_name @Queue(in clk: !firrtl.clock, in rst_n: !firrtl.uint<1>, in push_req_n: !firrtl.uint<1>, in pop_req_n: !firrtl.uint<1>, in diag_n: !firrtl.uint<1>, in data_in: !firrtl.uint<16>, out empty: !firrtl.uint<1>, out almost_empty: !firrtl.uint<1>, out half_full: !firrtl.uint<1>, out almost_full: !firrtl.uint<1>, out full: !firrtl.uint<1>, out error: !firrtl.uint<1>, out data_out: !firrtl.uint<16>)
    }
    // CHECK-NOT: firrtl.extmodule @Queue(in clk: !firrtl.clock, in rst_n: !firrtl.uint<1>, in push_req_n: !firrtl.uint<1>, in pop_req_n: !firrtl.uint<1>, in diag_n: !firrtl.uint<1>, in data_in: !firrtl.uint<16>, out empty: !firrtl.uint<1>, out almost_empty: !firrtl.uint<1>, out half_full: !firrtl.uint<1>, out almost_full: !firrtl.uint<1>, out full: !firrtl.uint<1>, out error: !firrtl.uint<1>, out data_out: !firrtl.uint<16>) attributes {defname = "Queue"}
    firrtl.extmodule @Queue(in clk: !firrtl.clock, in rst_n: !firrtl.uint<1>, in push_req_n: !firrtl.uint<1>, in pop_req_n: !firrtl.uint<1>, in diag_n: !firrtl.uint<1>, in data_in: !firrtl.uint<16>, out empty: !firrtl.uint<1>, out almost_empty: !firrtl.uint<1>, out half_full: !firrtl.uint<1>, out almost_full: !firrtl.uint<1>, out full: !firrtl.uint<1>, out error: !firrtl.uint<1>, out data_out: !firrtl.uint<16>) attributes {defname = "Queue"}
  }
}

//--- Extmodule.mlir
module {
  // CHECK-NOT: firrtl.circuit "Queue"
  firrtl.circuit "Queue" {
    // CHECK: firrtl.extmodule @Queue<af_level: ui1 = 1, rst_mode: ui2 = 3, depth: ui6 = 32, width: ui5 = 16, ae_level: ui1 = 1, err_mode: ui2 = 2>(in clk: !firrtl.clock, in rst_n: !firrtl.uint<1>, in push_req_n: !firrtl.uint<1>, in pop_req_n: !firrtl.uint<1>, in diag_n: !firrtl.uint<1>, in data_in: !firrtl.uint<16>, out empty: !firrtl.uint<1>, out almost_empty: !firrtl.uint<1>, out half_full: !firrtl.uint<1>, out almost_full: !firrtl.uint<1>, out full: !firrtl.uint<1>, out error: !firrtl.uint<1>, out data_out: !firrtl.uint<16>) attributes {convention = #firrtl<convention scalarized>, defname = "FIFO"}
    firrtl.extmodule @Queue<af_level: ui1 = 1, rst_mode: ui2 = 3, depth: ui6 = 32, width: ui5 = 16, ae_level: ui1 = 1, err_mode: ui2 = 2>(in clk: !firrtl.clock, in rst_n: !firrtl.uint<1>, in push_req_n: !firrtl.uint<1>, in pop_req_n: !firrtl.uint<1>, in diag_n: !firrtl.uint<1>, in data_in: !firrtl.uint<16>, out empty: !firrtl.uint<1>, out almost_empty: !firrtl.uint<1>, out half_full: !firrtl.uint<1>, out almost_full: !firrtl.uint<1>, out full: !firrtl.uint<1>, out error: !firrtl.uint<1>, out data_out: !firrtl.uint<16>) attributes {convention = #firrtl<convention scalarized>, defname = "FIFO"}
  }
}
