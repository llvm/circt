// REQUIRES: rtl-sim
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-rtl -verify-diagnostics > %t1.mlir
// RUN: circt-translate %t1.mlir -emit-verilog -verify-diagnostics > %t2.sv
// RUN: circt-rtl-sim.py %t2.sv %INC%/circt/Dialect/ESI/ESIPrimitives.sv --cycles 25 2>&1 | FileCheck %s

module {
  rtl.externmodule @IntCountProd(%clk: i1, %rstn: i1) -> (%ints: !esi.channel<i32>)
  rtl.externmodule @IntAcc(%clk: i1, %rstn: i1, %ints: !esi.channel<i32>) -> ()
  rtl.module @top(%clk: i1, %rstn: i1) -> () {
    %intStream = rtl.instance "prod" @IntCountProd(%clk, %rstn) : (i1, i1) -> (!esi.channel<i32>)
    %intStreamBuffered = esi.buffer %clk, %rstn, %intStream {stages=2} : i32
    rtl.instance "acc" @IntAcc(%clk, %rstn, %intStreamBuffered) : (i1, i1, !esi.channel<i32>) -> ()
  }
}
