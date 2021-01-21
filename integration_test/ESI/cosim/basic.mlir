// REQUIRES: esi-cosim
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-rtl | circt-translate --export-verilog > %t1.sv
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics > %t2.capnp
// RUN: esi-cosim-runner.py --schema %t2.capnp %s %t1.sv %S/../supplements/integers.sv
// PY: import basic
// PY: rpc = basic.BasicSystemTester(rpcschemapath, simhostport)
// PY: rpc.testIntAcc(25)

rtl.externmodule @IntAccNoBP(%clk: i1, %rstn: i1, %ints: !esi.channel<i32>) -> (%totalOut: !esi.channel<i32>)

rtl.module @top(%clk: i1, %rstn: i1) {
  %intsIn = esi.cosim %clk, %rstn, %intsTotalBuffered, 1 {name="TestEP"} : !esi.channel<i32> -> !esi.channel<i32>
  %intsInBuffered = esi.buffer %clk, %rstn, %intsIn {stages=2, name="intChan"} : i32
  %intsTotal = rtl.instance "acc" @IntAccNoBP(%clk, %rstn, %intsInBuffered) : (i1, i1, !esi.channel<i32>) -> (!esi.channel<i32>)
  %intsTotalBuffered = esi.buffer %clk, %rstn, %intsTotal {stages=2, name="totalChan"} : i32
}
