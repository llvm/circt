// REQUIRES: esi-cosim
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-rtl | circt-translate --emit-verilog > %t1.sv
// RUN: circt-translate %s -emit-esi-capnp -verify-diagnostics > %t2.capnp
// RUN: esi-cosim-runner.py --schema %t2.capnp %s %t1.sv
// PY: import loopback as test
// PY: rpc = test.LoopbackTester(rpcschemapath, simhostport)
// PY: rpc.test_i32(25)

rtl.module @top(%clk:i1, %rstn:i1) -> () {
  %cosimRecv = esi.cosim %clk, %rstn, %bufferedResp, 1 {name="TestEP"} : !esi.channel<i32> -> !esi.channel<i32>
  %bufferedResp = esi.buffer %clk, %rstn, %cosimRecv {stages=1} : i32
}
