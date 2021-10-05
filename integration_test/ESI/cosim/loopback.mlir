// REQUIRES: esi-cosim
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --hw-legalize-names | circt-opt --export-verilog > %t1.sv
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics > %t2.capnp
// RUN: esi-cosim-runner.py --schema %t2.capnp %s %t1.sv
// PY: import loopback as test
// PY: rpc = test.LoopbackTester(rpcschemapath, simhostport)
// PY: rpc.test_i32(25)
// PY: rpc.test_keytext(25)

hw.module @intLoopback(%clk:i1, %rstn:i1) -> () {
  %cosimRecv = esi.cosim %clk, %rstn, %bufferedResp, 1 {name="IntTestEP"} : !esi.channel<i32> -> !esi.channel<i32>
  %bufferedResp = esi.buffer %clk, %rstn, %cosimRecv {stages=1} : i32
}

!KeyText = type !hw.struct<text: !hw.array<6xi14>, key: !hw.array<4xi8>>
hw.module @twoListLoopback(%clk:i1, %rstn:i1) -> () {
  %cosim = esi.cosim %clk, %rstn, %resp, 2 {name="KeyTextEP"} : !esi.channel<!KeyText> -> !esi.channel<!KeyText>
  %resp = esi.buffer %clk, %rstn, %cosim {stages=4} : !KeyText
}

hw.module @top(%clk:i1, %rstn:i1) -> () {
  hw.instance "intLoopbackInst" @intLoopback(clk: %clk: i1, rstn: %rstn: i1) -> ()
  hw.instance "twoListLoopbackInst" @twoListLoopback(clk: %clk: i1, rstn: %rstn: i1) -> ()
}
