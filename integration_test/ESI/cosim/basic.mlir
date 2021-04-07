// REQUIRES: esi-cosim
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-rtl | circt-translate --export-verilog > %t1.sv
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics > %t2.capnp
// RUN: esi-cosim-runner.py --schema %t2.capnp %s %t1.sv %S/../supplements/integers.sv
// PY: import basic
// PY: rpc = basic.BasicSystemTester(rpcschemapath, simhostport)
// PY: rpc.testIntAcc(25)
// PY: rpc.testVectorSum(25)
// PY: rpc.testCrypto(25)

rtl.module.extern @IntAccNoBP(%clk: i1, %rstn: i1, %ints: !esi.channel<i32>) -> (%totalOut: !esi.channel<i32>)
rtl.module.extern @IntArrSum(%clk: i1, %rstn: i1, %arr: !esi.channel<!rtl.array<4 x si13>>) -> (%totalOut: !esi.channel<!rtl.array<2 x ui24>>)

rtl.module @ints(%clk: i1, %rstn: i1) {
  %intsIn = esi.cosim %clk, %rstn, %intsTotalBuffered, 1 {name="TestEP"} : !esi.channel<i32> -> !esi.channel<i32>
  %intsInBuffered = esi.buffer %clk, %rstn, %intsIn {stages=2, name="intChan"} : i32
  %intsTotal = rtl.instance "acc" @IntAccNoBP(%clk, %rstn, %intsInBuffered) : (i1, i1, !esi.channel<i32>) -> (!esi.channel<i32>)
  %intsTotalBuffered = esi.buffer %clk, %rstn, %intsTotal {stages=2, name="totalChan"} : i32
}

rtl.module @array(%clk: i1, %rstn: i1) {
  %arrIn = esi.cosim %clk, %rstn, %arrTotalBuffered, 2 {name="TestEP"} : !esi.channel<!rtl.array<2 x ui24>> -> !esi.channel<!rtl.array<4 x si13>>
  %arrInBuffered = esi.buffer %clk, %rstn, %arrIn {stages=2, name="arrChan"} : !rtl.array<4 x si13>
  %arrTotal = rtl.instance "acc" @IntArrSum(%clk, %rstn, %arrInBuffered) : (i1, i1, !esi.channel<!rtl.array<4 x si13>>) -> (!esi.channel<!rtl.array<2 x ui24>>)
  %arrTotalBuffered = esi.buffer %clk, %rstn, %arrTotal {stages=2, name="totalChan"} : !rtl.array<2 x ui24>
}

!DataPkt = type !rtl.struct<encrypted: i1, blob: !rtl.array<32 x i8>>
!pktChan = type !esi.channel<!DataPkt>
!Config  = type !rtl.struct<encrypt:   i1, otp:  !rtl.array<32 x i8>>
!cfgChan = type !esi.channel<!Config>

rtl.module.extern @Encryptor(%clk: i1, %rstn: i1, %in: !pktChan, %cfg: !cfgChan) -> (%x: !pktChan)

rtl.module @structs(%clk:i1, %rstn:i1) -> () {
  %compressedData = rtl.instance "otpCryptor" @Encryptor(%clk, %rstn, %inputData, %cfg) : (i1, i1, !pktChan, !cfgChan) -> !pktChan
  %inputData = esi.cosim %clk, %rstn, %compressedData, 3 {name="CryptoData"} : !pktChan -> !pktChan
  %c0 = rtl.constant 0 : i1
  %null, %nullReady = esi.wrap.vr %c0, %c0 : i1
  %cfg = esi.cosim %clk, %rstn, %null, 4 {name="CryptoConfig"} : !esi.channel<i1> -> !cfgChan
}

rtl.module @top(%clk: i1, %rstn: i1) {
  rtl.instance "ints" @ints (%clk, %rstn) : (i1, i1) -> ()
  rtl.instance "array" @array(%clk, %rstn) : (i1, i1) -> ()
  rtl.instance "structs" @structs(%clk, %rstn) : (i1, i1) -> ()
}
