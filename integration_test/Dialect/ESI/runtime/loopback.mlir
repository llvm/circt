// REQUIRES: esi-cosim, esi-runtime, rtl-sim
// RUN: rm -rf %t6 && mkdir %t6 && cd %t6
// RUN: circt-opt %s --esi-connect-services --esi-appid-hier=top=top --esi-build-manifest=top=top --esi-clean-metadata > %t4.mlir
// RUN: circt-opt %t4.mlir --lower-esi-to-physical --lower-esi-bundles --lower-esi-ports --lower-esi-to-hw=platform=cosim --lower-seq-to-sv --export-split-verilog -o %t3.mlir
// RUN: cd ..
// RUN: %python %s.py trace w:%t6/esi_system_manifest.json
// RUN: %python esi-cosim-runner.py --exec %s.py %t6/*.sv

!sendI8 = !esi.bundle<[!esi.channel<i8> to "send"]>
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>
!sendI0 = !esi.bundle<[!esi.channel<i0> to "send"]>
!recvI0 = !esi.bundle<[!esi.channel<i0> to "recv"]>

!anyFrom = !esi.bundle<[
  !esi.channel<!esi.any> from "recv",
  !esi.channel<!esi.any> to "send"]>

esi.service.decl @HostComms {
  esi.service.to_server @Send : !sendI8
  esi.service.to_client @Recv : !recvI8
}

esi.service.decl @MyService {
  esi.service.to_server @Send : !sendI0
  esi.service.to_client @Recv : !recvI0
}

hw.module @Loopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req.to_client <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) {esi.appid=#esi.appid<"loopback_tohw">} : !recvI8
  %dataOut = esi.bundle.unpack from %dataInBundle : !recvI8
  %dataOutBundle = esi.bundle.pack %dataOut : !sendI8
  esi.service.req.to_server %dataOutBundle -> <@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !sendI8

  %send = esi.service.req.to_client <@MyService::@Recv> (#esi.appid<"mysvc_recv">) : !recvI0
  %send_ch = esi.bundle.unpack from %send : !recvI0
  %sendi0_bundle = esi.bundle.pack %send_ch : !sendI0
  esi.service.req.to_server %sendi0_bundle -> <@MyService::@Send> (#esi.appid<"mysvc_send">) : !sendI0
}

esi.service.std.func @funcs

!structFunc = !esi.bundle<[
  !esi.channel<!hw.struct<a: ui4, b: si8>> to "arg",
  !esi.channel<!hw.array<1xsi8>> from "result"]>

hw.module @LoopbackStruct() {
  %callBundle = esi.service.req.to_client <@funcs::@call> (#esi.appid<"structFunc">) : !structFunc
  %arg = esi.bundle.unpack %resultChan from %callBundle : !structFunc

  %argData, %valid = esi.unwrap.vr %arg, %ready : !hw.struct<a: ui4, b: si8>
  %resultElem = hw.struct_extract %argData["b"] : !hw.struct<a: ui4, b: si8>
  %resultArray = hw.array_create %resultElem : si8
  %resultChan, %ready = esi.wrap.vr %resultArray, %valid : !hw.array<1xsi8>
}

esi.mem.ram @MemA i64 x 20
!write = !hw.struct<address: i5, data: i64>
!writeBundle = !esi.bundle<[!esi.channel<!write> to "req", !esi.channel<i0> from "ack"]>

hw.module @MemoryAccess1(in %clk : !seq.clock, in %rst : i1) {
  esi.service.instance #esi.appid<"mem"> svc @MemA impl as "sv_mem" (%clk, %rst) : (!seq.clock, i1) -> ()
  %write_struct = hw.aggregate_constant [0 : i5, 0 : i64] : !write
  %valid = hw.constant 0 : i1
  %write_ch, %ready = esi.wrap.vr %write_struct, %valid : !write
  %writeBundle, %done = esi.bundle.pack %write_ch : !writeBundle
  esi.service.req.to_server %writeBundle -> <@MemA::@write> (#esi.appid<"internal_write">) : !writeBundle
}

!func1Signature = !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
hw.module @CallableFunc1() {
  %call = esi.service.req.to_client <@funcs::@call> (#esi.appid<"func1">) : !func1Signature
  %arg = esi.bundle.unpack %arg from %call : !func1Signature
}

esi.manifest.sym @Loopback name "LoopbackIP" version "v0.0" summary "IP which simply echos bytes" {foo=1}

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  esi.service.instance #esi.appid<"cosim"> svc @HostComms impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  esi.service.instance #esi.appid<"cosim_default"> impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[0]>}
  hw.instance "m2" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[1]>}
  hw.instance "int_mem" @MemoryAccess1 (clk: %clk: !seq.clock, rst: %rst: i1) -> ()
  hw.instance "func1" @CallableFunc1() -> ()
  hw.instance "loopback_struct" @LoopbackStruct() -> ()
}
