// REQUIRES: esi-cosim
// RUN: rm -rf %t6 && mkdir %t6 && cd %t6
// RUN: circt-opt %s --esi-connect-services --esi-appid-hier=top=top --esi-build-manifest=top=top --esi-clean-metadata > %t4.mlir
// RUN: circt-opt %t4.mlir --lower-esi-to-physical --lower-esi-bundles --lower-esi-ports --lower-esi-to-hw=platform=cosim --lower-seq-to-sv --export-split-verilog -o %t3.mlir
// RUN: cd ..
// RUN: esi-cosim-runner.py --exec %S/loopback.py %t6/*.sv

!sendI8 = !esi.bundle<[!esi.channel<i8> to "send"]>
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>
!sendI0 = !esi.bundle<[!esi.channel<i0> to "send"]>

esi.service.decl @HostComms {
  esi.service.to_server @Send : !sendI8
  esi.service.to_client @Recv : !recvI8
  esi.service.to_server @SendI0 : !sendI0
}

hw.module @Loopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req.to_client <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) {esi.appid=#esi.appid<"loopback_tohw">} : !recvI8
  %dataOut = esi.bundle.unpack from %dataInBundle : !recvI8
  %dataOutBundle = esi.bundle.pack %dataOut : !sendI8
  esi.service.req.to_server %dataOutBundle -> <@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !sendI8

  %c0_0 = hw.constant 0 : i0
  %c0_1 = hw.constant 0 : i1
  %sendi0_channel, %ready = esi.wrap.vr %c0_0, %c0_1 : i0
  %sendi0_bundle = esi.bundle.pack %sendi0_channel : !sendI0
  esi.service.req.to_server %sendi0_bundle -> <@HostComms::@SendI0> (#esi.appid<"loopback_fromhw_i0">) : !sendI0
}

esi.manifest.sym @Loopback name "LoopbackIP" version "v0.0" summary "IP which simply echos bytes" {foo=1}

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  esi.service.instance #esi.appid<"cosim"> svc @HostComms impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[0]>}
  hw.instance "m2" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[1]>}
}
