// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --esi-connect-services | circt-opt | FileCheck %s --check-prefix=CONN

esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
}
// CHECK-LABEL: esi.service.decl @HostComms {
// CHECK:         esi.service.to_server @Send : !esi.channel<!esi.any>
// CHECK:         esi.service.to_client @Recv : !esi.channel<i8>

hw.module @Top (%clk: i1) -> (chksum: i8) {
  %c = esi.service.instance "topComms" @HostComms (%clk) : (i1) -> (i8)
  hw.instance "m1" @Loopback (clk: %clk: i1) -> ()
  hw.output %c : i8
}
// CHECK-LABEL: hw.module @Top(%clk: i1) -> (chksum: i8) {
// CHECK:         %0 = esi.service.instance "topComms" @HostComms(%clk) : (i1) -> i8
// CHECK:         hw.instance "m1" @Loopback(clk: %clk: i1) -> ()

// CONN-LABEL: hw.module @Top(%clk: i1) -> (chksum: i8) {
// CONN:         [[R0:%.+]]:2 = esi.service.impl_req "topComms" @HostComms(%clk, %m1.loopback_fromhw) : (i1, !esi.channel<i8>) -> (i8, !esi.channel<i8>) {
// CONN:         ^bb0(%arg0: !esi.channel<i8>):
// CONN:           %1 = esi.service.req.to_client <@HostComms::@Recv>(["m1", "loopback_tohw"]) : !esi.channel<i8>
// CONN:           esi.service.req.to_server %arg0 -> <@HostComms::@Send>(["m1", "loopback_fromhw"]) : !esi.channel<i8>
// CONN:         }
// CONN:         %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: i1, loopback_tohw: [[R0]]#1: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
// CONN:         hw.output [[R0]]#0 : i8

hw.module @Loopback (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
}
// CHECK-LABEL: hw.module @Loopback(%clk: i1) {
// CHECK:         %0 = esi.service.req.to_client <@HostComms::@Recv>(["loopback_tohw"]) : !esi.channel<i8>
// CHECK:         esi.service.req.to_server %0 -> <@HostComms::@Send>(["loopback_fromhw"]) : !esi.channel<i8>

// CONN-LABEL: hw.module @Loopback(%clk: i1, %loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>) {
// CONN:         hw.output %loopback_tohw : !esi.channel<i8>
