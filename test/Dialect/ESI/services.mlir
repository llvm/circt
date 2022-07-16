// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt --esi-connect-services %s | circt-opt | FileCheck %s --check-prefix=CONN

// CHECK-LABEL: esi.service.decl @HostComms {
// CHECK:         esi.service.to_server @Send : !esi.channel<!esi.any>
// CHECK:         esi.service.to_client @Recv : !esi.channel<i8>
esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
}

// CHECK-LABEL: hw.module @Top(%clk: i1, %rst: i1) {
// CHECK:         esi.service.instance @HostComms impl as "cosim"(%clk, %rst) : (i1, i1) -> ()
// CHECK:         hw.instance "m1" @Loopback(clk: %clk: i1) -> ()

// CONN-LABEL: hw.module @Top(%clk: i1) {
// CONN:         [[r1:%.+]]:2 = esi.service.impl_req @HostComms impl as "cosim"(%clk, %m1.loopback_fromhw) : (i1, !esi.channel<i8>) -> (i8, !esi.channel<i8>) {
// CONN:         ^bb0(%arg0: !esi.channel<i8>):
// CONN:           [[r2:%.+]] = esi.service.req.to_client <@HostComms::@Recv>(["m1", "loopback_tohw"]) : !esi.channel<i8>
// CONN:           esi.service.req.to_server %arg0 -> <@HostComms::@Send>(["m1", "loopback_fromhw"]) : !esi.channel<i8>
// CONN:         }
// CONN:         %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: i1, loopback_tohw: [[r1]]#1: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
hw.module @Top (%clk: i1, %rst: i1) {
  esi.service.instance @HostComms impl as  "cosim" (%clk, %rst) : (i1, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: i1) -> ()
}

// CHECK-LABEL: hw.module @Loopback(%clk: i1) {
// CHECK:         %0 = esi.service.req.to_client <@HostComms::@Recv>(["loopback_tohw"]) : !esi.channel<i8>
// CHECK:         esi.service.req.to_server %0 -> <@HostComms::@Send>(["loopback_fromhw"]) : !esi.channel<i8>

// CONN-LABEL: hw.module @Loopback(%clk: i1, %loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>) {
// CONN:         hw.output %loopback_tohw : !esi.channel<i8>
hw.module @Loopback (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
}

// CONN-LABEL: hw.module @Top2(%clk: i1) -> (chksum: i8) {
// CONN:         [[r0:%.+]]:3 = esi.service.impl_req @HostComms impl as "topComms2"(%clk, %r1.m1.loopback_fromhw, %r1.p1.producedMsgChan, %r1.p2.producedMsgChan) : (i1, !esi.channel<i8>, !esi.channel<i8>, !esi.channel<i8>) -> (i8, !esi.channel<i8>, !esi.channel<i8>) {
// CONN:         ^bb0(%arg0: !esi.channel<i8>, %arg1: !esi.channel<i8>, %arg2: !esi.channel<i8>):
// CONN:           %1 = esi.service.req.to_client <@HostComms::@Recv>(["r1", "m1", "loopback_tohw"]) : !esi.channel<i8>
// CONN:           %2 = esi.service.req.to_client <@HostComms::@Recv>(["r1", "c1", "consumingFromChan"]) : !esi.channel<i8>
// CONN:           esi.service.req.to_server %arg0 -> <@HostComms::@Send>(["r1", "m1", "loopback_fromhw"]) : !esi.channel<i8>
// CONN:           esi.service.req.to_server %arg1 -> <@HostComms::@Send>(["r1", "p1", "producedMsgChan"]) : !esi.channel<i8>
// CONN:           esi.service.req.to_server %arg2 -> <@HostComms::@Send>(["r1", "p2", "producedMsgChan"]) : !esi.channel<i8>
// CONN:         }
// CONN:         %r1.m1.loopback_fromhw, %r1.p1.producedMsgChan, %r1.p2.producedMsgChan = hw.instance "r1" @Rec(clk: %clk: i1, m1.loopback_tohw: [[r0]]#1: !esi.channel<i8>, c1.consumingFromChan: [[r0]]#2: !esi.channel<i8>) -> (m1.loopback_fromhw: !esi.channel<i8>, p1.producedMsgChan: !esi.channel<i8>, p2.producedMsgChan: !esi.channel<i8>)
// CONN:         hw.output [[r0]]#0 : i8
hw.module @Top2 (%clk: i1) -> (chksum: i8) {
  %c = esi.service.instance @HostComms impl as  "topComms2" (%clk) : (i1) -> (i8)
  hw.instance "r1" @Rec(clk: %clk: i1) -> ()
  hw.output %c : i8
}

// CONN-LABEL: hw.module @Rec(%clk: i1, %m1.loopback_tohw: !esi.channel<i8>, %c1.consumingFromChan: !esi.channel<i8>) -> (m1.loopback_fromhw: !esi.channel<i8>, p1.producedMsgChan: !esi.channel<i8>, p2.producedMsgChan: !esi.channel<i8>) {
// CONN:         %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: i1, loopback_tohw: %m1.loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
// CONN:         %c1.rawData = hw.instance "c1" @Consumer(clk: %clk: i1, consumingFromChan: %c1.consumingFromChan: !esi.channel<i8>) -> (rawData: i8)
// CONN:         %p1.producedMsgChan = hw.instance "p1" @Producer(clk: %clk: i1) -> (producedMsgChan: !esi.channel<i8>)
// CONN:         %p2.producedMsgChan = hw.instance "p2" @Producer(clk: %clk: i1) -> (producedMsgChan: !esi.channel<i8>)
// CONN:         hw.output %m1.loopback_fromhw, %p1.producedMsgChan, %p2.producedMsgChan : !esi.channel<i8>, !esi.channel<i8>, !esi.channel<i8>
hw.module @Rec(%clk: i1) -> () {
  hw.instance "m1" @Loopback (clk: %clk: i1) -> ()
  hw.instance "c1" @Consumer(clk: %clk: i1) -> (rawData: i8)
  hw.instance "p1" @Producer(clk: %clk: i1) -> ()
  hw.instance "p2" @Producer(clk: %clk: i1) -> ()
}

// CONN-LABEL: hw.module @Consumer(%clk: i1, %consumingFromChan: !esi.channel<i8>) -> (rawData: i8) {
// CONN:         %true = hw.constant true
// CONN:         %rawOutput, %valid = esi.unwrap.vr %consumingFromChan, %true : i8
// CONN:         hw.output %rawOutput : i8
hw.module @Consumer(%clk: i1) -> (rawData: i8) {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["consumingFromChan"]) : !esi.channel<i8>
  %rdy = hw.constant 1 : i1
  %rawData, %valid = esi.unwrap.vr %dataIn, %rdy: i8
  hw.output %rawData : i8
}

// CONN-LABEL: hw.module @Producer(%clk: i1) -> (producedMsgChan: !esi.channel<i8>) {
// CONN:         %c0_i8 = hw.constant 0 : i8
// CONN:         %true = hw.constant true
// CONN:         %chanOutput, %ready = esi.wrap.vr %c0_i8, %true : i8
// CONN:         hw.output %chanOutput : !esi.channel<i8>
hw.module @Producer(%clk: i1) -> () {
  %data = hw.constant 0 : i8
  %valid = hw.constant 1 : i1
  %dataIn, %rdy = esi.wrap.vr %data, %valid : i8
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["producedMsgChan"]) : !esi.channel<i8>
}

// CONN-LABEL: msft.module @MsTop {} (%clk: i1) -> (chksum: i8)
// CONN:         [[r1:%.+]]:2 = esi.service.impl_req @HostComms impl as "topComms"(%clk, %m1.loopback_fromhw) : (i1, !esi.channel<i8>) -> (i8, !esi.channel<i8>) {
// CONN:         ^bb0(%arg0: !esi.channel<i8>):
// CONN:           [[r2:%.+]] = esi.service.req.to_client <@HostComms::@Recv>(["m1", "loopback_tohw"]) : !esi.channel<i8>
// CONN:           esi.service.req.to_server %arg0 -> <@HostComms::@Send>(["m1", "loopback_fromhw"]) : !esi.channel<i8>
// CONN:         }
// CONN:         %m1.loopback_fromhw = msft.instance @m1 @MsLoopback(%clk, [[r1]]#1) : (i1, !esi.channel<i8>) -> !esi.channel<i8>
// CONN:         msft.output [[r1]]#0 : i8
msft.module @MsTop {} (%clk: i1) -> (chksum: i8) {
  %c = esi.service.instance @HostComms impl as  "topComms" (%clk) : (i1) -> (i8)
  msft.instance @m1 @MsLoopback (%clk) : (i1) -> ()
  msft.output %c : i8
}

// CONN-LABEL: msft.module @MsLoopback {} (%clk: i1, %loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
// CONN:         msft.output %loopback_tohw : !esi.channel<i8>
msft.module @MsLoopback {} (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
  msft.output
}
