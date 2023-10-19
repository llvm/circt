// RUN: circt-opt --esi-connect-services  %s | circt-opt | FileCheck %s --check-prefix=CONN

!sendI8 = !esi.bundle<[!esi.channel<i8> to "send"]>
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>
!reqResp = !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i8> from "resp"]>

esi.service.decl @HostComms {
  esi.service.to_server @Send : !sendI8
  esi.service.to_client @Recv : !recvI8
  esi.service.to_client @ReqResp : !reqResp
}


// CONN-LABEL: hw.module @Top(in %clk : !seq.clock, in %rst : i1) {
// CONN-DAG:     [[R2:%.+]] = esi.cosim %clk, %rst, %m1.loopback_fromhw, "m1.loopback_fromhw" : !esi.channel<i8> -> !esi.channel<i1>
// CONN-DAG:     [[R0:%.+]] = esi.null : !esi.channel<i1>
// CONN-DAG:     [[R1:%.+]] = esi.cosim %clk, %rst, [[R0]], "m1.loopback_tohw" : !esi.channel<i1> -> !esi.channel<i8>
// CONN:         %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: !seq.clock, loopback_tohw: [[R1]]: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
hw.module @Top (in %clk: !seq.clock, in %rst: i1) {
  esi.service.instance impl as  "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> ()
}


// CONN-LABEL:  hw.module @Loopback(in %clk : !seq.clock, in %loopback_tohw : !esi.bundle<[!esi.channel<i8> to "recv"]>, out loopback_fromhw : !esi.bundle<[!esi.channel<i8> to "send"]>) {
// CONN-NEXT:     %recv = esi.bundle.unpack  from %loopback_tohw : !esi.bundle<[!esi.channel<i8> to "recv"]>
// CONN-NEXT:     %bundle = esi.bundle.pack %recv : !esi.bundle<[!esi.channel<i8> to "send"]>
// CONN-NEXT:     hw.output %bundle : !esi.bundle<[!esi.channel<i8> to "send"]>
hw.module @Loopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !recvI8
  %dataOut = esi.bundle.unpack from %dataInBundle : !recvI8
  %dataOutBundle = esi.bundle.pack %dataOut : !sendI8
  esi.service.req.to_server %dataOutBundle -> <@HostComms::@Send> (["loopback_fromhw"]) : !sendI8
}

// CONN-LABEL: hw.module @Top2(in %clk : !seq.clock, out chksum : i8) {
// CONN:         [[r0:%.+]]:3 = esi.service.impl_req svc @HostComms impl as "topComms2"(%clk) : (!seq.clock) -> (i8, !esi.channel<i8>, !esi.channel<i8>) {
// CONN-DAG:       esi.service.req.to_client <@HostComms::@Recv>(["r1", "m1", "loopback_tohw"]) : !esi.channel<i8>
// CONN-DAG:       esi.service.req.to_client <@HostComms::@Recv>(["r1", "c1", "consumingFromChan"]) : !esi.channel<i8>
// CONN-DAG:       esi.service.req.to_server %r1.m1.loopback_fromhw -> <@HostComms::@Send>(["r1", "m1", "loopback_fromhw"]) : !esi.channel<i8>
// CONN-DAG:       esi.service.req.to_server %r1.p1.producedMsgChan -> <@HostComms::@Send>(["r1", "p1", "producedMsgChan"]) : !esi.channel<i8>
// CONN-DAG:       esi.service.req.to_server %r1.p2.producedMsgChan -> <@HostComms::@Send>(["r1", "p2", "producedMsgChan"]) : !esi.channel<i8>
// CONN:         }
// CONN:         %r1.m1.loopback_fromhw, %r1.p1.producedMsgChan, %r1.p2.producedMsgChan = hw.instance "r1" @Rec(clk: %clk: !seq.clock, m1.loopback_tohw: [[r0]]#1: !esi.channel<i8>, c1.consumingFromChan: [[r0]]#2: !esi.channel<i8>) -> (m1.loopback_fromhw: !esi.channel<i8>, p1.producedMsgChan: !esi.channel<i8>, p2.producedMsgChan: !esi.channel<i8>)
// CONN:         hw.output [[r0]]#0 : i8
hw.module @Top2 (in %clk: !seq.clock, out chksum: i8) {
  %c = esi.service.instance svc @HostComms impl as  "topComms2" (%clk) : (!seq.clock) -> (i8)
  hw.instance "r1" @Rec(clk: %clk: !seq.clock) -> ()
  hw.output %c : i8
}

// CONN-LABEL: hw.module @Rec(in %clk : !seq.clock, in %m1.loopback_tohw : !esi.channel<i8>, in %c1.consumingFromChan : !esi.channel<i8>, out m1.loopback_fromhw : !esi.channel<i8>, out p1.producedMsgChan : !esi.channel<i8>, out p2.producedMsgChan : !esi.channel<i8>) {
// CONN:         %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: !seq.clock, loopback_tohw: %m1.loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
// CONN:         %c1.rawData = hw.instance "c1" @Consumer(clk: %clk: !seq.clock, consumingFromChan: %c1.consumingFromChan: !esi.channel<i8>) -> (rawData: i8)
// CONN:         %p1.producedMsgChan = hw.instance "p1" @Producer(clk: %clk: !seq.clock) -> (producedMsgChan: !esi.channel<i8>)
// CONN:         %p2.producedMsgChan = hw.instance "p2" @Producer(clk: %clk: !seq.clock) -> (producedMsgChan: !esi.channel<i8>)
// CONN:         hw.output %m1.loopback_fromhw, %p1.producedMsgChan, %p2.producedMsgChan : !esi.channel<i8>, !esi.channel<i8>, !esi.channel<i8>
hw.module @Rec(in %clk: !seq.clock) {
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> ()
  hw.instance "c1" @Consumer(clk: %clk: !seq.clock) -> (rawData: i8)
  hw.instance "p1" @Producer(clk: %clk: !seq.clock) -> ()
  hw.instance "p2" @Producer(clk: %clk: !seq.clock) -> ()
}

// CONN-LABEL: hw.module @Consumer(in %clk : !seq.clock, in %consumingFromChan : !esi.channel<i8>, out rawData : i8) {
// CONN:         %true = hw.constant true
// CONN:         %rawOutput, %valid = esi.unwrap.vr %consumingFromChan, %true : i8
// CONN:         hw.output %rawOutput : i8
hw.module @Consumer(in %clk: !seq.clock, out rawData: i8) {
  %dataInBundle = esi.service.req.to_client <@HostComms::@Recv> (["consumingFromChan"]) : !recvI8
  %rdy = hw.constant 1 : i1
  %dataIn = esi.bundle.unpack from %dataInBundle : !recvI8
  %rawData, %valid = esi.unwrap.vr %dataIn, %rdy: i8
  hw.output %rawData : i8
}

// CONN-LABEL: hw.module @Producer(in %clk : !seq.clock, out producedMsgChan : !esi.channel<i8>) {
// CONN:         %c0_i8 = hw.constant 0 : i8
// CONN:         %true = hw.constant true
// CONN:         %chanOutput, %ready = esi.wrap.vr %c0_i8, %true : i8
// CONN:         hw.output %chanOutput : !esi.channel<i8>
hw.module @Producer(in %clk: !seq.clock) {
  %data = hw.constant 0 : i8
  %valid = hw.constant 1 : i1
  %dataIn, %rdy = esi.wrap.vr %data, %valid : i8
  %dataInBundle = esi.bundle.pack %dataIn : !sendI8
  esi.service.req.to_server %dataInBundle -> <@HostComms::@Send> (["producedMsgChan"]) : !sendI8
}

// CONN-LABEL: hw.module @InOutLoopback(in %clk : !seq.clock, in %loopback_inout : !esi.channel<i16>, out loopback_inout : !esi.channel<i8>) {
// CONN:          %rawOutput, %valid = esi.unwrap.vr %loopback_inout, %ready : i16
// CONN:          %0 = comb.extract %rawOutput from 0 : (i16) -> i8
// CONN:          %chanOutput, %ready = esi.wrap.vr %0, %valid : i8
// CONN:          hw.output %chanOutput : !esi.channel<i8>
hw.module @InOutLoopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req.to_client <@HostComms::@ReqResp> (["loopback_inout"]) : !reqResp
  %dataIn = esi.bundle.unpack %dataTrunc from %dataInBundle : !reqResp
  %unwrap, %valid = esi.unwrap.vr %dataIn, %rdy: i16
  %trunc = comb.extract %unwrap from 0 : (i16) -> (i8)
  %dataTrunc, %rdy = esi.wrap.vr %trunc, %valid : i8
}


// CONN-LABEL:  esi.pure_module @LoopbackCosimPure {
// CONN-NEXT:     [[clk:%.+]] = esi.pure_module.input "clk" : !seq.clock
// CONN-NEXT:     [[rst:%.+]] = esi.pure_module.input "rst" : i1
// CONN-NEXT:     [[r2:%.+]] = esi.cosim [[clk]], [[rst]], %m1.loopback_inout, "m1.loopback_inout" : !esi.channel<i8> -> !esi.channel<i16>
// CONN-NEXT:     %m1.loopback_inout = hw.instance "m1" @InOutLoopback(clk: [[clk]]: !seq.clock, loopback_inout: [[r2]]: !esi.channel<i16>) -> (loopback_inout: !esi.channel<i8>)
esi.pure_module @LoopbackCosimPure {
  %clk = esi.pure_module.input "clk" : !seq.clock
  %rst = esi.pure_module.input "rst" : i1
  esi.service.instance svc @HostComms impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @InOutLoopback(clk: %clk: !seq.clock) -> ()
}

// CONN-LABEL: esi.mem.ram @MemA i64 x 20
// CONN-LABEL: hw.module @MemoryAccess1(in %clk : !seq.clock, in %rst : i1, in %write : !esi.channel<!hw.struct<address: i5, data: i64>>, in %readAddress : !esi.channel<i5>, out readData : !esi.channel<i64>, out writeDone : !esi.channel<i0>) {
// CONN:         %MemA = sv.reg  : !hw.inout<uarray<20xi64>>
// CONN:         %chanOutput, %ready = esi.wrap.vr %c0_i0, %write_done : i0
// CONN:         %rawOutput, %valid = esi.unwrap.vr %write, %ready : !hw.struct<address: i5, data: i64>
// CONN:         %address = hw.struct_extract %rawOutput["address"] : !hw.struct<address: i5, data: i64>
// CONN:         %data = hw.struct_extract %rawOutput["data"] : !hw.struct<address: i5, data: i64>
// CONN:         %[[ANDVR:.*]] = comb.and %valid, %ready {sv.namehint = "write_go"} : i1
// CONN:         %write_done = seq.compreg sym @write_done  %[[ANDVR]], %clk reset %rst, %false  : i1
// CONN:         %chanOutput_0, %ready_1 = esi.wrap.vr %[[MEMREAD:.*]], %valid_3 : i64
// CONN:         %rawOutput_2, %valid_3 = esi.unwrap.vr %readAddress, %ready_1 : i5
// CONN:         %[[MEMREADIO:.*]] = sv.array_index_inout %MemA[%rawOutput_2] : !hw.inout<uarray<20xi64>>, i5
// CONN:         %[[MEMREAD]] = sv.read_inout %[[MEMREADIO]] : !hw.inout<i64>
// CONN:         %[[CLOCK:.+]] = seq.from_clock %clk
// CONN:         sv.alwaysff(posedge %[[CLOCK]]) {
// CONN:           sv.if %[[ANDVR]] {
// CONN:             %[[ARRIDX:.*]] = sv.array_index_inout %MemA[%address] : !hw.inout<uarray<20xi64>>, i5
// CONN:             sv.passign %[[ARRIDX]], %data : i64
// CONN:           }
// CONN:         }(syncreset : posedge %rst) {
// CONN:         }
// CONN:         hw.output %chanOutput_0, %chanOutput : !esi.channel<i64>, !esi.channel<i0>

esi.mem.ram @MemA i64 x 20
!write = !hw.struct<address: i5, data: i64>
!writeBundle = !esi.bundle<[!esi.channel<!write> to "req", !esi.channel<i0> from "ack"]>
!readBundle = !esi.bundle<[!esi.channel<i5> to "address", !esi.channel<i64> from "data"]>

hw.module @MemoryAccess1(in %clk : !seq.clock, in %rst : i1, in %write : !esi.channel<!write>, in %readAddress : !esi.channel<i5>, out readData : !esi.channel<i64>, out writeDone : !esi.channel<i0>) {
  esi.service.instance svc @MemA impl as "sv_mem" (%clk, %rst) : (!seq.clock, i1) -> ()
  %writeBundle, %done = esi.bundle.pack %write : !writeBundle
  esi.service.req.to_server %writeBundle -> <@MemA::@write> ([]) : !writeBundle

  %readBundle, %readData = esi.bundle.pack %readAddress : !readBundle
  esi.service.req.to_server %readBundle -> <@MemA::@read> ([]) : !readBundle
  hw.output %readData, %done : !esi.channel<i64>, !esi.channel<i0>
}

// CONN-LABEL: hw.module @MemoryAccess2Read(in %clk : !seq.clock, in %rst : i1, in %write : !esi.channel<!hw.struct<address: i5, data: i64>>, in %readAddress : !esi.channel<i5>, in %readAddress2 : !esi.channel<i5>, out readData : !esi.channel<i64>, out readData2 : !esi.channel<i64>, out writeDone : !esi.channel<i0>) {
// CONN:         %MemA = sv.reg : !hw.inout<uarray<20xi64>>
// CONN:         hw.output %chanOutput_0, %chanOutput_4, %chanOutput : !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i0>

hw.module @MemoryAccess2Read(in %clk: !seq.clock, in %rst: i1, in %write: !esi.channel<!write>, in %readAddress: !esi.channel<i5>, in %readAddress2: !esi.channel<i5>, out readData: !esi.channel<i64>, out readData2: !esi.channel<i64>, out writeDone: !esi.channel<i0>) {
  esi.service.instance svc @MemA impl as "sv_mem" (%clk, %rst) : (!seq.clock, i1) -> ()

  %writeBundle, %done = esi.bundle.pack %write : !writeBundle
  esi.service.req.to_server %writeBundle -> <@MemA::@write> ([]) : !writeBundle

  %readBundle, %readData = esi.bundle.pack %readAddress : !readBundle
  esi.service.req.to_server %readBundle -> <@MemA::@read> ([]) : !readBundle

  %readBundle2, %readData2 = esi.bundle.pack %readAddress2 : !readBundle
  esi.service.req.to_server %readBundle2 -> <@MemA::@read> ([]) : !readBundle

  hw.output %readData, %readData2, %done : !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i0>
}

// Check that it doesn't crap out on external modules.
hw.module.extern @extern()
