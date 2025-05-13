// RUN: circt-opt --esi-connect-services --canonicalize %s | circt-opt | FileCheck %s --check-prefix=CONN
// RUN: circt-opt --esi-connect-services --lower-esi-bundles %s

!sendAny = !esi.bundle<[!esi.channel<!esi.any> from "send"]>
!sendI8 = !esi.bundle<[!esi.channel<i8> from "send"]>
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>
!reqResp = !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i8> from "resp"]>

esi.service.decl @HostComms {
  esi.service.port @Send : !sendAny
  esi.service.port @Recv : !recvI8
  esi.service.port @ReqResp : !reqResp
}


// CONN-LABEL: hw.module @Top(in %clk : !seq.clock, in %rst : i1) {
// CONN-DAG:     [[R1:%.+]] = esi.cosim.from_host %clk, %rst, "loopback_tohw.recv" : !esi.channel<i8>
// CONN-DAG:     %bundle = esi.bundle.pack [[R1]] : !esi.bundle<[!esi.channel<i8> to "recv"]>
// CONN-DAG:     %bundle_0, %send = esi.bundle.pack  : !esi.bundle<[!esi.channel<i8> from "send"]>
// CONN-DAG:     esi.cosim.to_host %clk, %rst, %send, "loopback_fromhw.send" : !esi.channel<i8>
// CONN:         hw.instance "m1" @Loopback(clk: %clk: !seq.clock, loopback_tohw: %bundle: !esi.bundle<[!esi.channel<i8> to "recv"]>, loopback_fromhw: %bundle_0: !esi.bundle<[!esi.channel<i8> from "send"]>) -> ()
hw.module @Top (in %clk: !seq.clock, in %rst: i1) {
  esi.service.instance #esi.appid<"cosim"> impl as  "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> ()
}


// CONN-LABEL:  hw.module @Loopback(in %clk : !seq.clock, in %loopback_tohw : !esi.bundle<[!esi.channel<i8> to "recv"]>, in %loopback_fromhw : !esi.bundle<[!esi.channel<i8> from "send"]>) {
// CONN-NEXT:     esi.manifest.req #esi.appid<"loopback_tohw">, <@HostComms::@Recv>, !esi.bundle<[!esi.channel<i8> to "recv"]>
// CONN-NEXT:     esi.manifest.req #esi.appid<"loopback_fromhw">, <@HostComms::@Send>, !esi.bundle<[!esi.channel<i8> from "send"]>
// CONN-NEXT:     %recv = esi.bundle.unpack  from %loopback_tohw : !esi.bundle<[!esi.channel<i8> to "recv"]>
// CONN-NEXT:     esi.bundle.unpack %recv from %loopback_fromhw : !esi.bundle<[!esi.channel<i8> from "send"]>
hw.module @Loopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) : !recvI8
  %dataOutBundle = esi.service.req <@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !sendI8
  %dataIn = esi.bundle.unpack from %dataInBundle : !recvI8
  esi.bundle.unpack %dataIn from %dataOutBundle : !sendI8
}

// CONN-LABEL: hw.module @Top2(in %clk : !seq.clock, out chksum : i8) {
// CONN:         [[r0:%.+]]:6 = esi.service.impl_req #esi.appid<"topComms"> svc @HostComms impl as "topComms2"(%clk) : (!seq.clock) -> (i8, !esi.bundle<[!esi.channel<i8> to "recv"]>, !esi.bundle<[!esi.channel<i8> from "send"]>, !esi.bundle<[!esi.channel<i8> to "recv"]>, !esi.bundle<[!esi.channel<i8> from "send"]>, !esi.bundle<[!esi.channel<i8> from "send"]>) {
// CONN-DAG:       %1 = esi.service.impl_req.req <@HostComms::@Recv>([#esi.appid<"loopback_tohw">]) : !esi.bundle<[!esi.channel<i8> to "recv"]>
// CONN-DAG:       %2 = esi.service.impl_req.req <@HostComms::@Send>([#esi.appid<"loopback_fromhw">]) : !esi.bundle<[!esi.channel<i8> from "send"]>
// CONN-DAG:       %3 = esi.service.impl_req.req <@HostComms::@Recv>([#esi.appid<"consumingFromChan">]) : !esi.bundle<[!esi.channel<i8> to "recv"]>
// CONN-DAG:       %4 = esi.service.impl_req.req <@HostComms::@Send>([#esi.appid<"producer"[0]>, #esi.appid<"producedMsgChan">]) : !esi.bundle<[!esi.channel<i8> from "send"]>
// CONN-DAG:       %5 = esi.service.impl_req.req <@HostComms::@Send>([#esi.appid<"producer"[1]>, #esi.appid<"producedMsgChan">]) : !esi.bundle<[!esi.channel<i8> from "send"]>
// CONN:         hw.instance "r1" @Rec(clk: %clk: !seq.clock, loopback_tohw: %0#1: !esi.bundle<[!esi.channel<i8> to "recv"]>, loopback_fromhw: %0#2: !esi.bundle<[!esi.channel<i8> from "send"]>, consumingFromChan: %0#3: !esi.bundle<[!esi.channel<i8> to "recv"]>, producer_0.producedMsgChan: %0#4: !esi.bundle<[!esi.channel<i8> from "send"]>, producer_1.producedMsgChan: %0#5: !esi.bundle<[!esi.channel<i8> from "send"]>) -> ()
// CONN:         hw.output [[r0]]#0 : i8
hw.module @Top2 (in %clk: !seq.clock, out chksum: i8) {
  %c = esi.service.instance #esi.appid<"topComms"> svc @HostComms impl as  "topComms2" (%clk) : (!seq.clock) -> (i8)
  hw.instance "r1" @Rec(clk: %clk: !seq.clock) -> ()
  hw.output %c : i8
}

// CONN-LABEL:  hw.module @Rec(in %clk : !seq.clock, in %loopback_tohw : !esi.bundle<[!esi.channel<i8> to "recv"]>, in %loopback_fromhw : !esi.bundle<[!esi.channel<i8> from "send"]>, in %consumingFromChan : !esi.bundle<[!esi.channel<i8> to "recv"]>, in %producer_0.producedMsgChan : !esi.bundle<[!esi.channel<i8> from "send"]>, in %producer_1.producedMsgChan : !esi.bundle<[!esi.channel<i8> from "send"]>) {
// CONN:          hw.instance "m1" @Loopback(clk: %clk: !seq.clock, loopback_tohw: %loopback_tohw: !esi.bundle<[!esi.channel<i8> to "recv"]>, loopback_fromhw: %loopback_fromhw: !esi.bundle<[!esi.channel<i8> from "send"]>) -> ()
// CONN:          %c1.rawData = hw.instance "c1" @Consumer(clk: %clk: !seq.clock, consumingFromChan: %consumingFromChan: !esi.bundle<[!esi.channel<i8> to "recv"]>) -> (rawData: i8)
// CONN:          hw.instance "p1" @Producer(clk: %clk: !seq.clock, producedMsgChan: %producer_0.producedMsgChan: !esi.bundle<[!esi.channel<i8> from "send"]>) -> ()
// CONN:          hw.instance "p2" @Producer(clk: %clk: !seq.clock, producedMsgChan: %producer_1.producedMsgChan: !esi.bundle<[!esi.channel<i8> from "send"]>) -> ()
hw.module @Rec(in %clk: !seq.clock) {
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> ()
  hw.instance "c1" @Consumer(clk: %clk: !seq.clock) -> (rawData: i8)
  hw.instance "p1" @Producer(clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"producer"[0]>}
  hw.instance "p2" @Producer(clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"producer"[1]>}
}
// CONN-LABEL:   hw.module @Consumer(in %clk : !seq.clock, in %consumingFromChan : !esi.bundle<[!esi.channel<i8> to "recv"]>, out rawData : i8) {
// CONN:           %true = hw.constant true
// CONN:           %recv = esi.bundle.unpack  from %consumingFromChan : !esi.bundle<[!esi.channel<i8> to "recv"]>
// CONN:           %rawOutput, %valid = esi.unwrap.vr %recv, %true : i8
// CONN:           hw.output %rawOutput : i8
hw.module @Consumer(in %clk: !seq.clock, out rawData: i8) {
  %dataInBundle = esi.service.req <@HostComms::@Recv> (#esi.appid<"consumingFromChan">) : !recvI8
  %rdy = hw.constant 1 : i1
  %dataIn = esi.bundle.unpack from %dataInBundle : !recvI8
  %rawData, %valid = esi.unwrap.vr %dataIn, %rdy: i8
  hw.output %rawData : i8
}

// CONN-LABEL:   hw.module @Producer(in %clk : !seq.clock, in %producedMsgChan : !esi.bundle<[!esi.channel<i8> from "send"]>) {
// CONN:           %c0_i8 = hw.constant 0 : i8
// CONN:           %true = hw.constant true
// CONN:           %chanOutput, %ready = esi.wrap.vr %c0_i8, %true : i8
// CONN:           esi.bundle.unpack %chanOutput from %producedMsgChan : !esi.bundle<[!esi.channel<i8> from "send"]>
hw.module @Producer(in %clk: !seq.clock) {
  %data = hw.constant 0 : i8
  %valid = hw.constant 1 : i1
  %dataIn, %rdy = esi.wrap.vr %data, %valid : i8
  %dataInBundle = esi.service.req <@HostComms::@Send> (#esi.appid<"producedMsgChan">) : !sendI8
  esi.bundle.unpack %dataIn from %dataInBundle : !sendI8
}

// CONN-LABEL:   hw.module @InOutLoopback(in %clk : !seq.clock, in %loopback_inout : !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i8> from "resp"]>) {
// CONN:           %req = esi.bundle.unpack %chanOutput from %loopback_inout : !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i8> from "resp"]>
// CONN:           %rawOutput, %valid = esi.unwrap.vr %req, %ready : i16
// CONN:           %0 = comb.extract %rawOutput from 0 : (i16) -> i8
// CONN:           %chanOutput, %ready = esi.wrap.vr %0, %valid : i8
hw.module @InOutLoopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req <@HostComms::@ReqResp> (#esi.appid<"loopback_inout">) : !reqResp
  %dataIn = esi.bundle.unpack %dataTrunc from %dataInBundle : !reqResp
  %unwrap, %valid = esi.unwrap.vr %dataIn, %rdy: i16
  %trunc = comb.extract %unwrap from 0 : (i16) -> (i8)
  %dataTrunc, %rdy = esi.wrap.vr %trunc, %valid : i8
}


// CONN-LABEL:  esi.pure_module @LoopbackCosimPure {
// CONN-NEXT:     [[clk:%.+]] = esi.pure_module.input "clk" : !seq.clock
// CONN-NEXT:     [[rst:%.+]] = esi.pure_module.input "rst" : i1
// CONN-NEXT:     esi.manifest.service_impl #esi.appid<"cosim"> svc @HostComms by "cosim" engine with {} {
// CONN-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inout">] req <@HostComms::@ReqResp>(!esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i8> from "resp"]>) channels {req = {name = "loopback_inout.req", type = "cosim"}, resp = {name = "loopback_inout.resp", type = "cosim"}}
// CONN-NEXT:     }
// CONN-NEXT:     [[r2:%.+]] = esi.cosim.from_host [[clk]], [[rst]], "loopback_inout.req" : !esi.channel<i16>
// CONN-NEXT:     %bundle, %resp = esi.bundle.pack [[r2]] : !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i8> from "resp"]>
// CONN-NEXT:     esi.cosim.to_host [[clk]], [[rst]], %resp, "loopback_inout.resp" : !esi.channel<i8>
// CONN-NEXT:     hw.instance "m1" @InOutLoopback(clk: %0: !seq.clock, loopback_inout: %bundle: !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i8> from "resp"]>) -> ()
esi.pure_module @LoopbackCosimPure {
  %clk = esi.pure_module.input "clk" : !seq.clock
  %rst = esi.pure_module.input "rst" : i1
  esi.service.instance #esi.appid<"cosim"> svc @HostComms impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
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
!writeBundle = !esi.bundle<[!esi.channel<!write> from "req", !esi.channel<i0> to "ack"]>
!readBundle = !esi.bundle<[!esi.channel<i5> from "address", !esi.channel<i64> to "data"]>

hw.module @MemoryAccess1(in %clk : !seq.clock, in %rst : i1, in %write : !esi.channel<!write>, in %readAddress : !esi.channel<i5>, out readData : !esi.channel<i64>, out writeDone : !esi.channel<i0>) {
  esi.service.instance #esi.appid<"mem"> svc @MemA impl as "sv_mem" (%clk, %rst) : (!seq.clock, i1) -> ()
  %writeBundle = esi.service.req <@MemA::@write> (#esi.appid<"write">) : !writeBundle
  %done = esi.bundle.unpack %write from %writeBundle : !writeBundle

  %readBundle = esi.service.req <@MemA::@read> (#esi.appid<"read">) : !readBundle
  %readData = esi.bundle.unpack %readAddress from %readBundle : !readBundle
  hw.output %readData, %done : !esi.channel<i64>, !esi.channel<i0>
}

// CONN-LABEL: hw.module @MemoryAccess2Read(in %clk : !seq.clock, in %rst : i1, in %write : !esi.channel<!hw.struct<address: i5, data: i64>>, in %readAddress : !esi.channel<i5>, in %readAddress2 : !esi.channel<i5>, out readData : !esi.channel<i64>, out readData2 : !esi.channel<i64>, out writeDone : !esi.channel<i0>) {
// CONN:         %MemA = sv.reg : !hw.inout<uarray<20xi64>>
// CONN:         hw.output %chanOutput_0, %chanOutput_4, %chanOutput : !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i0>

hw.module @MemoryAccess2Read(in %clk: !seq.clock, in %rst: i1, in %write: !esi.channel<!write>, in %readAddress: !esi.channel<i5>, in %readAddress2: !esi.channel<i5>, out readData: !esi.channel<i64>, out readData2: !esi.channel<i64>, out writeDone: !esi.channel<i0>) {
  esi.service.instance #esi.appid<"mem"> svc @MemA impl as "sv_mem" (%clk, %rst) : (!seq.clock, i1) -> ()

  %writeBundle = esi.service.req <@MemA::@write> (#esi.appid<"write">) : !writeBundle
  %done = esi.bundle.unpack %write from %writeBundle : !writeBundle

  %readBundle = esi.service.req <@MemA::@read> (#esi.appid<"read"[0]>) : !readBundle
  %readData = esi.bundle.unpack %readAddress from %readBundle : !readBundle

  %readBundle2 = esi.service.req <@MemA::@read> (#esi.appid<"read"[1]>) : !readBundle
  %readData2 = esi.bundle.unpack %readAddress2 from %readBundle2 : !readBundle

  hw.output %readData, %readData2, %done : !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i0>
}

// Check that it doesn't crap out on external modules.
hw.module.extern @extern()

// CONN-LABEL:  esi.service.std.func @funcs
esi.service.std.func @funcs

// CONN-LABEL:   hw.module @CallableFunc1(in %func1 : !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>) {
// CONN-NEXT:      esi.manifest.req #esi.appid<"func1">, <@funcs::@call> std "esi.service.std.func", !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
// CONN-NEXT:      %arg = esi.bundle.unpack %arg from %func1 : !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
!func1Signature = !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
hw.module @CallableFunc1() {
  %call = esi.service.req <@funcs::@call> (#esi.appid<"func1">) : !func1Signature
  %arg = esi.bundle.unpack %arg from %call : !func1Signature
}

// CONN-LABEL:   hw.module @CallableAccel1(in %clk : !seq.clock, in %rst : i1) {
// CONN-NEXT:      hw.instance "func1" @CallableFunc1(func1: %bundle: !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>) -> ()
// CONN-NEXT:      esi.manifest.service_impl #esi.appid<"funcComms"> svc @funcs std "esi.service.std.func" by "cosim" engine with {} {
// CONN-NEXT:        esi.manifest.impl_conn [#esi.appid<"func1">] req <@funcs::@call>(!esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>) channels {arg = {name = "func1.arg", type = "cosim"}, result = {name = "func1.result", type = "cosim"}}
// CONN-NEXT:      }
// CONN-NEXT:      %0 = esi.cosim.from_host %clk, %rst, "func1.arg" : !esi.channel<i16>
// CONN-NEXT:      %bundle, %result = esi.bundle.pack %0 : !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
// CONN-NEXT:      esi.cosim.to_host %clk, %rst, %result, "func1.result" : !esi.channel<i16>
hw.module @CallableAccel1(in %clk: !seq.clock, in %rst: i1) {
  hw.instance "func1" @CallableFunc1() -> ()
  esi.service.instance #esi.appid<"funcComms"> svc @funcs impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
}

esi.service.std.mmio @mmio
!mmioReq = !esi.bundle<[!esi.channel<ui32> to "offset", !esi.channel<i64> from "data"]>
!mmioRWReq = !esi.bundle<[!esi.channel<!hw.struct<write: i1, offset: ui32, data: i64>> to "cmd", !esi.channel<i64> from "data"]>

// CONN-LABEL:  hw.module @MMIOManifest(in %clk : !seq.clock, in %rst : i1, in %manifest : !esi.bundle<[!esi.channel<ui32> to "offset", !esi.channel<i64> from "data"]>, in %manifestRW : !esi.bundle<[!esi.channel<!hw.struct<write: i1, offset: ui32, data: i64>> to "cmd", !esi.channel<i64> from "data"]>) {
// CONN-NEXT:     %true = hw.constant true
// CONN-NEXT:     %c0_i64 = hw.constant 0 : i64
// CONN-NEXT:     esi.manifest.req #esi.appid<"manifest">, <@mmio::@read> std "esi.service.std.mmio", !esi.bundle<[!esi.channel<ui32> to "offset", !esi.channel<i64> from "data"]>
// CONN-NEXT:     %chanOutput, %ready = esi.wrap.vr %c0_i64, %true : i64
// CONN-NEXT:     %offset = esi.bundle.unpack %chanOutput from %manifest : !esi.bundle<[!esi.channel<ui32> to "offset", !esi.channel<i64> from "data"]>
// CONN-NEXT:     esi.manifest.req #esi.appid<"manifestRW">, <@mmio::@read_write> std "esi.service.std.mmio", !esi.bundle<[!esi.channel<!hw.struct<write: i1, offset: ui32, data: i64>> to "cmd", !esi.channel<i64> from "data"]>
hw.module @MMIOManifest(in %clk: !seq.clock, in %rst: i1) {
  %req = esi.service.req <@mmio::@read> (#esi.appid<"manifest">) : !mmioReq
  %data = hw.constant 0 : i64
  %valid = hw.constant 1 : i1
  %data_ch, %ready = esi.wrap.vr %data, %valid : i64
  %addr = esi.bundle.unpack %data_ch from %req : !mmioReq

  %reqRW = esi.service.req <@mmio::@read_write> (#esi.appid<"manifestRW">) : !mmioRWReq
  %dataChannel, %dataChannelReady = esi.wrap.vr %data, %valid: i64
  %cmdChannel = esi.bundle.unpack %dataChannel from %reqRW : !mmioRWReq
}

// CONN-LABEL:  esi.service.std.hostmem @hostmem
// CONN-LABEL:  hw.module @HostmemRW(in %clk : !seq.clock, in %rst : i1, in %write : !esi.channel<!hw.struct<address: ui64, tag: ui8, data: i128>>, in %readAddress : !esi.channel<!hw.struct<address: ui64, tag: ui8>>, in %hostmemWrite : !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8, data: i128>> from "req", !esi.channel<ui8> to "ackTag"]>, in %hostmemRead : !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8>> from "req", !esi.channel<!hw.struct<tag: ui8, data: i64>> to "resp"]>, out readData : !esi.channel<!hw.struct<tag: ui8, data: i64>>, out writeDone : !esi.channel<ui8>) {
// CONN-NEXT:     esi.manifest.req #esi.appid<"hostmemWrite">, <@hostmem::@write> std "esi.service.std.hostmem", !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8, data: i128>> from "req", !esi.channel<ui8> to "ackTag"]>
// CONN-NEXT:     %ackTag = esi.bundle.unpack %write from %hostmemWrite : !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8, data: i128>> from "req", !esi.channel<ui8> to "ackTag"]>
// CONN-NEXT:     esi.manifest.req #esi.appid<"hostmemRead">, <@hostmem::@read> std "esi.service.std.hostmem", !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8>> from "req", !esi.channel<!hw.struct<tag: ui8, data: i64>> to "resp"]>
// CONN-NEXT:     %resp = esi.bundle.unpack %readAddress from %hostmemRead : !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8>> from "req", !esi.channel<!hw.struct<tag: ui8, data: i64>> to "resp"]>
// CONN-NEXT:     hw.output %resp, %ackTag : !esi.channel<!hw.struct<tag: ui8, data: i64>>, !esi.channel<ui8>

esi.service.std.hostmem @hostmem
!hostmemReadReq = !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8>> from "req", !esi.channel<!hw.struct<tag: ui8, data: i64>> to "resp"]>
!hostmemWriteReq = !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8, data: i128>> from "req", !esi.channel<ui8> to "ackTag"]>

hw.module @HostmemRW(in %clk : !seq.clock, in %rst : i1, in %write : !esi.channel<!hw.struct<address: ui64, tag: ui8, data: i128>>, in %readAddress : !esi.channel<!hw.struct<address: ui64, tag: ui8>>, out readData : !esi.channel<!hw.struct<tag: ui8, data: i64>>, out writeDone : !esi.channel<ui8>) {
  %writeBundle = esi.service.req <@hostmem::@write> (#esi.appid<"hostmemWrite">) : !hostmemWriteReq
  %ackTag = esi.bundle.unpack %write from %writeBundle : !hostmemWriteReq

  %readBundle = esi.service.req <@hostmem::@read> (#esi.appid<"hostmemRead">) : !hostmemReadReq
  %readData = esi.bundle.unpack %readAddress from %readBundle : !hostmemReadReq

  hw.output %readData, %ackTag: !esi.channel<!hw.struct<tag: ui8, data: i64>>, !esi.channel<ui8>
}

esi.service.std.telemetry @telemetry
hw.module @TelemetryTest1(in %clk : !seq.clock, in %rst : i1, in %value: !esi.channel<ui64>) {
  %telemetryBundle = esi.service.req <@telemetry::@report> (#esi.appid<"telemetry">) : !esi.bundle<[!esi.channel<i1> to "get", !esi.channel<ui64> from "data"]>
  esi.bundle.unpack %value from %telemetryBundle : !esi.bundle<[!esi.channel<i1> to "get", !esi.channel<ui64> from "data"]>
}
