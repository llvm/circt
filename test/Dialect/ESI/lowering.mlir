// RUN: circt-opt %s --lower-esi-to-physical -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-esi-ports -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=IFACE %s
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --hw-flatten-io --lower-esi-to-hw | FileCheck --check-prefix=HW %s

hw.module.extern @Sender(in %clk: !seq.clock, out x: !esi.channel<i4>, out y: i8) attributes {esi.bundle}
hw.module.extern @ArrSender(out x: !esi.channel<!hw.array<4xi64>>) attributes {esi.bundle}
hw.module.extern @Reciever(in %a: !esi.channel<i4>, in %clk: !seq.clock) attributes {esi.bundle}
hw.module.extern @i0SenderReceiver(in %in: !esi.channel<i0>, out out: !esi.channel<i0>)

// CHECK-LABEL: hw.module.extern @Sender(in %clk : !seq.clock, out x : !esi.channel<i4>, out y : i8)
// CHECK-LABEL: hw.module.extern @Reciever(in %a : !esi.channel<i4>, in %clk : !seq.clock)

// IFACE-LABEL: sv.interface @IValidReady_i4 {
// IFACE-NEXT:    sv.interface.signal @valid : i1
// IFACE-NEXT:    sv.interface.signal @ready : i1
// IFACE-NEXT:    sv.interface.signal @data : i4
// IFACE-NEXT:    sv.interface.modport @sink (input @ready, output @valid, output @data)
// IFACE-NEXT:    sv.interface.modport @source (input @valid, input @data, output @ready)
// IFACE-LABEL: sv.interface @IValidReady_ArrayOf4xi64 {
// IFACE-NEXT:    sv.interface.signal @valid : i1
// IFACE-NEXT:    sv.interface.signal @ready : i1
// IFACE-NEXT:    sv.interface.signal @data : !hw.array<4xi64>
// IFACE-NEXT:    sv.interface.modport @sink  (input @ready, output @valid, output @data)
// IFACE-NEXT:    sv.interface.modport @source  (input @valid, input @data, output @ready)
// IFACE-LABEL: hw.module.extern @Sender(in %clk : !seq.clock, in %x : !sv.modport<@IValidReady_i4::@sink>, out y : i8)
// IFACE-LABEL: hw.module.extern @ArrSender(in %x : !sv.modport<@IValidReady_ArrayOf4xi64::@sink>)
// IFACE-LABEL: hw.module.extern @Reciever(in %a : !sv.modport<@IValidReady_i4::@source>, in %clk : !seq.clock)
// IFACE-LABEL: hw.module.extern @i0SenderReceiver(in %in : i0, in %in_valid : i1, in %out_ready : i1, out in_ready : i1, out out : i0, out out_valid : i1)

hw.module @test(in %clk: !seq.clock, in %rst:i1) {

  %esiChan2, %0 = hw.instance "sender2" @Sender(clk: %clk: !seq.clock) -> (x: !esi.channel<i4>, y: i8)
  %bufferedChan2 = esi.buffer %clk, %rst, %esiChan2 { stages = 4 } : !esi.channel<i4> -> !esi.channel<i4>
  hw.instance "recv2" @Reciever (a: %bufferedChan2: !esi.channel<i4>, clk: %clk: !seq.clock) -> ()

  // CHECK:      %sender2.x, %sender2.y = hw.instance "sender2" @Sender(clk: %clk: !seq.clock) -> (x: !esi.channel<i4>, y: i8)
  // CHECK-NEXT:  %0 = esi.stage %clk, %rst, %sender2.x : !esi.channel<i4>
  // CHECK-NEXT:  %1 = esi.stage %clk, %rst, %0 : !esi.channel<i4>
  // CHECK-NEXT:  %2 = esi.stage %clk, %rst, %1 : !esi.channel<i4>
  // CHECK-NEXT:  %3 = esi.stage %clk, %rst, %2 : !esi.channel<i4>
  // CHECK-NEXT:  hw.instance "recv2" @Reciever(a: %3: !esi.channel<i4>, clk: %clk: !seq.clock) -> ()

  // IFACE-LABEL: hw.module @test(in %clk : !seq.clock, in %rst : i1) {
  // IFACE-NEXT:    %i4FromSender2 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  // IFACE-NEXT:    %[[#modport1:]] = sv.modport.get %i4FromSender2 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // IFACE-NEXT:    %[[#channel:]] = esi.wrap.iface %[[#modport1:]] : !sv.modport<@IValidReady_i4::@source> -> !esi.channel<i4>
  // IFACE-NEXT:    %[[#modport2:]] = sv.modport.get %i4FromSender2 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // IFACE-NEXT:    %sender2.y = hw.instance "sender2" @Sender(clk: %clk: !seq.clock, x: %2: !sv.modport<@IValidReady_i4::@sink>) -> (y: i8)
  // IFACE-NEXT:    %[[#buffer:]] = esi.buffer %clk, %rst, %[[#channel:]] {stages = 4 : i64} : !esi.channel<i4> -> !esi.channel<i4>
  // IFACE-NEXT:    %i4ToRecv2 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  // IFACE-NEXT:    %[[#modport3:]] = sv.modport.get %i4ToRecv2 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // IFACE-NEXT:    esi.unwrap.iface %[[#buffer:]] into %[[#modport3:]] : (!esi.channel<i4>, !sv.modport<@IValidReady_i4::@sink>)
  // IFACE-NEXT:    %[[#modport4:]] = sv.modport.get %i4ToRecv2 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // IFACE-NEXT:    hw.instance "recv2" @Reciever(a: %[[#modport4:]]: !sv.modport<@IValidReady_i4::@source>, clk: %clk: !seq.clock) -> ()

  // HW-LABEL:    hw.module @test(in %clk : !seq.clock, in %rst : i1)
  // After all 3 ESI lowering passes, there shouldn't be any ESI constructs!
  // HW-NOT: esi
}

// https://github.com/llvm/circt/issues/8295
hw.module @bug8295(out x: !esi.channel<i4>, out sv: i1, out sr: i1, out sd: i4) {
  %sv, %sr, %sd = esi.snoop.vr %x : !esi.channel<i4>
  %data = hw.constant 0 : i4
  %valid = hw.constant 1 : i1
  %x, %ready = esi.wrap.vr %data, %valid : i4
  hw.output %x, %sv, %sr, %sd : !esi.channel<i4>, i1, i1, i4
}


hw.module @add11(in %clk: !seq.clock, in %ints: !esi.channel<i32>, out mutatedInts: !esi.channel<i32>, out c4: i4) {
  %i, %i_valid = esi.unwrap.vr %ints, %rdy : i32
  %c11 = hw.constant 11 : i32
  %m = comb.add %c11, %i : i32
  %mutInts, %rdy = esi.wrap.vr %m, %i_valid : i32
  %c4 = hw.constant 0 : i4
  hw.output %mutInts, %c4 : !esi.channel<i32>, i4
}
// HW-LABEL: hw.module @add11(in %clk : !seq.clock, in %ints : i32, in %ints_valid : i1, in %mutatedInts_ready : i1, out ints_ready : i1, out mutatedInts : i32, out mutatedInts_valid : i1, out c4 : i4) {
// HW:   %{{.+}} = hw.constant 11 : i32
// HW:   [[RES0:%.+]] = comb.add %ints, %{{.+}} : i32
// HW:   %{{.+}} = hw.constant 0 : i4
// HW:   hw.output %mutatedInts_ready, [[RES0]], %ints_valid, %{{.+}} : i1, i32, i1, i4

hw.module @InternRcvr(in %in: !esi.channel<!hw.array<4xi8>>) {}

hw.module @test2(in %clk: !seq.clock, in %rst:i1) {
  %ints, %c4 = hw.instance "adder" @add11(clk: %clk: !seq.clock, ints: %ints: !esi.channel<i32>) -> (mutatedInts: !esi.channel<i32>, c4: i4)

  %valid, %ready, %data = esi.snoop.vr %ints: !esi.channel<i32>
  %xact = comb.and %valid, %ready : i1

  %tx_xact, %tx_data = esi.snoop.xact %ints: !esi.channel<i32>

  %nullBit = esi.null : !esi.channel<i4>
  hw.instance "nullRcvr" @Reciever(a: %nullBit: !esi.channel<i4>, clk: %clk: !seq.clock) -> ()

  %nullArray = esi.null : !esi.channel<!hw.array<4xi8>>
  hw.instance "nullInternRcvr" @InternRcvr(in: %nullArray: !esi.channel<!hw.array<4xi8>>) -> ()
}
// HW-LABEL: hw.module @test2(in %clk : !seq.clock, in %rst : i1) {
// HW-NEXT:    %adder.ints_ready, %adder.mutatedInts, %adder.mutatedInts_valid, %adder.c4 = hw.instance "adder" @add11(clk: %clk: !seq.clock, ints: %adder.mutatedInts: i32, ints_valid: %adder.mutatedInts_valid: i1, mutatedInts_ready: %adder.ints_ready: i1) -> (ints_ready: i1, mutatedInts: i32, mutatedInts_valid: i1, c4: i4)
// HW-NEXT:    [[XACT:%.+]] = comb.and %adder.mutatedInts_valid, %adder.ints_ready : i1
// HW-NEXT:    [[TX_XACT:%.+]] = comb.and %adder.mutatedInts_valid, %adder.ints_ready : i1
// HW:         [[ZERO:%.+]] = hw.bitcast %c0_i4 : (i4) -> i4
// HW:         sv.interface.signal.assign %i4ToNullRcvr(@IValidReady_i4::@data) = [[ZERO]] : i4
// HW:         [[ZM:%.+]] = sv.modport.get %{{.+}} @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
// HW:         hw.instance "nullRcvr" @Reciever(a: [[ZM]]: !sv.modport<@IValidReady_i4::@source>, clk: %clk: !seq.clock) -> ()
// HW:         %c0_i32 = hw.constant 0 : i32
// HW:         [[ZA:%.+]] = hw.bitcast %c0_i32 : (i32) -> !hw.array<4xi8>
// HW:         %nullInternRcvr.in_ready = hw.instance "nullInternRcvr" @InternRcvr(in: [[ZA]]: !hw.array<4xi8>, in_valid: %false_0: i1) -> (in_ready: i1)

hw.module @twoChannelArgs(in %clk: !seq.clock, in %ints: !esi.channel<i32>, in %foo: !esi.channel<i7>) {
  %rdy = hw.constant 1 : i1
  %i, %i_valid = esi.unwrap.vr %ints, %rdy : i32
  %i2, %i2_valid = esi.unwrap.vr %foo, %rdy : i7
}
// HW-LABEL: hw.module @twoChannelArgs(in %clk : !seq.clock, in %ints : i32, in %ints_valid : i1, in %foo : i7, in %foo_valid : i1, out ints_ready : i1, out foo_ready : i1)
// HW:   %true = hw.constant true
// HW:   hw.output %true, %true : i1, i1

// IFACE: %i1ToHandshake_fork0FromArg0 = sv.interface.instance : !sv.interface<@IValidReady_i1>
hw.module.extern @handshake_fork_1ins_2outs_ctrl(in %in0: !esi.channel<i1>, in %clock: i1, in %reset: i1, out out0: !esi.channel<i1>, out out1: !esi.channel<i1>) attributes {esi.bundle}
hw.module @test_constant(in %arg0: !esi.channel<i1>, in %clock: i1, in %reset: i1, out outCtrl: !esi.channel<i1>) {
  %handshake_fork0.out0, %handshake_fork0.out1 = hw.instance "handshake_fork0" @handshake_fork_1ins_2outs_ctrl(in0: %arg0: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>)
  hw.output %handshake_fork0.out1 : !esi.channel<i1>
}

// HW-LABEL: hw.module @i0Typed(in %a : i0, in %a_valid : i1, in %clk : !seq.clock, in %rst : i1, in %x_ready : i1, out a_ready : i1, out x : i0, out x_valid : i1) {
// HW:         %pipelineStage.a_ready, %pipelineStage.x, %pipelineStage.x_valid = hw.instance "pipelineStage" @ESI_PipelineStage1<WIDTH: ui32 = 0>(clk: %clk: !seq.clock, rst: %rst: i1, a: %a: i0, a_valid: %a_valid: i1, x_ready: %x_ready: i1) -> (a_ready: i1, x: i0, x_valid: i1)
// HW:         hw.output %pipelineStage.a_ready, %pipelineStage.x, %pipelineStage.x_valid : i1, i0, i1
// HW:       }
hw.module @i0Typed(in %a: !esi.channel<i0>, in %clk: !seq.clock, in %rst: i1, out x: !esi.channel<i0>) {
  %0 = esi.buffer %clk, %rst, %a  : !esi.channel<i0> -> !esi.channel<i0>
  %i0Value, %valid = esi.unwrap.vr %0, %ready : i0
  %chanOutput, %ready = esi.wrap.vr %i0Value, %valid : i0
  hw.output %chanOutput : !esi.channel<i0>
}

// IFACE: hw.module @HandshakeToESIWrapper(in %clock : i1, in %reset : i1, in %in_ctrl : i0, in %in_ctrl_valid : i1, in %in0_ld_data0 : i32, in %in0_ld_data0_valid : i1, in %in1_ld_data0 : i32, in %in1_ld_data0_valid : i1, in %out_ctrl_ready : i1, in %in0_ld_addr0_ready : i1, in %in1_ld_addr0_ready : i1, in %out0_ready : i1, out in_ctrl_ready : i1, out in0_ld_data0_ready : i1, out in1_ld_data0_ready : i1, out out_ctrl : i0, out out_ctrl_valid : i1, out in0_ld_addr0 : i64, out in0_ld_addr0_valid : i1, out in1_ld_addr0 : i64, out in1_ld_addr0_valid : i1, out out0 : i32, out out0_valid : i1)
hw.module @HandshakeToESIWrapper(in %clock : i1, in %reset : i1, in %in_ctrl : !esi.channel<i0>, in %in0_ld_data0: !esi.channel<i32>, in %in1_ld_data0: !esi.channel<i32>, out out_ctrl: !esi.channel<i0>, out in0_ld_addr0: !esi.channel<i64>, out in1_ld_addr0: !esi.channel<i64>, out out0: !esi.channel<i32>) {
  %i0 = hw.constant 0 : i0
  %c1 = hw.constant 1 : i1
  %c32 = hw.constant 1 : i32
  %c64 = hw.constant 1 : i64
  %chanOutput, %ready = esi.wrap.vr %i0, %c1 : i0
  %chanOutput_2, %ready_3 = esi.wrap.vr %c64, %c1 : i64
  %chanOutput_6, %ready_7 = esi.wrap.vr %c64, %c1 : i64
  %chanOutput_8, %ready_9 = esi.wrap.vr %c32, %c1 : i32
  hw.output %chanOutput, %chanOutput_2, %chanOutput_6, %chanOutput_8 : !esi.channel<i0>, !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i32>
}

// IFACE: hw.module @ServiceWrapper(in %clock : i1, in %reset : i1, in %ctrl : i0, in %ctrl_valid : i1, in %port0 : i32, in %port0_valid : i1, in %port1 : i32, in %port1_valid : i1, in %ctrl_ready : i1, in %port0_ready : i1, in %port1_ready : i1, out ctrl_ready : i1, out port0_ready : i1, out port1_ready : i1, out ctrl : i0, out ctrl_valid : i1, out port0 : i64, out port0_valid : i1, out port1 : i64, out port1_valid : i1)
hw.module @ServiceWrapper(in %clock: i1, in %reset: i1, in %ctrl: !esi.channel<i0>, in %port0: !esi.channel<i32>, in %port1: !esi.channel<i32>, out ctrl: !esi.channel<i0>, out port0: !esi.channel<i64>, out port1: !esi.channel<i64>) {
  %HandshakeToESIWrapper.out_ctrl, %HandshakeToESIWrapper.in0_ld_addr0, %HandshakeToESIWrapper.in1_ld_addr0, %HandshakeToESIWrapper.out0 = hw.instance "HandshakeToESIWrapper" @HandshakeToESIWrapper(clock: %clock: i1, reset: %reset: i1, in_ctrl: %ctrl: !esi.channel<i0>, in0_ld_data0: %port0: !esi.channel<i32>, in1_ld_data0: %port1: !esi.channel<i32>) -> (out_ctrl: !esi.channel<i0>, in0_ld_addr0: !esi.channel<i64>, in1_ld_addr0: !esi.channel<i64>, out0: !esi.channel<i32>)
  hw.output %HandshakeToESIWrapper.out_ctrl, %HandshakeToESIWrapper.in0_ld_addr0, %HandshakeToESIWrapper.in1_ld_addr0 : !esi.channel<i0>, !esi.channel<i64>, !esi.channel<i64>
}

// IFACE-LABEL:  hw.module @i1FifoLoopback(in %in : i3, in %in_empty : i1, in %out_rden : i1, out in_rden : i1, out out : i3, out out_empty : i1)
// IFACE-NEXT:     %chanOutput, %rden = esi.wrap.fifo %in, %in_empty : !esi.channel<i3, FIFO>
// IFACE-NEXT:     %data, %empty = esi.unwrap.fifo %chanOutput, %out_rden : !esi.channel<i3, FIFO>
// IFACE-NEXT:     hw.output %rden, %data, %empty : i1, i3, i1
// HW-LABEL:     hw.module @i1FifoLoopback(in %in : i3, in %in_empty : i1, in %out_rden : i1, out in_rden : i1, out out : i3, out out_empty : i1)
// HW-NEXT:        hw.output %out_rden, %in, %in_empty : i1, i3, i1
hw.module @i1FifoLoopback(in %in: !esi.channel<i3, FIFO>, out out: !esi.channel<i3, FIFO>) {
  hw.output %in : !esi.channel<i3, FIFO>
}

// IFACE-LABEL:  hw.module @fifoLoopbackTop()
// IFACE-NEXT:     %data, %empty = esi.unwrap.fifo %chanOutput, %foo.in_rden : !esi.channel<i3, FIFO>
// IFACE-NEXT:     %chanOutput, %rden = esi.wrap.fifo %foo.out, %foo.out_empty : !esi.channel<i3, FIFO>
// IFACE-NEXT:     %foo.in_rden, %foo.out, %foo.out_empty = hw.instance "foo" @i1FifoLoopback(in: %data: i3, in_empty: %empty: i1, out_rden: %rden: i1) -> (in_rden: i1, out: i3, out_empty: i1)
// HW-LABEL:     hw.module @fifoLoopbackTop()
// HW-NEXT:        %foo.in_rden, %foo.out, %foo.out_empty = hw.instance "foo" @i1FifoLoopback(in: %foo.out: i3, in_empty: %foo.out_empty: i1, out_rden: %foo.in_rden: i1) -> (in_rden: i1, out: i3, out_empty: i1)
hw.module @fifoLoopbackTop() {
  %chan = hw.instance "foo" @i1FifoLoopback(in: %chan: !esi.channel<i3, FIFO>) -> (out: !esi.channel<i3, FIFO>)
}

// IFACE-LABEL:  hw.module @structFifoLoopback(in %in_in : !hw.struct<a: i3, b: i7>, in %in_flatBroke_in : i1, in %out_readEnable_in : i1, out in_readEnable : i1, out out : !hw.struct<a: i3, b: i7>, out out_flatBroke : i1)
// IFACE-NEXT:     %chanOutput, %rden = esi.wrap.fifo %in_in, %in_flatBroke_in : !esi.channel<!hw.struct<a: i3, b: i7>, FIFO>
// IFACE-NEXT:     %data, %empty = esi.unwrap.fifo %chanOutput, %out_readEnable_in : !esi.channel<!hw.struct<a: i3, b: i7>, FIFO>
// IFACE-NEXT:     hw.output %rden, %data, %empty : i1, !hw.struct<a: i3, b: i7>, i1
!st1 = !hw.struct<a: i3, b: i7>
hw.module @structFifoLoopback(in %in: !esi.channel<!st1, FIFO>, out out: !esi.channel<!st1, FIFO>)
    attributes {esi.portFlattenStructs, esi.portRdenSuffix="_readEnable",
                esi.portEmptySuffix="_flatBroke", esi.portInSuffix="_in"} {
  hw.output %in : !esi.channel<!st1, FIFO>
}

// IFACE-LABEL:  hw.module @structFifoLoopbackTop()
// IFACE-NEXT:    %data, %empty = esi.unwrap.fifo %chanOutput, %foo.in_readEnable : !esi.channel<!hw.struct<a: i3, b: i7>, FIFO>
// IFACE-NEXT:    %chanOutput, %rden = esi.wrap.fifo %foo.out, %foo.out_flatBroke : !esi.channel<!hw.struct<a: i3, b: i7>, FIFO>
// IFACE-NEXT:    %foo.in_readEnable, %foo.out, %foo.out_flatBroke = hw.instance "foo" @structFifoLoopback(in_in: %data: !hw.struct<a: i3, b: i7>, in_flatBroke_in: %empty: i1, out_readEnable_in: %rden: i1) -> (in_readEnable: i1, out: !hw.struct<a: i3, b: i7>, out_flatBroke: i1)
// IFACE-NEXT:    hw.output
hw.module @structFifoLoopbackTop() {
  %chan = hw.instance "foo" @structFifoLoopback(in: %chan: !esi.channel<!st1, FIFO>) -> (out: !esi.channel<!st1, FIFO>)
}

// IFACE-LABEL:  hw.module @i1FifoDelay1Loopback(in %in : i3, in %in_empty : i1, in %out_rden : i1, out in_rden : i1, out out : i3, out out_empty : i1)
// IFACE-NEXT:     %chanOutput, %rden = esi.wrap.fifo %in, %in_empty : !esi.channel<i3, FIFO(1)>
// IFACE-NEXT:     %data, %empty = esi.unwrap.fifo %chanOutput, %out_rden : !esi.channel<i3, FIFO(1)>
// IFACE-NEXT:     hw.output %rden, %data, %empty : i1, i3, i1
// HW-LABEL:     hw.module @i1FifoDelay1Loopback(in %in : i3, in %in_empty : i1, in %out_rden : i1, out in_rden : i1, out out : i3, out out_empty : i1)
// HW-NEXT:        hw.output %out_rden, %in, %in_empty : i1, i3, i1
hw.module @i1FifoDelay1Loopback(in %in: !esi.channel<i3, FIFO(1)>, out out: !esi.channel<i3, FIFO(1)>) {
  hw.output %in : !esi.channel<i3, FIFO(1)>
}

// IFACE-LABEL:  hw.module @i3LoopbackOddNames(in %in : i3, in %in_good : i1, in %out_letErRip : i1, out in_letErRip_out : i1, out out_out : i3, out out_good_out : i1) attributes {esi.portFlattenStructs, esi.portOutSuffix = "_out", esi.portReadySuffix = "_letErRip", esi.portValidSuffix = "_good"} {
// IFACE-NEXT:    %chanOutput, %ready = esi.wrap.vr %in, %in_good : i3
// IFACE-NEXT:    %rawOutput, %valid = esi.unwrap.vr %chanOutput, %out_letErRip : i3
// IFACE-NEXT:    hw.output %ready, %rawOutput, %valid : i1, i3, i1
hw.module @i3LoopbackOddNames(in %in: !esi.channel<i3>, out out: !esi.channel<i3>)
    attributes {esi.portFlattenStructs, esi.portValidSuffix="_good",
                esi.portReadySuffix="_letErRip", esi.portOutSuffix="_out"} {
  hw.output %in : !esi.channel<i3>
}

// CHECK-LABEL:  hw.module @fifo1(in %clk : !seq.clock, in %rst : i1, in %in : !esi.channel<i32, FIFO>, out out : !esi.channel<i32, FIFO(2)>) {
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     %data, %empty = esi.unwrap.fifo %in, [[R2:%.+]] : !esi.channel<i32, FIFO>
// CHECK-NEXT:     %out, %full, %empty_0 = seq.fifo depth 12 rd_latency 2   in %data rdEn %rden wrEn [[R2]] clk %clk rst %rst : i32
// CHECK-NEXT:     [[R0:%.+]] = comb.xor %empty, %true {sv.namehint = "inputNotEmpty"} : i1
// CHECK-NEXT:     [[R1:%.+]] = comb.xor %full, %true {sv.namehint = "seqFifoNotFull"} : i1
// CHECK-NEXT:     [[R2]] = comb.and [[R0]], [[R1]] {sv.namehint = "inputEn"} : i1
// CHECK-NEXT:     %chanOutput, %rden = esi.wrap.fifo %out, %empty_0 : !esi.channel<i32, FIFO(2)>
// CHECK-NEXT:     hw.output %chanOutput : !esi.channel<i32, FIFO(2)>
hw.module @fifo1(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i32, FIFO>, out out: !esi.channel<i32, FIFO(2)>) {
  %fifo = esi.fifo in %in clk %clk rst %rst depth 12 : !esi.channel<i32, FIFO> -> !esi.channel<i32, FIFO(2)>
  hw.output %fifo : !esi.channel<i32, FIFO(2)>
}

// CHECK-LABEL:  hw.module @fifoValidReadyInput(in %clk : !seq.clock, in %rst : i1, in %in : !esi.channel<i32>, out out : !esi.channel<i32, FIFO(2)>) {
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %in, [[InputEN:%.+]] : i32
// CHECK-NEXT:     %true_0 = hw.constant true
// CHECK-NEXT:     [[DataNotAvailable:%.+]] = comb.xor bin %valid, %true_0 {sv.namehint = "dataNotAvailable"} : i1
// CHECK-NEXT:     %out, %full, %empty = seq.fifo depth 12 rd_latency 2   in %rawOutput rdEn %rden wrEn [[InputEN]] clk %clk rst %rst : i32
// CHECK-NEXT:     [[InputNotEmpty:%.+]] = comb.xor [[DataNotAvailable]], %true {sv.namehint = "inputNotEmpty"} : i1
// CHECK-NEXT:     [[SeqFifoNotFull:%.+]] = comb.xor %full, %true {sv.namehint = "seqFifoNotFull"} : i1
// CHECK-NEXT:     [[InputEN]] = comb.and [[InputNotEmpty]], [[SeqFifoNotFull]] {sv.namehint = "inputEn"} : i1
// CHECK-NEXT:     %chanOutput, %rden = esi.wrap.fifo %out, %empty : !esi.channel<i32, FIFO(2)>
// CHECK-NEXT:     hw.output %chanOutput : !esi.channel<i32, FIFO(2)>
hw.module @fifoValidReadyInput(
    in %clk: !seq.clock, in %rst: i1,
    in %in: !esi.channel<i32, ValidReady>,
    out out: !esi.channel<i32, FIFO(2)>) {
  %fifo = esi.fifo in %in clk %clk rst %rst depth 12 : !esi.channel<i32, ValidReady> -> !esi.channel<i32, FIFO(2)>
  hw.output %fifo : !esi.channel<i32, FIFO(2)>
}

// CHECK-LABEL:  hw.module @fifoValidReadyOutput(in %clk : !seq.clock, in %rst : i1, in %in : !esi.channel<i32, FIFO>, out out : !esi.channel<i32>) {
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     %data, %empty = esi.unwrap.fifo %in, [[InputEN:%.+]] : !esi.channel<i32, FIFO>
// CHECK-NEXT:     %out, %full, %empty_0 = seq.fifo depth 12   in %data rdEn [[OutputRdEn:%.+]] wrEn [[InputEN]] clk %clk rst %rst : i32
// CHECK-NEXT:     [[InputNotEmpty:%.+]] = comb.xor %empty, %true {sv.namehint = "inputNotEmpty"} : i1
// CHECK-NEXT:     [[SeqFifoNotFull:%.+]] = comb.xor %full, %true {sv.namehint = "seqFifoNotFull"} : i1
// CHECK-NEXT:     [[InputEN]] = comb.and [[InputNotEmpty]], [[SeqFifoNotFull]] {sv.namehint = "inputEn"} : i1
// CHECK-NEXT:     %true_1 = hw.constant true
// CHECK-NEXT:     [[R3:%.+]] = comb.xor bin %empty_0, %true_1 : i1
// CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr %out, [[R3]] : i32
// CHECK-NEXT:     [[OutputRdEn]] = comb.and [[R3]], %ready {sv.namehint = "outputRdEn"} : i1
// CHECK-NEXT:     hw.output %chanOutput : !esi.channel<i32>
hw.module @fifoValidReadyOutput(
    in %clk: !seq.clock, in %rst: i1,
    in %in: !esi.channel<i32, FIFO>,
    out out: !esi.channel<i32, ValidReady>) {
  %fifo = esi.fifo in %in clk %clk rst %rst depth 12 : !esi.channel<i32, FIFO> -> !esi.channel<i32, ValidReady>
  hw.output %fifo : !esi.channel<i32, ValidReady>
}

// CHECK-LABEL:  hw.module @testConversion(in %clk : !seq.clock, in %rst : i1) {
// CHECK-NEXT:     [[R0:%.+]] = esi.stage %clk, %rst, [[R4:%.+]] : !esi.channel<i4>
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr [[R0]], %rden : i4
// CHECK-NEXT:     %chanOutput, %rden = esi.wrap.fifo %rawOutput, [[R1:%.+]] : !esi.channel<i4, FIFO>
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     [[R1]] = comb.xor %valid, %true : i1
// CHECK-NEXT:     %data, %empty = esi.unwrap.fifo %chanOutput, [[R2:%.+]] : !esi.channel<i4, FIFO>
// CHECK-NEXT:     %chanOutput_0, %ready = esi.wrap.vr %data, [[R3:%.+]] : i4
// CHECK-NEXT:     [[R2]] = comb.and %ready, [[R3]] : i1
// CHECK-NEXT:     %true_1 = hw.constant true
// CHECK-NEXT:     [[R3]] = comb.xor %empty, %true_1 : i1
// CHECK-NEXT:     [[R4]] = esi.stage %clk, %rst, %chanOutput_0 : !esi.channel<i4>

hw.module @testConversion(in %clk: !seq.clock, in %rst:i1) {
  %intsFifo = esi.buffer %clk, %rst, %intsVR : !esi.channel<i4> -> !esi.channel<i4, FIFO>
  %intsVR = esi.buffer %clk, %rst, %intsFifo : !esi.channel<i4, FIFO> -> !esi.channel<i4>
}

hw.module @testSnoopTx(in %clk: !seq.clock, in %rst:i1, in %input: !esi.channel<i32, FIFO>, out output: !esi.channel<i32>, out fifo_xact: i1, out vr_xact: i1) {
  // Test FIFO signaling
  %fifo_tx_xact, %fifo_tx_data = esi.snoop.xact %input : !esi.channel<i32, FIFO>

  // Test ValidReady signaling
  %vrChan = esi.buffer %clk, %rst, %input : !esi.channel<i32, FIFO> -> !esi.channel<i32>
  %vr_tx_xact, %vr_tx_data = esi.snoop.xact %vrChan : !esi.channel<i32>

  hw.output %vrChan, %fifo_tx_xact, %vr_tx_xact : !esi.channel<i32>, i1, i1
}
// HW-LABEL:   hw.module @testSnoopTx(in %clk : !seq.clock, in %rst : i1, in %input : i32, in %input_empty : i1, in %output_ready : i1, out input_rden : i1, out output : i32, out output_valid : i1, out fifo_xact : i1, out vr_xact : i1) {
// HW:         [[NOT_EMPTY1:%.+]] = comb.xor %input_empty, %true : i1
// HW:         [[FIFO_XACT:%.+]] = comb.and [[NOT_EMPTY1]], [[R2:%.+]] : i1
// HW:         [[R2]] = comb.and %pipelineStage.a_ready, [[NOT_EMPTY2:%.+]] : i1
// HW:         [[NOT_EMPTY2]] = comb.xor %input_empty, %true_0 : i1
// HW:         %pipelineStage.a_ready, %pipelineStage.x, %pipelineStage.x_valid = hw.instance "pipelineStage" @ESI_PipelineStage2<WIDTH: ui32 = 32>(clk: %clk: !seq.clock, rst: %rst: i1, a: %input: i32, a_valid: %3: i1, x_ready: %output_ready: i1) -> (a_ready: i1, x: i32, x_valid: i1)
// HW:         [[VR_XACT:%.+]] = comb.and %pipelineStage.x_valid, %output_ready : i1
// HW:         hw.output [[R2]], %pipelineStage.x, %pipelineStage.x_valid, [[FIFO_XACT]], [[VR_XACT]] : i1, i32, i1, i1, i1

// Test snoop on a channel passed through a module (tests channels at module boundaries)
hw.module @SnoopPassthrough(in %chan_in: !esi.channel<i32>, out chan_out: !esi.channel<i32>, out snoop_valid: i1, out snoop_ready: i1, out snoop_data: i32) {
  %valid, %ready, %data = esi.snoop.vr %chan_in : !esi.channel<i32>
  hw.output %chan_in, %valid, %ready, %data : !esi.channel<i32>, i1, i1, i32
}
// HW-LABEL: hw.module @SnoopPassthrough(in %chan_in : i32, in %chan_in_valid : i1, in %chan_out_ready : i1, out chan_in_ready : i1, out chan_out : i32, out chan_out_valid : i1, out snoop_valid : i1, out snoop_ready : i1, out snoop_data : i32) {
// HW:         hw.output %chan_out_ready, %chan_in, %chan_in_valid, %chan_in_valid, %chan_out_ready, %chan_in : i1, i32, i1, i1, i1, i32

hw.module @SnoopWithInstanceConsumer(in %in_data: i32, in %in_valid: i1, out out: !esi.channel<i32>, out snoop_valid: i1, out snoop_ready: i1, out snoop_data: i32) {
  %chan, %ready_out = esi.wrap.vr %in_data, %in_valid : i32
  %valid, %ready_snoop, %data = esi.snoop.vr %chan : !esi.channel<i32>
  %out, %pt_snoop_valid, %pt_snoop_ready, %pt_snoop_data = hw.instance "passthrough" @SnoopPassthrough(chan_in: %chan: !esi.channel<i32>) -> (chan_out: !esi.channel<i32>, snoop_valid: i1, snoop_ready: i1, snoop_data: i32)
  hw.output %out, %valid, %ready_snoop, %data : !esi.channel<i32>, i1, i1, i32
}
// HW-LABEL: hw.module @SnoopWithInstanceConsumer(in %in_data : i32, in %in_valid : i1, in %out_ready : i1, out out : i32, out out_valid : i1, out snoop_valid : i1, out snoop_ready : i1, out snoop_data : i32) {
// HW:         %passthrough.chan_in_ready, %passthrough.chan_out, %passthrough.chan_out_valid, %passthrough.snoop_valid, %passthrough.snoop_ready, %passthrough.snoop_data = hw.instance "passthrough" @SnoopPassthrough(chan_in: %in_data: i32, chan_in_valid: %in_valid: i1, chan_out_ready: %out_ready: i1) -> (chan_in_ready: i1, chan_out: i32, chan_out_valid: i1, snoop_valid: i1, snoop_ready: i1, snoop_data: i32)
// HW:         hw.output %passthrough.chan_out, %passthrough.chan_out_valid, %in_valid, %passthrough.chan_in_ready, %in_data : i32, i1, i1, i1, i32

// Test snoop transaction on FIFO channels at module boundaries
hw.module @SnoopFIFOPassthrough(in %chan_in: !esi.channel<i32, FIFO>, out chan_out: !esi.channel<i32, FIFO>, out snoop_xact: i1, out snoop_data: i32) {
  %transaction, %data = esi.snoop.xact %chan_in : !esi.channel<i32, FIFO>
  hw.output %chan_in, %transaction, %data : !esi.channel<i32, FIFO>, i1, i32
}
// HW-LABEL: hw.module @SnoopFIFOPassthrough(in %chan_in : i32, in %chan_in_empty : i1, in %chan_out_rden : i1, out chan_in_rden : i1, out chan_out : i32, out chan_out_empty : i1, out snoop_xact : i1, out snoop_data : i32) {
// HW:         [[NOT_EMPTY:%.+]] = comb.xor %chan_in_empty, %true : i1
// HW:         [[XACT:%.+]] = comb.and [[NOT_EMPTY]], %chan_out_rden : i1
// HW:         hw.output %chan_out_rden, %chan_in, %chan_in_empty, [[XACT]], %chan_in : i1, i32, i1, i1, i32

hw.module @SnoopFIFOWithInstanceConsumer(in %in_data: i32, in %in_empty: i1, out out: !esi.channel<i32, FIFO>, out snoop_xact: i1, out snoop_data: i32) {
  %chan, %rden_out = esi.wrap.fifo %in_data, %in_empty : !esi.channel<i32, FIFO>
  %transaction, %data = esi.snoop.xact %chan : !esi.channel<i32, FIFO>
  %out, %pt_snoop_xact, %pt_snoop_data = hw.instance "fifo_passthrough" @SnoopFIFOPassthrough(chan_in: %chan: !esi.channel<i32, FIFO>) -> (chan_out: !esi.channel<i32, FIFO>, snoop_xact: i1, snoop_data: i32)
  hw.output %out, %transaction, %data : !esi.channel<i32, FIFO>, i1, i32
}
// HW-LABEL: hw.module @SnoopFIFOWithInstanceConsumer(in %in_data : i32, in %in_empty : i1, in %out_rden : i1, out out : i32, out out_empty : i1, out snoop_xact : i1, out snoop_data : i32) {
// HW:         [[NOT_EMPTY:%.+]] = comb.xor %in_empty, %true : i1
// HW:         [[XACT:%.+]] = comb.and [[NOT_EMPTY]], %fifo_passthrough.chan_in_rden : i1
// HW:         %fifo_passthrough.chan_in_rden, %fifo_passthrough.chan_out, %fifo_passthrough.chan_out_empty, %fifo_passthrough.snoop_xact, %fifo_passthrough.snoop_data = hw.instance "fifo_passthrough" @SnoopFIFOPassthrough(chan_in: %in_data: i32, chan_in_empty: %in_empty: i1, chan_out_rden: %out_rden: i1) -> (chan_in_rden: i1, chan_out: i32, chan_out_empty: i1, snoop_xact: i1, snoop_data: i32)
// HW:         hw.output %fifo_passthrough.chan_out, %fifo_passthrough.chan_out_empty, [[XACT]], %in_data : i32, i1, i1, i32
