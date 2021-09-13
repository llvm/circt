// RUN: circt-opt %s --lower-esi-to-physical -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-esi-ports -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=IFACE %s
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=HW %s

hw.module.extern @Sender(%clk: i1) -> ( %x: !esi.channel<i4>, %y: i8 )
hw.module.extern @ArrSender() -> (%x: !esi.channel<!hw.array<4xi64>>)
hw.module.extern @Reciever(%a: !esi.channel<i4>, %clk: i1)

// CHECK-LABEL: hw.module.extern @Sender(%clk: i1) -> (%x: !esi.channel<i4>, %y: i8)
// CHECK-LABEL: hw.module.extern @Reciever(%a: !esi.channel<i4>, %clk: i1)

// IFACE-LABEL: sv.interface @IValidReady_i4 {
// IFACE-NEXT:    sv.interface.signal @valid : i1
// IFACE-NEXT:    sv.interface.signal @ready : i1
// IFACE-NEXT:    sv.interface.signal @data : i4
// IFACE-NEXT:    sv.interface.modport @sink ("input" @ready, "output" @valid, "output" @data)
// IFACE-NEXT:    sv.interface.modport @source ("input" @valid, "input" @data, "output" @ready)
// IFACE-LABEL: sv.interface @IValidReady_ArrayOf4xi64 {
// IFACE-NEXT:    sv.interface.signal @valid : i1
// IFACE-NEXT:    sv.interface.signal @ready : i1
// IFACE-NEXT:    sv.interface.signal @data : !hw.array<4xi64>
// IFACE-NEXT:    sv.interface.modport @sink  ("input" @ready, "output" @valid, "output" @data)
// IFACE-NEXT:    sv.interface.modport @source  ("input" @valid, "input" @data, "output" @ready)
// IFACE-LABEL: hw.module.extern @Sender(%clk: i1, %x: !sv.modport<@IValidReady_i4::@sink>) -> (%y: i8)
// IFACE-LABEL: hw.module.extern @ArrSender(%x: !sv.modport<@IValidReady_ArrayOf4xi64::@sink>)
// IFACE-LABEL: hw.module.extern @Reciever(%a: !sv.modport<@IValidReady_i4::@source>, %clk: i1)


hw.module @test(%clk:i1, %rstn:i1) {

  %esiChan2, %0 = hw.instance "sender2" @Sender(clk: %clk: i1) -> (x: !esi.channel<i4>, y: i8)
  %bufferedChan2 = esi.buffer %clk, %rstn, %esiChan2 { stages = 4 } : i4
  hw.instance "recv2" @Reciever (a: %bufferedChan2: !esi.channel<i4>, clk: %clk: i1) -> ()

  // CHECK:      %sender2.x, %sender2.y = hw.instance "sender2" @Sender(clk: %clk: i1) -> (x: !esi.channel<i4>, y: i8)
  // CHECK-NEXT:  %0 = esi.stage %clk, %rstn, %sender2.x : i4
  // CHECK-NEXT:  %1 = esi.stage %clk, %rstn, %0 : i4
  // CHECK-NEXT:  %2 = esi.stage %clk, %rstn, %1 : i4
  // CHECK-NEXT:  %3 = esi.stage %clk, %rstn, %2 : i4
  // CHECK-NEXT:  hw.instance "recv2" @Reciever(a: %3: !esi.channel<i4>, clk: %clk: i1) -> ()

  // IFACE-LABEL: hw.module @test(%clk: i1, %rstn: i1) {
  // IFACE-NEXT:    %0 = sv.interface.instance {name = "i4FromSender2"} : !sv.interface<@IValidReady_i4>
  // IFACE-NEXT:    %1 = sv.modport.get %0 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // IFACE-NEXT:    %2 = esi.wrap.iface %1 : !sv.modport<@IValidReady_i4::@source> -> !esi.channel<i4>
  // IFACE-NEXT:    %3 = sv.modport.get %0 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // IFACE-NEXT:    %sender2.y = hw.instance "sender2" @Sender(clk: %clk: i1, x: %3: !sv.modport<@IValidReady_i4::@sink>) -> (y: i8)
  // IFACE-NEXT:    %4 = esi.buffer %clk, %rstn, %2 {stages = 4 : i64} : i4
  // IFACE-NEXT:    %5 = sv.interface.instance {name = "i4ToRecv2"} : !sv.interface<@IValidReady_i4>
  // IFACE-NEXT:    %6 = sv.modport.get %5 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // IFACE-NEXT:    esi.unwrap.iface %4 into %6 : (!esi.channel<i4>, !sv.modport<@IValidReady_i4::@sink>)
  // IFACE-NEXT:    %7 = sv.modport.get %5 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // IFACE-NEXT:    hw.instance "recv2" @Reciever(a: %7: !sv.modport<@IValidReady_i4::@source>, clk: %clk: i1) -> ()

  // After all 3 ESI lowering passes, there shouldn't be any ESI constructs!
  // HW-NOT: esi
}

hw.module @add11(%clk: i1, %ints: !esi.channel<i32>) -> (%mutatedInts: !esi.channel<i32>, %c4: i4) {
  %i, %i_valid = esi.unwrap.vr %ints, %rdy : i32
  %c11 = hw.constant 11 : i32
  %m = comb.add %c11, %i : i32
  %mutInts, %rdy = esi.wrap.vr %m, %i_valid : i32
  %c4 = hw.constant 0 : i4
  hw.output %mutInts, %c4 : !esi.channel<i32>, i4
}
// HW-LABEL: hw.module @add11(%clk: i1, %ints: i32, %ints_valid: i1, %mutatedInts_ready: i1) -> (%mutatedInts: i32, %mutatedInts_valid: i1, %c4: i4, %ints_ready: i1) {
// HW:   %{{.+}} = hw.constant 11 : i32
// HW:   [[RES0:%.+]] = comb.add %{{.+}}, %ints : i32
// HW:   %{{.+}} = hw.constant 0 : i4
// HW:   hw.output [[RES0]], %ints_valid, %{{.+}}, %mutatedInts_ready : i32, i1, i4, i1

hw.module @InternRcvr(%in: !esi.channel<!hw.array<4xi8>>) -> () {}

hw.module @test2(%clk:i1, %rstn:i1) {
  %ints, %c4 = hw.instance "adder" @add11(clk: %clk: i1, ints: %ints: !esi.channel<i32>) -> (mutatedInts: !esi.channel<i32>, c4: i4)

  %nullBit = esi.null : !esi.channel<i4>
  hw.instance "nullRcvr" @Reciever(a: %nullBit: !esi.channel<i4>, clk: %clk: i1) -> ()

  %nullArray = esi.null : !esi.channel<!hw.array<4xi8>>
  hw.instance "nullInternRcvr" @InternRcvr(in: %nullArray: !esi.channel<!hw.array<4xi8>>) -> ()
}
// HW-LABEL: hw.module @test2(%clk: i1, %rstn: i1) {
// HW:   %adder.mutatedInts, %adder.mutatedInts_valid, %adder.c4, %adder.ints_ready = hw.instance "adder" @add11(clk: %clk: i1, ints: %adder.mutatedInts: i32, ints_valid: %adder.mutatedInts_valid: i1, mutatedInts_ready: %adder.ints_ready: i1) -> (mutatedInts: i32, mutatedInts_valid: i1, c4: i4, ints_ready: i1)
// HW:   [[ZERO:%.+]] = hw.bitcast %c0_i4 : (i4) -> i4
// HW:   sv.interface.signal.assign %1(@IValidReady_i4::@data) = [[ZERO]] : i4
// HW:   [[ZM:%.+]] = sv.modport.get %{{.+}} @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
// HW:   hw.instance "nullRcvr" @Reciever(a: [[ZM]]: !sv.modport<@IValidReady_i4::@source>, clk: %clk: i1) -> ()
// HW:   %c0_i32 = hw.constant 0 : i32
// HW:   [[ZA:%.+]] = hw.bitcast %c0_i32 : (i32) -> !hw.array<4xi8>
// HW:   %nullInternRcvr.in_ready = hw.instance "nullInternRcvr" @InternRcvr(in: [[ZA]]: !hw.array<4xi8>, in_valid: %false_0: i1) -> (in_ready: i1)

hw.module @twoChannelArgs(%clk: i1, %ints: !esi.channel<i32>, %foo: !esi.channel<i7>) -> () {
  %rdy = hw.constant 1 : i1
  %i, %i_valid = esi.unwrap.vr %ints, %rdy : i32
  %i2, %i2_valid = esi.unwrap.vr %foo, %rdy : i7
}
// HW-LABEL: hw.module @twoChannelArgs(%clk: i1, %ints: i32, %ints_valid: i1, %foo: i7, %foo_valid: i1) -> (%ints_ready: i1, %foo_ready: i1)
// HW:   %true = hw.constant true
// HW:   hw.output %true, %true : i1, i1
