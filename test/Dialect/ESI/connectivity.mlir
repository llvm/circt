// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --verify-esi-connections

hw.module @Sender(out x: !esi.channel<i1, ValidReady>) {
  %0 = arith.constant 0 : i1
  // Don't transmit any data.
  %ch, %rcvrRdy = esi.wrap.vr %0, %0 : !esi.channel<i1, ValidReady>
  hw.output %ch : !esi.channel<i1, ValidReady>
}
hw.module @Reciever(in %a: !esi.channel<i1, ValidReady>) {
  %rdy = arith.constant 1 : i1
  // Recieve bits.
  %data, %valid = esi.unwrap.vr %a, %rdy : !esi.channel<i1, ValidReady>
}
!FooStruct = !hw.struct<a: si4, b: !hw.array<3 x ui4>>
hw.module @StructRcvr(in %a: !esi.channel<!FooStruct, ValidReady>) {
  %rdy = arith.constant 1 : i1
  // Recieve bits.
  %data, %valid = esi.unwrap.vr %a, %rdy : !esi.channel<!FooStruct, ValidReady>
}

// CHECK-LABEL: hw.module @Sender(out x : !esi.channel<i1, ValidReady>) {
// CHECK:        %chanOutput, %ready = esi.wrap.vr %false, %false : !esi.channel<i1, ValidReady>
// CHECK-LABEL: hw.module @Reciever(in %a : !esi.channel<i1, ValidReady>) {
// CHECK:        %rawOutput, %valid = esi.unwrap.vr %a, %true : !esi.channel<i1, ValidReady>
// CHECK-LABEL: hw.module @StructRcvr(in %a : !esi.channel<!hw.struct<a: si4, b: !hw.array<3xui4>>, ValidReady>)

hw.module @test(in %clk: !seq.clock, in %rst: i1) {
  %esiChan = hw.instance "sender" @Sender() -> (x: !esi.channel<i1, ValidReady>)
  %bufferedChan = esi.buffer %clk, %rst, %esiChan : !esi.channel<i1, ValidReady> -> !esi.channel<i1, ValidReady>
  hw.instance "recv" @Reciever (a: %bufferedChan: !esi.channel<i1, ValidReady>) -> ()

  // CHECK:  %sender.x = hw.instance "sender" @Sender() -> (x: !esi.channel<i1, ValidReady>)
  // CHECK-NEXT:  %0 = esi.buffer %clk, %rst, %sender.x : !esi.channel<i1, ValidReady> -> !esi.channel<i1, ValidReady>
  // CHECK-NEXT:  hw.instance "recv" @Reciever(a: %0: !esi.channel<i1, ValidReady>) -> ()

  %esiChan2 = hw.instance "sender" @Sender() -> (x: !esi.channel<i1, ValidReady>)
  %bufferedChan2 = esi.buffer %clk, %rst, %esiChan2 { stages = 4 } : !esi.channel<i1, ValidReady> -> !esi.channel<i1, ValidReady>
  hw.instance "recv" @Reciever (a: %bufferedChan2: !esi.channel<i1, ValidReady>) -> ()

  // CHECK-NEXT:  %sender.x_0 = hw.instance "sender" @Sender() -> (x: !esi.channel<i1, ValidReady>)
  // CHECK-NEXT:  [[R1:%.+]] = esi.buffer %clk, %rst, %sender.x_0 {stages = 4 : i64} : !esi.channel<i1, ValidReady> -> !esi.channel<i1, ValidReady>
  // CHECK-NEXT:  hw.instance "recv" @Reciever(a: [[R1]]: !esi.channel<i1, ValidReady>) -> ()

  %valid, %ready, %data = esi.snoop.vr %bufferedChan2 : !esi.channel<i1, ValidReady>
  // CHECK-NEXT:  %valid, %ready, %data = esi.snoop.vr [[R1]] : !esi.channel<i1, ValidReady>

  %nullBit = esi.null : !esi.channel<i1, ValidReady>
  hw.instance "nullRcvr" @Reciever(a: %nullBit: !esi.channel<i1, ValidReady>) -> ()
  // CHECK-NEXT:  [[NULLI1:%.+]] = esi.null : !esi.channel<i1, ValidReady>
  // CHECK-NEXT:  hw.instance "nullRcvr" @Reciever(a: [[NULLI1]]: !esi.channel<i1, ValidReady>) -> ()
}

// CHECK-LABEL: hw.module @i0Typed(in %a : !esi.channel<i0, ValidReady>, in %clk : !seq.clock, in %rst : i1, out x : !esi.channel<i0, ValidReady>) {
// CHECK-NEXT:    %0 = esi.buffer %clk, %rst, %a  : !esi.channel<i0, ValidReady> -> !esi.channel<i0, ValidReady>
// CHECK-NEXT:    %1 = esi.stage %clk, %rst, %0  : !esi.channel<i0, ValidReady>
// CHECK-NEXT:    %rawOutput, %valid = esi.unwrap.vr %1, %ready : !esi.channel<i0, ValidReady>
// CHECK-NEXT:    %chanOutput, %ready = esi.wrap.vr %rawOutput, %valid : !esi.channel<i0, ValidReady>
// CHECK-NEXT:    hw.output %chanOutput : !esi.channel<i0, ValidReady>
// CHECK-NEXT:  }

hw.module @i0Typed(in %a: !esi.channel<i0, ValidReady>, in %clk : !seq.clock, in %rst : i1, out x: !esi.channel<i0, ValidReady>) {
  %bufferedA = esi.buffer %clk, %rst, %a : !esi.channel<i0, ValidReady> -> !esi.channel<i0, ValidReady>
  %stagedA = esi.stage %clk, %rst, %bufferedA : !esi.channel<i0, ValidReady>
  %rawOutput, %valid = esi.unwrap.vr %stagedA, %rcvrRdy : !esi.channel<i0, ValidReady>
  %ch, %rcvrRdy = esi.wrap.vr %rawOutput, %valid : !esi.channel<i0, ValidReady>
  hw.output %ch : !esi.channel<i0, ValidReady>
}

hw.module.extern @i1Fifo(in %in: !esi.channel<i1, FIFO>, out out: !esi.channel<i1, FIFO>)

// CHECK-LABEL:  hw.module @fifoWrapUnwrap()
// CHECK-NEXT:     %chanOutput, %rden = esi.wrap.fifo %data, %empty : !esi.channel<i1, FIFO>
// CHECK-NEXT:     %foo.out = hw.instance "foo" @i1Fifo(in: %chanOutput: !esi.channel<i1, FIFO>) -> (out: !esi.channel<i1, FIFO>)
// CHECK-NEXT:     %data, %empty = esi.unwrap.fifo %foo.out, %rden : !esi.channel<i1, FIFO>
hw.module @fifoWrapUnwrap() {
  %in, %rden = esi.wrap.fifo %data, %empty : !esi.channel<i1, FIFO>
  %out = hw.instance "foo" @i1Fifo(in: %in: !esi.channel<i1, FIFO>) -> (out: !esi.channel<i1, FIFO>)
  %data, %empty = esi.unwrap.fifo %out, %rden : !esi.channel<i1, FIFO>
}

// Non-zero data delays round-trip for all signaling kinds, including the
// default ValidReady.
// CHECK-LABEL: hw.module.extern @DataDelayTypes(in %a : !esi.channel<i8, ValidReady(2)>, in %b : !esi.channel<i8, ValidOnly(1)>, out x : !esi.channel<i8, FIFO(3)>)
hw.module.extern @DataDelayTypes(in %a: !esi.channel<i8, ValidReady(2)>, in %b: !esi.channel<i8, ValidOnly(1)>, out x: !esi.channel<i8, FIFO(3)>)

// CHECK-LABEL:  hw.module @stageDataDelay(in %clk : !seq.clock, in %rst : i1, in %a : !esi.channel<i8, ValidReady(2)>, out x : !esi.channel<i8, ValidReady(2)>)
// CHECK-NEXT:     %0 = esi.stage %clk, %rst, %a : !esi.channel<i8, ValidReady(2)>
// CHECK-NEXT:     hw.output %0 : !esi.channel<i8, ValidReady(2)>
hw.module @stageDataDelay(in %clk: !seq.clock, in %rst: i1, in %a: !esi.channel<i8, ValidReady(2)>, out x: !esi.channel<i8, ValidReady(2)>) {
  %staged = esi.stage %clk, %rst, %a : !esi.channel<i8, ValidReady(2)>
  hw.output %staged : !esi.channel<i8, ValidReady(2)>
}
