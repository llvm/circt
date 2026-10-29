// `esi.wrap.vr` doesn't print a channel's data delay, so this output can't be
// reparsed.
// RUN: circt-opt --lower-esi-ports %s -verify-diagnostics | FileCheck %s

// Lowering a ValidReady port keeps its data delay.
// CHECK-LABEL:  hw.module @vrDataDelay(in %clk : !seq.clock, in %rst : i1, in %in : i8, in %in_valid : i1, in %out_ready : i1, out in_ready : i1, out out : i8, out out_valid : i1)
// CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr %in, %in_valid : i8
// CHECK-NEXT:     [[STAGE:%.+]] = esi.stage %clk, %rst, %chanOutput : !esi.channel<i8, ValidReady(2)>
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr [[STAGE]], %out_ready : i8
hw.module @vrDataDelay(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i8, ValidReady(2)>, out out: !esi.channel<i8, ValidReady(2)>) {
  %0 = esi.stage %clk, %rst, %in : !esi.channel<i8, ValidReady(2)>
  hw.output %0 : !esi.channel<i8, ValidReady(2)>
}
