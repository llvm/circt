// RUN: circt-opt --lower-esi-ports %s | circt-opt -verify-diagnostics | FileCheck %s

// Lowered ports take the place of their original port,
// so ports that are interleaved by direction stay interleaved.

// CHECK-LABEL: hw.module @Interleaved(in %clk : !seq.clock, out o : i8, in %c : i8, in %c_valid : i1, out c_ready : i1, out p : i1, in %rst : i1)
// CHECK:         %chanOutput, %ready = esi.wrap.vr %c, %c_valid
// CHECK:         %rawOutput, %valid = esi.unwrap.vr %chanOutput
// CHECK:         hw.output %rawOutput, %ready, %valid : i8, i1, i1
hw.module @Interleaved(in %clk : !seq.clock, out o : i8, in %c : !esi.channel<i8, ValidReady>, out p : i1, in %rst : i1) {
  %rdy = hw.constant true
  %d, %v = esi.unwrap.vr %c, %rdy : !esi.channel<i8, ValidReady>
  hw.output %d, %v : i8, i1
}

// CHECK-LABEL: hw.module @OutputFirst(out o : i8, in %clk : !seq.clock, in %c : i8, in %c_valid : i1, out c_ready : i1, out p : i1)
hw.module @OutputFirst(out o : i8, in %clk : !seq.clock, in %c : !esi.channel<i8, ValidReady>, out p : i1) {
  %rdy = hw.constant true
  %d, %v = esi.unwrap.vr %c, %rdy : !esi.channel<i8, ValidReady>
  hw.output %d, %v : i8, i1
}

// CHECK-LABEL: hw.module @Parent(in %clk : !seq.clock, in %c : i8, in %c_valid : i1, in %rst : i1, out c_ready : i1, out o : i8, out p : i1)
// CHECK:         %rawOutput, %valid = esi.unwrap.vr %chanOutput, %inst.c_ready
// CHECK:         %inst.o, %inst.c_ready, %inst.p = hw.instance "inst" @Interleaved(clk: %clk: !seq.clock, c: %rawOutput: i8, c_valid: %valid: i1, rst: %rst: i1) -> (o: i8, c_ready: i1, p: i1)
// CHECK:         hw.output %ready, %inst.o, %inst.p : i1, i8, i1
hw.module @Parent(in %clk : !seq.clock, in %c : !esi.channel<i8, ValidReady>, in %rst : i1, out o : i8, out p : i1) {
  %o, %p = hw.instance "inst" @Interleaved(clk: %clk: !seq.clock, c: %c: !esi.channel<i8, ValidReady>, rst: %rst: i1) -> (o: i8, p: i1)
  hw.output %o, %p : i8, i1
}
