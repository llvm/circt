// RUN: circt-opt %s --seq-infer-clock-ports | FileCheck %s
// RUN: circt-opt %s --seq-infer-clock-ports="promote-public-ports=false" | FileCheck %s --check-prefixes=CHECK-NOPUB
// RUN: circt-opt %s --seq-infer-clock-ports | circt-opt --seq-infer-clock-ports | FileCheck %s

// CHECK-LABEL: hw.module private @Leaf(in %clk : !seq.clock
// CHECK-NOT: seq.to_clock
// CHECK: seq.firreg %d clock %clk : i32
hw.module private @Leaf(in %clk : i1, in %d : i32, out q : i32) {
  %clock = seq.to_clock %clk
  %q = seq.firreg %d clock %clock : i32
  hw.output %q : i32
}

// CHECK-LABEL: hw.module private @Mid(in %clk : !seq.clock
// CHECK: hw.instance "l" @Leaf(clk: %clk: !seq.clock, d: %d: i32) -> (q: i32)
hw.module private @Mid(in %clk : i1, in %d : i32, out q : i32) {
  %q = hw.instance "l" @Leaf(clk: %clk: i1, d: %d: i32) -> (q: i32)
  hw.output %q : i32
}

// CHECK-LABEL: hw.module @Top(in %clk : !seq.clock
// CHECK: hw.instance "m" @Mid(clk: %clk: !seq.clock, d: %d: i32) -> (q: i32)
// CHECK: seq.clock_inv %clk
hw.module @Top(in %clk : i1, in %d : i32, out q : i32) {
  %q = hw.instance "m" @Mid(clk: %clk: i1, d: %d: i32) -> (q: i32)
  %clock = seq.to_clock %clk
  %inv = seq.clock_inv %clock
  %out = seq.firreg %q clock %inv : i32
  hw.output %out : i32
}

// CHECK-NOPUB-LABEL: hw.module @Top(in %clk : i1
// CHECK-NOPUB: seq.to_clock %clk


// CHECK-LABEL: hw.module private @Data(in %clk : !seq.clock
// CHECK: [[WIRE:%.+]] = seq.from_clock %clk
// CHECK: hw.output [[WIRE]], %q : i1, i32
hw.module private @Data(in %clk : i1, out o : i1, out q : i32) {
  %c42 = hw.constant 42 : i32
  %clock = seq.to_clock %clk
  %q = seq.firreg %c42 clock %clock : i32
  hw.output %clk, %q : i1, i32
}

// CHECK-LABEL: hw.module private @SubData
// CHECK: hw.output %x : i1
hw.module private @SubData(in %x : i1, out o : i1) {
  hw.output %x : i1
}

// CHECK-LABEL: hw.module private @Mixed(in %clk : !seq.clock
// CHECK: [[DATAWIRE:%.+]] = seq.from_clock %clk
// CHECK: @Leaf(clk: %clk: !seq.clock
// CHECK: @SubData(x: [[DATAWIRE]]: i1
hw.module private @Mixed(in %clk : i1, out q : i32) {
  %c42 = hw.constant 42 : i32
  %clock = seq.to_clock %clk
  %a = hw.instance "clock" @Leaf(clk: %clk: i1, d: %c42: i32) -> (q: i32)
  %b = hw.instance "data" @SubData(x: %clk: i1) -> (o: i1)
  hw.output %a : i32
}

// CHECK-LABEL: hw.module private @Gate
// CHECK: comb.and %clk, %en : i1
// CHECK-LABEL: hw.module @GateUse(in %clk : i1
// CHECK: @Gate(clk: %clk: i1
// CHECK: seq.to_clock %gate.gclk
hw.module private @Gate(in %clk : i1, in %en : i1, out gclk : i1) {
  %g = comb.and %clk, %en : i1
  hw.output %g : i1
}

hw.module @GateUse(in %clk : i1, in %en : i1, out q : i32) {
  %g = hw.instance "gate" @Gate(clk: %clk: i1, en: %en: i1) -> (gclk: i1)
  %c42 = hw.constant 42 : i32
  %clock = seq.to_clock %g
  %q = seq.firreg %c42 clock %clock : i32
  hw.output %q : i32
}

// CHECK-LABEL: hw.module private @ExternUse(in %clk : !seq.clock
// CHECK: [[EXTWIRE:%.+]] = seq.from_clock %clk
// CHECK: @Ext(clk: [[EXTWIRE]]: i1
hw.module.extern @Ext(in %clk : i1, out o : i1)

hw.module private @ExternUse(in %clk : i1, out q : i32) {
  %c42 = hw.constant 42 : i32
  %child = hw.instance "child" @Leaf(clk: %clk: i1, d: %c42: i32) -> (q: i32)
  %ext = hw.instance "ext" @Ext(clk: %clk: i1) -> (o: i1)
  hw.output %child : i32
}

// CHECK-LABEL: hw.module private @InnerSym(in %clk : i1
// CHECK: seq.to_clock %clk
hw.module private @InnerSym(
    in %clk : i1 {hw.exportPort = #hw<innerSym@sym>}, out q : i32) {
  %c42 = hw.constant 42 : i32
  %clock = seq.to_clock %clk
  %q = seq.firreg %c42 clock %clock : i32
  hw.output %q : i32
}

// CHECK-LABEL: hw.module private @Assert(in %clk : !seq.clock
// CHECK: [[ASSERTWIRE:%.+]] = seq.from_clock %clk
// CHECK: verif.clocked_assert %true, posedge [[ASSERTWIRE]] : i1
hw.module private @Assert(in %clk : i1, out q : i1) {
  %c42 = hw.constant 42 : i32
  %clock = seq.to_clock %clk
  %q = seq.firreg %c42 clock %clock : i32
  %prop = hw.constant 1 : i1
  verif.clocked_assert %prop, posedge %clk : i1
  hw.output %prop : i1
}

// CHECK-LABEL: hw.module private @Already(in %clk : !seq.clock
// CHECK-NOT: seq.to_clock
// CHECK: seq.firreg %c42_i32 clock %clk : i32
hw.module private @Already(in %clk : !seq.clock, out q : i32) {
  %c42 = hw.constant 42 : i32
  %q = seq.firreg %c42 clock %clk : i32
  hw.output %q : i32
}

// CHECK-LABEL: hw.module private @CanonicalClockOps(in %clk : !seq.clock
// CHECK-NOT: seq.from_clock
// CHECK: [[GATED:%.+]] = seq.clock_gate %clk, %en
// CHECK: seq.clock_inv [[GATED]]
// CHECK: [[MUXED:%.+]] = seq.clock_mux %en, %clk, %clk2
// CHECK: seq.clock_div [[MUXED]] by 1
hw.module private @CanonicalClockOps(in %clk : !seq.clock,
                                     in %clk2 : !seq.clock, in %en : i1,
                                     out q : i32) {
  %gated = seq.clock_gate %clk, %en
  %inv = seq.clock_inv %gated
  %muxed = seq.clock_mux %en, %clk, %clk2
  %div = seq.clock_div %muxed by 1
  %c42 = hw.constant 42 : i32
  %q = seq.firreg %c42 clock %div : i32
  hw.output %q : i32
}

// Forwarding modules are promoted even if they don't use the clock themselves.
// CHECK-LABEL: hw.module private @FwdInner(in %clk : !seq.clock, out o : !seq.clock)
// CHECK-NEXT: hw.output %clk : !seq.clock
hw.module private @FwdInner(in %clk : i1, out o : i1) {
  hw.output %clk : i1
}

// CHECK-LABEL: hw.module private @FwdMid(in %clk : !seq.clock, out o : !seq.clock)
// CHECK-NEXT: %inner.o = hw.instance "inner" @FwdInner(clk: %clk: !seq.clock) -> (o: !seq.clock)
// CHECK-NEXT: hw.output %inner.o : !seq.clock
hw.module private @FwdMid(in %clk : i1, out o : i1) {
  %o = hw.instance "inner" @FwdInner(clk: %clk: i1) -> (o: i1)
  hw.output %o : i1
}

// Graph regions allow the cast to precede the instance. Data uses share one
// `seq.from_clock`.
// CHECK-LABEL: hw.module private @FwdTop(in %clk : !seq.clock, out a : i1, out b : i1, out q : i32)
// CHECK-NOT: seq.to_clock
// CHECK: seq.firreg %c42_i32 clock %mid.o : i32
// CHECK-NEXT: %mid.o = hw.instance "mid" @FwdMid(clk: %clk: !seq.clock) -> (o: !seq.clock)
// CHECK-NEXT: [[FWD:%.+]] = seq.from_clock %mid.o
// CHECK-NEXT: comb.xor [[FWD]], [[FWD]] : i1
// CHECK-NEXT: comb.and [[FWD]], [[FWD]] : i1
hw.module private @FwdTop(in %clk : i1, out a : i1, out b : i1, out q : i32) {
  %c42 = hw.constant 42 : i32
  %clock = seq.to_clock %o
  %q = seq.firreg %c42 clock %clock : i32
  %o = hw.instance "mid" @FwdMid(clk: %clk: i1) -> (o: i1)
  %x = comb.xor %o, %o : i1
  %y = comb.and %o, %o : i1
  hw.output %x, %y, %q : i1, i1, i32
}

// Gating logic stays `i1`, even when forwarded through another module.
// CHECK-LABEL: hw.module private @GateFwd(in %clk : i1, in %en : i1, out gclk : i1)
// CHECK-LABEL: hw.module private @GateFwdUse(in %clk : i1
// CHECK: seq.to_clock %fwd.gclk
hw.module private @GateFwd(in %clk : i1, in %en : i1, out gclk : i1) {
  %g = hw.instance "gate" @Gate(clk: %clk: i1, en: %en: i1) -> (gclk: i1)
  hw.output %g : i1
}

hw.module private @GateFwdUse(in %clk : i1, in %en : i1, out q : i32) {
  %g = hw.instance "fwd" @GateFwd(clk: %clk: i1, en: %en: i1) -> (gclk: i1)
  %c42 = hw.constant 42 : i32
  %clock = seq.to_clock %g
  %q = seq.firreg %c42 clock %clock : i32
  hw.output %q : i32
}

// Forwarding an unpromotable port blocks promotion of the output.
// CHECK-LABEL: hw.module private @SymFwd(in %clk : i1 {{.*}}, out o : i1)
// CHECK-LABEL: hw.module private @SymFwdUse(in %clk : i1
// CHECK: seq.to_clock %fwd.o
hw.module private @SymFwd(
    in %clk : i1 {hw.exportPort = #hw<innerSym@sym>}, out o : i1) {
  hw.output %clk : i1
}

hw.module private @SymFwdUse(in %clk : i1, out q : i32) {
  %o = hw.instance "fwd" @SymFwd(clk: %clk: i1) -> (o: i1)
  %c42 = hw.constant 42 : i32
  %clock = seq.to_clock %o
  %q = seq.firreg %c42 clock %clock : i32
  hw.output %q : i32
}
