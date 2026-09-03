// RUN: circt-opt %s --seq-propagate-clock-domains | FileCheck %s

// A comb.mux joins the input domains. The register body is evaluated separately
// for each instance: the annotation on the shared module operation is therefore
// the union, while the two hw.instance results retain their individual domains.
// CHECK-LABEL: hw.module private @Register
hw.module private @Register(in %data: i1, in %clock: !seq.clock, out result: i1) {
  // CHECK: seq.compreg %data, %clock {seq.clock_domains = {{\[\["A", "B"\]\]}}} : i1
  %reg = seq.compreg %data, %clock : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @Top(
// CHECK-SAME: in %dataA : i1 {seq.clock_domains = ["A"]},
// CHECK-SAME: in %dataB : i1 {seq.clock_domains = ["B"]},
// CHECK-SAME: in %clockA : !seq.clock {seq.clock_domains = ["A"]},
// CHECK-SAME: in %clockB : !seq.clock {seq.clock_domains = ["B"]},
// CHECK-SAME: out resultA : i1,
// CHECK-SAME: out resultB : i1)
hw.module @Top(
    in %dataA: i1 {seq.clock_domains = ["A"]},
    in %dataB: i1 {seq.clock_domains = ["B"]},
    in %clockA: !seq.clock {seq.clock_domains = ["A"]},
    in %clockB: !seq.clock {seq.clock_domains = ["B"]},
    out resultA: i1,
    out resultB: i1) {
  // CHECK: %[[MIXED:.*]] = comb.mux %dataA, %dataA, %dataB {seq.clock_domains = {{\[\["A", "B"\]\]}}} : i1
  %mixed = comb.mux %dataA, %dataA, %dataB : i1
  // CHECK: hw.instance "registerA" @Register(data: %[[MIXED]]: i1, clock: %clockA: !seq.clock) -> (result: i1) {seq.clock_domains = {{\[\["A"\]\]}}}
  %resultA = hw.instance "registerA" @Register(data: %mixed : i1, clock: %clockA : !seq.clock) -> (result: i1)
  // CHECK: hw.instance "registerB" @Register(data: %[[MIXED]]: i1, clock: %clockB: !seq.clock) -> (result: i1) {seq.clock_domains = {{\[\["B"\]\]}}}
  %resultB = hw.instance "registerB" @Register(data: %mixed : i1, clock: %clockB : !seq.clock) -> (result: i1)
  hw.output %resultA, %resultB : i1, i1
}

// Public clocks without an explicit annotation receive stable, distinct domain
// names. Unannotated data instead conservatively starts in the unknown domain.
// CHECK-LABEL: hw.module @Unannotated(
// CHECK-SAME: in %data : i1,
// CHECK-SAME: in %clockA : !seq.clock,
// CHECK-SAME: in %clockB : !seq.clock,
// CHECK-SAME: out resultA : i1,
// CHECK-SAME: out resultB : i1)
hw.module @Unannotated(in %data: i1, in %clockA: !seq.clock,
                       in %clockB: !seq.clock, out resultA: i1,
                       out resultB: i1) {
  // CHECK: %[[UNKNOWN:.*]] = comb.and %data, %data {seq.clock_domains = {{\[\["<unknown>"\]\]}}} : i1
  %unknown = comb.and %data, %data : i1
  // CHECK: seq.compreg %[[UNKNOWN]], %clockA {seq.clock_domains = {{\[\["Unannotated.clockA"\]\]}}} : i1
  %resultA = seq.compreg %unknown, %clockA : i1
  // CHECK: seq.compreg %[[UNKNOWN]], %clockB {seq.clock_domains = {{\[\["Unannotated.clockB"\]\]}}} : i1
  %resultB = seq.compreg %unknown, %clockB : i1
  hw.output %resultA, %resultB : i1, i1
}
