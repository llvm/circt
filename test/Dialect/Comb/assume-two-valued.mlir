// RUN: circt-opt %s --comb-assume-two-valued | FileCheck %s

// CHECK-LABEL: hw.module @ceq
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp eq %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @ceq(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp ceq %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @weq
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp eq %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @weq(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp weq %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @cne
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp ne %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @cne(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp cne %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @wne
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp ne %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @wne(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp wne %a, %b : i1
  hw.output %0 : i1
}
