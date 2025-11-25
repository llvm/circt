// RUN: circt-opt %s --comb-assume-two-valued | FileCheck %s

// CHECK-LABEL: hw.module @ceq
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp eq %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @ceq(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp ceq %a, %b : i1
  hw.output %0 : i1
}
