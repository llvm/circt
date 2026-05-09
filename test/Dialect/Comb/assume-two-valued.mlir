// RUN: circt-opt %s --comb-assume-two-valued | FileCheck %s

// CHECK-LABEL: hw.module @ceq
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp bin eq %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @ceq(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp ceq %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @weq
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp bin eq %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @weq(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp weq %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @cne
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp bin ne %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @cne(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp cne %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @wne
// CHECK-NEXT:    [[EQ:%.+]] = comb.icmp bin ne %a, %b : i1
// CHECK-NEXT:    hw.output [[EQ]] : i1
hw.module @wne(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.icmp wne %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @and
// CHECK-NEXT:    %[[X:.*]] = comb.and bin %a, %b
hw.module @and(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @add
// CHECK-NEXT:    %[[X:.*]] = comb.add bin %a, %b
hw.module @add(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.add %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @or
// CHECK-NEXT:    %[[X:.*]] = comb.or bin %a, %b
hw.module @or(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.or %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @sub
// CHECK-NEXT:    %[[X:.*]] = comb.sub bin %a, %b
hw.module @sub(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.sub %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @xor
// CHECK-NEXT:    %[[X:.*]] = comb.xor bin %a, %b
hw.module @xor(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.xor %a, %b : i1
  hw.output %0 : i1
}
