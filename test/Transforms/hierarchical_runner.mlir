// RUN: circt-opt --pass-pipeline='builtin.module(hierarchical-runner{top-name=top pipeline=canonicalize})' %s | FileCheck %s --check-prefixes=CHECK,EXCLUDE_BOUND
// RUN: circt-opt --pass-pipeline='builtin.module(hierarchical-runner{top-name=top pipeline=canonicalize include-bound-instances=true})' %s | FileCheck %s --check-prefixes=CHECK,INCLUDE_BOUND

// CHECK-LABEL: hw.module @bound
hw.module @bound(out out: i8) {
    %c1_i8 = hw.constant 1 : i8
    %add = comb.add %c1_i8, %c1_i8 : i8
    // EXCLUDE_BOUND: %[[ADD:.+]] = comb.add %c1_i8, %c1_i8
    // EXCLUDE_BOUND-NEXT:  hw.output %[[ADD]] : i8
    // INCLUDE_BOUND: hw.output %c2_i8
    hw.output %add : i8
}

// CHECK-LABEL: hw.module.extern @extern
hw.module.extern @extern()

// CHECK-LABEL: hw.module @top
hw.module @top(out out: i8) {
    %c1_i8 = hw.constant 1 : i8
    %add = comb.add %c1_i8, %c1_i8 : i8
    hw.instance "extern" @extern() -> ()
    %0 = hw.instance "child" @child() -> (out: i8)
    // CHECK: hw.output %c2_i8
    hw.output %add : i8
}

// CHECK-LABEL: hw.module @child
hw.module @child(out out: i8) {
    %c1_i8 = hw.constant 1 : i8
    %add = comb.add %c1_i8, %c1_i8 : i8
    %0 = hw.instance "bound" @bound() -> (out: i8) {doNotPrint}
    // CHECK: hw.output %c2_i8
    hw.output %add : i8
}

// CHECK-LABEL: hw.module @notIncluded
hw.module @notIncluded(out out: i8) {
    %c1_i8 = hw.constant 1 : i8
    %add = comb.add %c1_i8, %c1_i8 : i8
    // CHECK: %[[ADD:.+]] = comb.add %c1_i8, %c1_i8
    // CHECK-NEXT:  hw.output %[[ADD]] : i8
    hw.output %add : i8
}
