// RUN: circt-opt %s --synth-maximum-and-cover | FileCheck %s

// CHECK-LABEL: @SingleFanoutCollapse
hw.module @SingleFanoutCollapse(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out o1: i1) {
  // CHECK-NEXT: %[[AND:.+]] = synth.aig.and_inv %a, %b, %c, %d : i1
  // CHECK-NEXT: hw.output %[[AND]] : i1
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c, %d : i1
  hw.output %1 : i1
}

// CHECK-LABEL: @MultiFanoutNoCollapse
hw.module @MultiFanoutNoCollapse(in %a: i1, in %b: i1, in %c: i1, out o1: i1, out o2: i1) {
  // CHECK-NEXT: %[[AND0:.+]] = synth.aig.and_inv %a, %b : i1
  // CHECK-NEXT: %[[AND1:.+]] = synth.aig.and_inv %[[AND0]], %c : i1
  // CHECK-NEXT: hw.output %[[AND0]], %[[AND1]] : i1, i1
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c : i1
  hw.output %0, %1 : i1, i1
}

// CHECK-LABEL: @InvertedNoCollapse
hw.module @InvertedNoCollapse(in %a: i1, in %b: i1, in %c: i1, out o1: i1) {
  // CHECK-NEXT: %[[AND0:.+]] = synth.aig.and_inv %a, %b : i1
  // CHECK-NEXT: %[[AND1:.+]] = synth.aig.and_inv not %[[AND0]], %c : i1
  // CHECK-NEXT: hw.output %[[AND1]] : i1
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0, %c : i1
  hw.output %1 : i1
}

// CHECK-LABEL: @ComplexTree
hw.module @ComplexTree(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1, in %g: i1, out o1: i1) {
  // CHECK-NEXT: %[[AND0:.+]] = synth.aig.and_inv %d, not %e : i1
  // CHECK-NEXT: %[[AND1:.+]] = synth.aig.and_inv not %c, not %[[AND0]], %f : i1
  // CHECK-NEXT: %[[AND2:.+]] = synth.aig.and_inv %a, not %b, not %[[AND1]], %g : i1
  // CHECK-NEXT: hw.output %[[AND2]] : i1
  
  %1 = synth.aig.and_inv %a, not %b : i1
  %2 = synth.aig.and_inv %d, not %e : i1
  %3 = synth.aig.and_inv not %2, %f : i1
  %4 = synth.aig.and_inv not %c, %3 : i1
  %5 = synth.aig.and_inv %1, not %4 : i1
  %6 = synth.aig.and_inv %5, %g : i1

  hw.output %6 : i1
}
