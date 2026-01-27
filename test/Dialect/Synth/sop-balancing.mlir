// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-sop-balancing{max-cut-input-size=6}))' %s | FileCheck %s


// Chain of ANDs gets balanced into a tree

// CHECK-LABEL: hw.module @and_chain
hw.module @and_chain(in %a : i1, in %b : i1, in %c : i1, in %d : i1, out result : i1) {
    // Linear chain that should be balanced
    // Original: ((a & b) & c) & d - depth 3
    // Balanced: (a & b) & (c & d) - depth 2
    %and0 = synth.aig.and_inv %a, %b : i1
    %and1 = synth.aig.and_inv %and0, %c : i1
    %and2 = synth.aig.and_inv %and1, %d : i1
    
    // CHECK-DAG: %[[LEFT:.*]] = synth.aig.and_inv %a, %b
    // CHECK-DAG: %[[RIGHT:.*]] = synth.aig.and_inv %c, %d
    // CHECK: %[[RESULT:.*]] = synth.aig.and_inv %[[LEFT]], %[[RIGHT]]
    // CHECK: hw.output %[[RESULT]]
    hw.output %and2 : i1
}

// Example from ICCAD paper Sec 3.2.
// CHECK-LABEL: hw.module @balance
hw.module @balance(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1, out o1: i1) {
    // Original: ab + c(d + ef) - depth 4
    // Balanced: (ab + cd) + cef - depth 3
    %0 = synth.aig.and_inv %a, %b : i1
    %1 = synth.aig.and_inv %e, %f : i1
    %2 = synth.aig.and_inv not %d, not %1 : i1
    %3 = synth.aig.and_inv %c, not %2 : i1
    %4 = synth.aig.and_inv not %0, not %3 : i1
    %5 = synth.aig.and_inv not %4 : i1
    // CHECK-DAG: %[[AB:.*]] = synth.aig.and_inv %a, %b
    // CHECK-DAG: %[[EF:.*]] = synth.aig.and_inv %e, %f
    // CHECK-DAG: %[[CEF:.*]] = synth.aig.and_inv %[[EF]], %c
    // CHECK-DAG: %[[CD:.*]] = synth.aig.and_inv %d, %c
    // CHECK-DAG: %[[RESULT1:.*]] = synth.aig.and_inv not %[[AB]], not %[[CD]]
    // CHECK-DAG: %[[RESULT2:.*]] = synth.aig.and_inv not %[[CEF]], %[[RESULT1]]
    // CHECK-DAG: %[[FINAL:.*]] = synth.aig.and_inv not %[[RESULT2]]
    // CHECK: hw.output %[[FINAL]]
    hw.output %5 : i1
}

