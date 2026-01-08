// RUN: circt-opt %s --synth-lower-word-to-bits | FileCheck %s
// CHECK: hw.module @Basic
hw.module @Basic(in %a: i2, in %b: i2, out f: i2) {
  %0 = synth.aig.and_inv not %a, %b : i2
  %1 = synth.aig.and_inv not %0, not %0 : i2
  // CHECK-DAG: %[[EXTRACT_A_1:.+]] = comb.extract %a from 1
  // CHECK-DAG: %[[EXTRACT_B_1:.+]] = comb.extract %b from 1
  // CHECK-DAG: %[[AND_INV_0:.+]] = synth.aig.and_inv not %[[EXTRACT_A_1]], %[[EXTRACT_B_1]]
  // CHECK-DAG: %[[EXTRACT_A_0:.+]] = comb.extract %a from 0
  // CHECK-DAG: %[[EXTRACT_B_0:.+]] = comb.extract %b from 0
  // CHECK-DAG: %[[AND_INV_1:.+]] = synth.aig.and_inv not %[[EXTRACT_A_0]], %[[EXTRACT_B_0]]
  // CHECK-DAG: %[[AND_INV_2:.+]] = synth.aig.and_inv not %[[AND_INV_0]], not %[[AND_INV_0]]
  // CHECK-DAG: %[[AND_INV_3:.+]] = synth.aig.and_inv not %[[AND_INV_1]], not %[[AND_INV_1]]
  // CHECK-DAG: %[[CONCAT:.+]] = comb.concat %[[AND_INV_2]], %[[AND_INV_3]]
  hw.output %1 : i2
}

// CHECK-LABEL: hw.module @Basic_MIG
hw.module @Basic_MIG(in %a: i2, in %b: i2, in %c: i2, out f: i2) {
  %0 = synth.mig.maj_inv not %a, %b, %c : i2
  // CHECK-DAG: %[[EXTRACT_A_1:.+]] = comb.extract %a from 1 : (i2) -> i1
  // CHECK-DAG: %[[EXTRACT_B_1:.+]] = comb.extract %b from 1 : (i2) -> i1
  // CHECK-DAG: %[[EXTRACT_C_1:.+]] = comb.extract %c from 1 : (i2) -> i1
  // CHECK-DAG: %[[MAJ_INV_0:.+]] = synth.mig.maj_inv not %[[EXTRACT_A_1]], %[[EXTRACT_B_1]], %[[EXTRACT_C_1]] : i1
  // CHECK-DAG: %[[EXTRACT_A_0:.+]] = comb.extract %a from 0 : (i2) -> i1
  // CHECK-DAG: %[[EXTRACT_B_0:.+]] = comb.extract %b from 0 : (i2) -> i1
  // CHECK-DAG: %[[EXTRACT_C_0:.+]] = comb.extract %c from 0 : (i2) -> i1
  // CHECK-DAG: %[[MAJ_INV_1:.+]] = synth.mig.maj_inv not %[[EXTRACT_A_0]], %[[EXTRACT_B_0]], %[[EXTRACT_C_0]] : i1
  // CHECK-DAG: %[[CONCAT:.+]] = comb.concat %[[MAJ_INV_0]], %[[MAJ_INV_1]] : i1, i1
  // CHECK-DAG: hw.output %[[CONCAT]] : i2
  hw.output %0 : i2
}

// CHECK-LABEL: hw.module @Basic_Comb
hw.module @Basic_Comb(in %cond: i1, in %a: i2, in %b: i2, out f: i2, out g: i2, out h: i2, out i: i2) {
  %0 = comb.and %a, %b : i2
  %1 = comb.or %a, %b : i2
  %2 = comb.xor %a, %b : i2
  %3 = comb.mux %cond, %a, %b : i2
  // CHECK-DAG: %[[A0:.+]] = comb.extract %a from 0 : (i2) -> i1
  // CHECK-DAG: %[[A1:.+]] = comb.extract %a from 1 : (i2) -> i1
  // CHECK-DAG: %[[B0:.+]] = comb.extract %b from 0 : (i2) -> i1
  // CHECK-DAG: %[[B1:.+]] = comb.extract %b from 1 : (i2) -> i1
  // CHECK-DAG: %[[AND0:.+]] = comb.and %[[A0]], %[[B0]] : i1
  // CHECK-DAG: %[[AND1:.+]] = comb.and %[[A1]], %[[B1]] : i1
  // CHECK-DAG: %[[AND:.+]] = comb.concat %[[AND1]], %[[AND0]] : i1, i1
  // CHECK-DAG: %[[OR0:.+]] = comb.or %[[A0]], %[[B0]] : i1
  // CHECK-DAG: %[[OR1:.+]] = comb.or %[[A1]], %[[B1]] : i1
  // CHECK-DAG: %[[OR:.+]] = comb.concat %[[OR1]], %[[OR0]] : i1, i1
  // CHECK-DAG: %[[XOR0:.+]] = comb.xor %[[A0]], %[[B0]] : i1
  // CHECK-DAG: %[[XOR1:.+]] = comb.xor %[[A1]], %[[B1]] : i1
  // CHECK-DAG: %[[XOR:.+]] = comb.concat %[[XOR1]], %[[XOR0]] : i1, i1
  // CHECK-DAG: %[[MUX0:.+]] = comb.mux %cond, %[[A0]], %[[B0]] : i1
  // CHECK-DAG: %[[MUX1:.+]] = comb.mux %cond, %[[A1]], %[[B1]] : i1
  // CHECK-DAG: %[[MUX:.+]] = comb.concat %[[MUX1]], %[[MUX0]] : i1, i1
  // CHECK-DAG: hw.output %[[AND]], %[[OR]], %[[XOR]], %[[MUX]] : i2, i2, i2, i2
  hw.output %0, %1, %2, %3 : i2, i2, i2, i2
}

// CHECK-LABEL: hw.module @Constant_Propagation
hw.module @Constant_Propagation(in %a: i2, out f: i2) {
  // Create a scenario where some bits are known constants
  %c0_i2 = hw.constant 0 : i2
  %0 = synth.aig.and_inv %a, %c0_i2 : i2
  // CHECK-NOT: synth.aig.and_inv
  hw.output %0 : i2
}


// CHECK-LABEL: hw.module @Complex_Constant_Patterns
hw.module @Complex_Constant_Patterns(in %a: i2, in %b: i2, out f: i2) {
  // Test with mixed constant patterns
  %c1_i2 = hw.constant 1 : i2 // 0b0101
  %c2_i2 = hw.constant 2 : i2 // 0b1010
  %0 = synth.aig.and_inv %a, %c1_i2 : i2 // should keep bits 0 'a', bits 1 is zero
  %1 = synth.aig.and_inv not %b, not %c2_i2 : i2 // should keep bits 0 not 'b', bits 1 is zero
  %2 = synth.mig.maj_inv %c2_i2, %0, %1 : i2  // bits 1 is 0
  // CHECK-DAG: %false = hw.constant false
  // CHECK-DAG: %[[EXTRACT_A_0:.+]] = comb.extract %a from 0 : (i2) -> i1
  // CHECK-DAG: %[[EXTRACT_B_0:.+]] = comb.extract %b from 0 : (i2) -> i1
  // CHECK-DAG: %[[AND_INV_0:.+]] = synth.aig.and_inv not %[[EXTRACT_B_0]], not %false : i1
  // CHECK-DAG: %[[MAJ_INV_0:.+]] = synth.mig.maj_inv %false, %[[EXTRACT_A_0]], %[[AND_INV_0]] : i1
  // CHECK-DAG: %[[CONCAT:.+]] = comb.concat %false, %[[MAJ_INV_0]] : i1, i1
  // CHECK-DAG: hw.output %[[CONCAT]]
  hw.output %2 : i2
}

// CHECK-LABEL: hw.module @Namehint
hw.module @Namehint(in %a: i2, in %b: i2, out f: i2) {
  %0 = synth.aig.and_inv %a, %b {sv.namehint = "foo"}: i2
  // CHECK:      %[[AND_INV_0:.+]] = synth.aig.and_inv
  // CHECK-SAME:  {sv.namehint = "foo[0]"} : i1
  // CHECK-NEXT: %[[AND_INV_1:.+]] = synth.aig.and_inv
  // CHECK-SAME:  {sv.namehint = "foo[1]"} : i1
  hw.output %0 : i2
}
