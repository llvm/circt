// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @bitvectors
func.func @bitvectors() {
  // A bit-width divisible by 4 is always printed in hex
  // CHECK: %bv_x5a = smt.bv.constant <"#x5a"> {smt.some_attr}
  %bv_x5a = smt.bv.constant <"#b01011010"> {smt.some_attr}

  // A bit-width not divisible by 4 is always printed in binary
  // Also, make sure leading zeros are printed
  // CHECK: %bv_b0101101 = smt.bv.constant <"#b0101101"> {smt.some_attr}
  %bv_b0101101 = smt.bv.constant <"#b0101101"> {smt.some_attr}

  // CHECK: %bv_x3c = smt.bv.constant <"#x3c"> {smt.some_attr}
  %bv_x3c = smt.bv.constant <"#x3c"> {smt.some_attr}

  // Make sure leading zeros are printed
  // CHECK: %bv_x03c = smt.bv.constant <"#x03c"> {smt.some_attr}
  %bv_x03c = smt.bv.constant <"#x03c"> {smt.some_attr}

  // It is allowed to fully quantify the attribute including an explicit type
  // CHECK: %bv_x3cd = smt.bv.constant <"#x3cd">
  %bv_x3cd = smt.bv.constant #smt.bv<"#x3cd"> : !smt.bv<12>

  return
}
