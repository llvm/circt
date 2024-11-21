// RUN: circt-opt --lower-contracts %s | FileCheck %s


// CHECK: verif.formal @Mul9 {
// CHECK:   %c9_i42 = hw.constant 9 : i42
// CHECK:   %c3_i42 = hw.constant 3 : i42
// CHECK:   %0 = verif.symbolic_value : i42
// CHECK:   %1 = comb.shl %0, %c3_i42 : i42
// CHECK:   %2 = comb.add %0, %1 : i42
// CHECK:   %3 = comb.mul %0, %c9_i42 : i42
// CHECK:   %4 = comb.icmp eq %2, %3 : i42
// CHECK:   verif.assert %4 : i1
// CHECK: }

hw.module @Mul9(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42    // 8*a
  %1 = comb.add %a, %0 : i42         // a + 8*a
  %2 = verif.contract %1 : i42 {
    %3 = comb.mul %a, %c9_i42 : i42  // 9*a
    %4 = comb.icmp eq %2, %3 : i42  // 9*a == a + 8*a
    verif.ensure %4 : i1
  }
  hw.output %2 : i42
}

