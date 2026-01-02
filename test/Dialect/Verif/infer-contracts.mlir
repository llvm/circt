// RUN: circt-opt --infer-contracts %s | FileCheck %s

// CHECK-LABEL: hw.module @Mul9
// CHECK-NEXT:   %c3_i42 = hw.constant 3 : i42
// CHECK-NEXT:   %c9_i42 = hw.constant 9 : i42
// CHECK-NEXT:   [[TMP0:%.+]] = comb.shl %a, %c3_i42 : i42    
// CHECK-NEXT:   [[TMP1:%.+]] = comb.add %a, [[TMP0]] : i42
// CHECK-NEXT:   [[TMP2:%.+]] = verif.contract [[TMP1]] : i42 {
// CHECK-NEXT:     [[TMP3:%.+]] = comb.mul %a, %c9_i42 : i42
// CHECK-NEXT:     [[TMP4:%.+]] = comb.icmp eq [[TMP2]], [[TMP3]] : i42
// CHECK-NEXT:     verif.ensure [[TMP4]] : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output [[TMP2]] : i42
// CHECK-NEXT: }

hw.module @Mul9(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42  
  %1 = comb.add %a, %0 : i42  
  %2 = comb.mul %a, %c9_i42 : i42
  %3 = comb.icmp eq %2, %1 : i42
  verif.assert %3
  hw.output %1
}

