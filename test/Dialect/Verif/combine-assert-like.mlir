// RUN: circt-opt --combine-assert-like %s | FileCheck %s

// CHECK-LABEL: hw.module @ManyAssumes
// CHECK-NEXT:   [[TMP0:%.+]] = comb.add bin %a, %b : i42
// CHECK-NEXT:   %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp bin uge %a, %c0_i42 : i42
// CHECK-NEXT:   [[TMP2:%.+]] = comb.icmp bin uge %b, %c0_i42 : i42
// CHECK-NEXT:   [[TMP3:%.+]] = comb.and [[TMP1]], [[TMP2]] : i1
// CHECK-NEXT:   verif.assume [[TMP3]] : i1
// CHECK-NEXT:   hw.output [[TMP0]] : i42
// CHECK-NEXT: }

hw.module @ManyAssumes(in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add bin %a, %b : i42
  %c0_i42 = hw.constant 0 : i42
  %1 = comb.icmp bin uge %a, %c0_i42 : i42
  %2 = comb.icmp bin uge %b, %c0_i42 : i42
  verif.assume %1 : i1
  verif.assume %2 : i1
  hw.output %0 : i42
}

//------

// CHECK-LABEL: hw.module @ManyAssertsAndAssumes
// CHECK-NEXT:   %c1_i42 = hw.constant 1 : i42
// CHECK-NEXT:   [[TMP0:%.+]] = comb.shl %a, %c1_i42 : i42
// CHECK-NEXT:   %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp ult %a, %c2_i42 : i42
// CHECK-NEXT:   %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:   [[TMP2:%.+]] = comb.icmp uge %a, %c0_i42 : i42
// CHECK-NEXT:   [[REQ:%.+]] = comb.and [[TMP1]], [[TMP2]] : i1
// CHECK-NEXT:   verif.assume [[REQ]] : i1
// CHECK-NEXT:   [[TMP3:%.+]] = comb.mul %a, %c2_i42 : i42
// CHECK-NEXT:   [[TMP4:%.+]] = comb.icmp eq [[TMP0]], [[TMP3]] : i42
// CHECK-NEXT:   [[TMP5:%.+]] = comb.add %a, %a : i42
// CHECK-NEXT:   [[TMP6:%.+]] = comb.icmp eq [[TMP0]], [[TMP5]] : i42
// CHECK-NEXT:   [[ENS:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1
// CHECK-NEXT:   verif.assert [[ENS]] : i1
// CHECK-NEXT:   hw.output [[TMP0]] : i42
// CHECK-NEXT: }

hw.module @ManyAssertsAndAssumes(in %a: i42, out z: i42) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.shl %a, %c1_i42 : i42
  %c2_i42 = hw.constant 2 : i42
  %1 = comb.icmp ult %a, %c2_i42 : i42
  %c0_i42 = hw.constant 0 : i42
  %2 = comb.icmp uge %a, %c0_i42 : i42
  verif.assume %1 : i1
  verif.assume %2 : i1
  %3 = comb.mul %a, %c2_i42 : i42
  %4 = comb.icmp eq %0, %3 : i42
  %5 = comb.add %a, %a : i42
  %6 = comb.icmp eq %0, %5 : i42
  verif.assert %4 : i1
  verif.assert %6 : i1
  hw.output %0 : i42
}
