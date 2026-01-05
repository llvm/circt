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

//------

// CHECK-LABEL: hw.module @ManyAssertsAndAssumes
// CHECK-NEXT:   %c1_i42 = hw.constant 1 : i42
// CHECK-NEXT:   [[TMP0:%.+]] = comb.shl %a, %c1_i42 : i42
// CHECK-NEXT:   %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp ult %a, %c2_i42 : i42
// CHECK-NEXT:   [[EN0:%.+]] = comb.and [[TMP1]], %en : i1
// CHECK-NEXT:   %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:   [[TMP2:%.+]] = comb.icmp uge %a, %c0_i42 : i42
// CHECK-NEXT:   [[EN1:%.+]] = comb.and [[TMP2]], %en : i1
// CHECK-NEXT:   [[REQ:%.+]] = comb.and [[EN0]], [[EN1]] : i1
// CHECK-NEXT:   verif.assume [[REQ]] : i1
// CHECK-NEXT:   [[TMP3:%.+]] = comb.mul %a, %c2_i42 : i42
// CHECK-NEXT:   [[TMP4:%.+]] = comb.icmp eq [[TMP0]], [[TMP3]] : i42
// CHECK-NEXT:   [[EN2:%.+]] = comb.and [[TMP4]], %en : i1
// CHECK-NEXT:   [[TMP5:%.+]] = comb.add %a, %a : i42
// CHECK-NEXT:   [[TMP6:%.+]] = comb.icmp eq [[TMP0]], [[TMP5]] : i42
// CHECK-NEXT:   [[EN3:%.+]] = comb.and [[TMP6]], %en : i1
// CHECK-NEXT:   [[ENS:%.+]] = comb.and [[EN2]], [[EN3]] : i1
// CHECK-NEXT:   verif.assert [[ENS]] : i1
// CHECK-NEXT:   hw.output [[TMP0]] : i42
// CHECK-NEXT: }

hw.module @ManyAssertsAndAssumesWithEnable(in %a: i42, in %en: i1, out z: i42) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.shl %a, %c1_i42 : i42
  %c2_i42 = hw.constant 2 : i42
  %1 = comb.icmp ult %a, %c2_i42 : i42
  verif.assume %1 if %en : i1
  %c0_i42 = hw.constant 0 : i42
  %2 = comb.icmp uge %a, %c0_i42 : i42
  verif.assume %2 if %en: i1
  %3 = comb.mul %a, %c2_i42 : i42
  %4 = comb.icmp eq %0, %3 : i42
  verif.assert %4 if %en : i1
  %5 = comb.add %a, %a : i42
  %6 = comb.icmp eq %0, %5 : i42
  verif.assert %6 if %en : i1
  hw.output %0 : i42
}

//------

// CHECK-LABEL: hw.module @IgnoreLTL
// CHECK-NEXT:   [[TMP0:%.+]] = ltl.clock %prop, posedge %a : !ltl.property
// CHECK-NEXT:   verif.assert [[TMP0]] : !ltl.property
// CHECK-NEXT:   [[TMP1:%.+]] = comb.and %b, %a : i1
// CHECK-NEXT:   verif.assert [[TMP1]] : i1
// CHECK-NEXT: }

hw.module @IgnoreLTL(in %a: i1, in %b: i1, in %prop: !ltl.property, out z: i1) {
  %0 = ltl.clock %prop, posedge %a : !ltl.property
  verif.assert %0 : !ltl.property
  %1 = comb.and %a, %b : i1
  verif.assert %1 : i1
  hw.output %1 : i1
}

