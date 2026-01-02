// RUN: circt-opt --infer-contracts %s | FileCheck %s

// CHECK-LABEL: hw.module @AssertOnly
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

hw.module @AssertOnly(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42  
  %1 = comb.add %a, %0 : i42  
  %2 = comb.mul %a, %c9_i42 : i42
  %3 = comb.icmp eq %2, %1 : i42
  verif.assert %3
  hw.output %1
}

//------

// CHECK-LABEL: hw.module @AssertAssume
// CHECK-NEXT:   %c4_i8 = hw.constant 4 : i8
// CHECK-NEXT:   %c2_i8 = hw.constant 2 : i8
// CHECK-NEXT:   %c1_i8 = hw.constant 1 : i8
// CHECK-NEXT:   %b2 = comb.extract %b from 2 : (i8) -> i1
// CHECK-NEXT:   %b1 = comb.extract %b from 1 : (i8) -> i1
// CHECK-NEXT:   %b0 = comb.extract %b from 0 : (i8) -> i1
// CHECK-NEXT:   [[TMP0:%.+]] = comb.shl %a, %c4_i8 : i8
// CHECK-NEXT:   [[TMP1:%.+]] = comb.mux %b2, [[TMP0]], %a : i8
// CHECK-NEXT:   [[TMP2:%.+]] = comb.shl [[TMP1]], %c2_i8 : i8
// CHECK-NEXT:   [[TMP3:%.+]] = comb.mux %b1, [[TMP2]], [[TMP1]] : i8
// CHECK-NEXT:   [[TMP4:%.+]] = comb.shl [[TMP3]], %c1_i8 : i8
// CHECK-NEXT:   [[TMP5:%.+]] = comb.mux %b0, [[TMP4]], [[TMP3]] : i8
// CHECK-NEXT:   [[CONT:%.+]] = verif.contract [[TMP5]] : i8 {
// CHECK-NEXT:     %c8_i8 = hw.constant 8 : i8
// CHECK-NEXT:     [[TMP6:%.+]] = comb.icmp ult %b, %c8_i8 : i8
// CHECK-NEXT:     verif.require [[TMP6]] : i1
// CHECK-NEXT:     [[TMP7:%.+]] = comb.shl %a, %b : i8
// CHECK-NEXT:     [[TMP8:%.+]] = comb.icmp eq [[CONT]], [[TMP7]] : i8
// CHECK-NEXT:     verif.ensure [[TMP8]] : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output [[CONT]] : i8
// CHECK-NEXT: }   

hw.module @AssertAssume(in %a: i8, in %b: i8, out z: i8) {
  %c4_i8 = hw.constant 4 : i8
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %b2 = comb.extract %b from 2 : (i8) -> i1
  %b1 = comb.extract %b from 1 : (i8) -> i1
  %b0 = comb.extract %b from 0 : (i8) -> i1
  %0 = comb.shl %a, %c4_i8 : i8
  %1 = comb.mux %b2, %0, %a : i8
  %2 = comb.shl %1, %c2_i8 : i8
  %3 = comb.mux %b1, %2, %1 : i8
  %4 = comb.shl %3, %c1_i8 : i8
  %5 = comb.mux %b0, %4, %3 : i8
  %c8_i8 = hw.constant 8 : i8
  %6 = comb.icmp ult %b, %c8_i8 : i8
  verif.assume %6 : i1
  %7 = comb.shl %a, %b : i8
  %8 = comb.icmp eq %5, %7 : i8
  verif.assert %8 : i1
  hw.output %5 : i8
}

//------

// CHECK-LABEL: hw.module @AssumeOnly
// CHECK-NEXT:   [[TMP0:%.+]] = comb.add bin %a, %b : i42
// CHECK-NEXT:   [[CONT:%.+]] = verif.contract [[TMP0]] : i42 {
// CHECK-NEXT:     %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:     [[TMP1:%.+]] = comb.icmp bin uge %a, %c0_i42 : i1
// CHECK-NEXT:     verif.require [[TMP1]] : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output [[CONT]] : i42
// CHECK-NEXT: }

hw.module @AssumeOnly(in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add bin %a, %b : i42
  %c0_i42 = hw.constant 0 : i42
  %1 = comb.icmp bin uge %a, %c0_i42 : i1
  verif.assume %1 : i1
  hw.output %0 : i42
}

//------

// CHECK-LABEL: hw.module @ManyAssumes
// CHECK-NEXT:   [[TMP0:%.+]] = comb.add bin %a, %b : i42
// CHECK-NEXT:   [[CONT:%.+]] = verif.contract [[TMP0]] : i42 {
// CHECK-NEXT:     %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:     [[TMP1:%.+]] = comb.icmp bin uge %a, %c0_i42 : i1
// CHECK-NEXT:     [[TMP2:%.+]] = comb.icmp bin uge %b, %c0_i42 : i1
// CHECK-NEXT:     [[TMP3:%.+]] = comb.and bin [[TMP1]], [[TMP2]] : i1
// CHECK-NEXT:     verif.require [[TMP3]] : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output [[CONT]] : i42
// CHECK-NEXT: }

hw.module @ManyAssumes(in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add bin %a, %b : i42
  %c0_i42 = hw.constant 0 : i42
  %1 = comb.icmp bin uge %a, %c0_i42 : i1
  verif.assume %1 : i1
  %2 = comb.icmp bin uge %b, %c0_i42 : i1
  verif.assume %2 : i1
  hw.output %0 : i42
}

//------

hw.module @ManyAssertsAndAssumes(in %a: i42, out z: i42) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.shl %a, %c1_i42 : i42
  %1 = verif.contract %0 : i42 {
    %c2_i42 = hw.constant 2 : i42
    %req = comb.icmp ult %a, %c2_i42 : i42
    verif.require %req : i1
    %c0_i42 = hw.constant 0 : i42
    %req1 = comb.icmp uge %a, %c0_i42 : i1
    verif.require %req1
    %2 = comb.mul %a, %c2_i42 : i42
    %3 = comb.icmp eq %1, %2 : i42
    verif.ensure %3 : i1
    %5 = comb.add %a, %a : i42
    %6 = comb.icmp eq %1, %5 : i42
    verif.ensure %6 : i1
  }
  hw.output %1 : i42
}
