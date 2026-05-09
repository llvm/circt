// RUN: circt-opt --pass-pipeline='builtin.module(any(fold-assume))' --split-input-file %s | FileCheck %s

// CHECK-LABEL: hw.module @basic
// CHECK-NEXT:   [[TMP0:%.+]] = comb.add bin %a, %b : i42
// CHECK-NEXT:   %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp bin uge %a, %c0_i42 : i42
// CHECK-NEXT:   [[TMP2:%.+]] = comb.icmp bin uge %b, %c0_i42 : i42
// CHECK-NEXT:   verif.assert [[TMP2]] if [[TMP1]] : i1
// CHECK-NEXT:   hw.output [[TMP0]] : i42
// CHECK-NEXT: }

hw.module @basic(in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add bin %a, %b : i42
  %c0_i42 = hw.constant 0 : i42
  %1 = comb.icmp bin uge %a, %c0_i42 : i42
  %2 = comb.icmp bin uge %b, %c0_i42 : i42
  verif.assume %1 : i1
  verif.assert %2 : i1
  hw.output %0 : i42
}

// -----

// CHECK-LABEL: verif.formal @formal
// CHECK-NEXT:   %a = verif.symbolic_value : i42
// CHECK-NEXT:   %b = verif.symbolic_value : i42
// CHECK-NEXT:   [[TMP0:%.+]] = comb.add bin %a, %b : i42
// CHECK-NEXT:   %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp bin uge %a, %c0_i42 : i42
// CHECK-NEXT:   [[TMP2:%.+]] = comb.icmp bin uge %b, %c0_i42 : i42
// CHECK-NEXT:   verif.assert [[TMP2]] if [[TMP1]] : i1
// CHECK-NEXT: }

verif.formal @formal {} {
  %a = verif.symbolic_value : i42
  %b = verif.symbolic_value : i42
  %0 = comb.add bin %a, %b : i42
  %c0_i42 = hw.constant 0 : i42
  %1 = comb.icmp bin uge %a, %c0_i42 : i42
  %2 = comb.icmp bin uge %b, %c0_i42 : i42
  verif.assume %1 : i1
  verif.assert %2 : i1
}

// -----

// CHECK-LABEL: hw.module @noAssert
// CHECK-NEXT:   [[TMP0:%.+]] = comb.add bin %a, %b : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp bin uge %a, %b : i42
// CHECK-NEXT:   [[C1:%.+]] = hw.constant true
// CHECK-NEXT:   verif.assert [[C1]] if [[TMP1]] : i1
// CHECK-NEXT:   hw.output [[TMP0]] : i42
// CHECK-NEXT: }

hw.module @noAssert(in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add bin %a, %b : i42
  %1 = comb.icmp bin uge %a, %b : i42
  verif.assume %1 : i1
  hw.output %0 : i42
}

// -----

// CHECK-LABEL: hw.module @enable
// CHECK-NEXT:   [[TMP0:%.+]] = comb.add bin %a, %b : i42
// CHECK-NEXT:   %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp bin eq %a, %b : i42
// CHECK-NEXT:   [[TMP2:%.+]] = comb.icmp bin uge %b, %c0_i42 : i42
// CHECK-NEXT:   [[AND:%.+]] = comb.and %en, [[TMP1]] : i1
// CHECK-NEXT:   verif.assert [[TMP2]] if [[AND]] : i1
// CHECK-NEXT:   hw.output [[TMP0]] : i42
// CHECK-NEXT: }

hw.module @enable(in %a: i42, in %b: i42, in %en: i1, out z: i42) {
  %0 = comb.add bin %a, %b : i42
  %c0_i42 = hw.constant 0 : i42
  %1 = comb.icmp bin eq %a, %b : i42
  %2 = comb.icmp bin uge %b, %c0_i42 : i42
  verif.assume %1 : i1
  verif.assert %2 if %en : i1
  hw.output %0 : i42
}

//------

// CHECK-LABEL: hw.module @IgnoreLTL
// CHECK-NEXT:   [[TMP0:%.+]] = ltl.clock %prop, posedge %a : !ltl.property
// CHECK-NEXT:   verif.assume [[TMP0]] : !ltl.property
// CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp ugt %a, %b : i1
// CHECK-NEXT:   verif.assert [[TMP1]] : i1
// CHECK-NEXT:   hw.output [[TMP1]] : i1
// CHECK-NEXT: }

hw.module @IgnoreLTL(in %a: i1, in %b: i1, in %prop: !ltl.property, out z: i1) {
  %0 = ltl.clock %prop, posedge %a : !ltl.property
  verif.assume %0 : !ltl.property
  %1 = comb.icmp ugt %a, %b : i1
  verif.assert %1 : i1
  hw.output %1 : i1
}

//------

sv.macro.decl @SYNTHESIS

// CHECK-LABEL: hw.module @ManyBlocks
// CHECK-NEXT:   %c1_i42 = hw.constant 1 : i42
// CHECK-NEXT:   [[TMP0:%.+]] = comb.shl %a, %c1_i42 : i42
// CHECK-NEXT:   sv.always posedge %clk {
// CHECK-NEXT:     sv.ifdef.procedural @SYNTHESIS {
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:       [[TMP1:%.+]] = comb.icmp ult %a, %c2_i42 : i42
// CHECK-NEXT:       %c0_i42 = hw.constant 0 : i42
// CHECK-NEXT:       [[TMP2:%.+]] = comb.icmp uge %a, %c0_i42 : i42
// CHECK-NEXT:       sv.if %en_assume {
// CHECK-NEXT:         verif.assert [[TMP2]] if [[TMP1]] : i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         verif.assert [[TMP1]] if [[TMP2]] : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       [[TMP3:%.+]] = comb.mul %a, %c2_i42 : i42
// CHECK-NEXT:       [[TMP4:%.+]] = comb.icmp eq [[TMP0]], [[TMP3]] : i42
// CHECK-NEXT:       [[TMP5:%.+]] = comb.add %a, %a : i42
// CHECK-NEXT:       [[TMP6:%.+]] = comb.icmp eq [[TMP0]], [[TMP5]] : i42
// CHECK-NEXT:       sv.if %en_assert {
// CHECK-NEXT:         verif.assert [[TMP6]] if [[TMP4]] : i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         verif.assert [[TMP4]] if [[TMP6]] : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       [[COND:%.+]] = comb.or %en_assert, %en_assume : i1
// CHECK-NEXT:       verif.assert [[COND]] : i1
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output [[TMP0]] : i42
// CHECK-NEXT:  }

hw.module @ManyBlocks(in %clk: i1, in %a: i42, in %en_assume: i1, in %en_assert: i1, out z: i42) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.shl %a, %c1_i42 : i42
  sv.always posedge %clk {
    sv.ifdef.procedural @SYNTHESIS {
    } else {
      %c2_i42 = hw.constant 2 : i42
      %1 = comb.icmp ult %a, %c2_i42 : i42
      %c0_i42 = hw.constant 0 : i42
      %2 = comb.icmp uge %a, %c0_i42 : i42
      sv.if %en_assume {
        verif.assume %1 : i1
        verif.assert %2 : i1
      } else {
        verif.assume %2 : i1
        verif.assert %1 : i1
      }
      %3 = comb.mul %a, %c2_i42 : i42
      %4 = comb.icmp eq %0, %3 : i42
      %5 = comb.add %a, %a : i42
      %6 = comb.icmp eq %0, %5 : i42
      sv.if %en_assert {
        verif.assume %4 : i1
        verif.assert %6 : i1
      } else {
        verif.assume %6 : i1
        verif.assert %4 : i1
      }
      %cond = comb.or %en_assert, %en_assume : i1
      verif.assert %cond : i1
    }
  }
  hw.output %0 : i42
}


