// RUN: circt-opt %s --arc-inline | FileCheck %s

// CHECK-LABEL: func.func @Simple
func.func @Simple(%arg0: i4, %arg1: i1) -> (i4, i4) {
  // CHECK-NEXT: %0 = comb.and %arg0, %arg0
  // CHECK-NEXT: %1 = arc.state @SimpleB(%arg0) clock %arg1 lat 1
  // CHECK-NEXT: return %0, %1
  %0 = arc.state @SimpleA(%arg0) lat 0 : (i4) -> i4
  %1 = arc.state @SimpleB(%arg0) clock %arg1 lat 1 : (i4) -> i4
  return %0, %1 : i4, i4
}
// CHECK-NEXT:  }
// CHECK-NOT: arc.define @SimpleA
arc.define @SimpleA(%arg0: i4) -> i4 {
  %0 = comb.and %arg0, %arg0 : i4
  arc.output %0 : i4
}
// CHECK-LABEL: arc.define @SimpleB
arc.define @SimpleB(%arg0: i4) -> i4 {
  %0 = comb.xor %arg0, %arg0 : i4
  arc.output %0 : i4
}


hw.module @nestedRegionTest(%arg0: i4, %arg1: i4) -> (out0: i4) {
  %0 = arc.state @sub3(%arg0, %arg1) lat 0 : (i4, i4) -> i4
  hw.output %0 : i4
}

arc.define @sub3(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.extract %arg0 from 2 : (i4) -> i1
  %1 = scf.if %0 -> (i4) {
    %2 = comb.xor bin %arg0, %arg1 : i4
    scf.yield %2 : i4
  } else {
    %2 = comb.and bin %arg0, %arg1 : i4
    scf.yield %2 : i4
  }
  arc.output %1 : i4
}

// CHECK-LABEL: hw.module @nestedRegionTest
// CHECK-NEXT: [[EXT:%.+]] = comb.extract %arg0 from 2 : (i4) -> i1
// CHECK-NEXT: [[IFRES:%.+]] = scf.if [[EXT]] -> (i4) {
// CHECK-NEXT:   [[XOR:%.+]] = comb.xor bin %arg0, %arg1 : i4
// CHECK-NEXT:   scf.yield [[XOR]] : i4
// CHECK-NEXT: } else {
// CHECK-NEXT:   [[AND:%.+]] = comb.and bin %arg0, %arg1 : i4
// CHECK-NEXT:   scf.yield [[AND]] : i4
// CHECK-NEXT: }
// CHECK-NEXT: hw.output [[IFRES]] : i4

hw.module @opsInNestedRegionsAreAlsoCounted(%arg0: i4, %arg1: i4) -> (out0: i4, out1: i4) {
  %0 = arc.state @sub4(%arg0, %arg1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @sub4(%arg0, %arg1) lat 0 : (i4, i4) -> i4
  hw.output %0, %1 : i4, i4
}

arc.define @sub4(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.extract %arg0 from 2 : (i4) -> i1
  %1 = scf.if %0 -> (i4) {
    %2 = comb.xor bin %arg0, %arg1 : i4
    %3 = comb.and bin %2, %arg1 : i4
    scf.yield %3 : i4
  } else {
    %2 = comb.and bin %arg0, %arg1 : i4
    %3 = comb.or bin %arg0, %2 : i4
    scf.yield %3 : i4
  }
  arc.output %1 : i4
}

// CHECK-LABEL: hw.module @opsInNestedRegionsAreAlsoCounted
// CHECK-NEXT:   [[STATERES1:%.+]] = arc.state @sub4(%arg0, %arg1) lat 0 : (i4, i4) -> i4
// CHECK-NEXT:   [[STATERES2:%.+]] = arc.state @sub4(%arg0, %arg1) lat 0 : (i4, i4) -> i4
// CHECK-NEXT:   hw.output [[STATERES1]], [[STATERES2]] : i4, i4

hw.module @nestedBlockArgumentsTest(%arg0: index, %arg1: i4) -> (out0: i4) {
  %0 = arc.state @sub5(%arg0, %arg1) lat 0 : (index, i4) -> i4
  hw.output %0 : i4
}

arc.define @sub5(%arg0: index, %arg1: i4) -> i4 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %iv = %c0 to %arg0 step %c1 iter_args (%i = %arg1) -> (i4) {
    %1 = comb.add %i, %arg1 : i4
    scf.yield %1 : i4
  }
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @nestedBlockArgumentsTest
// CHECK:      [[RES:%.+]] = scf.for {{%.+}} = %c0 to %arg0 step %c1 iter_args([[A0:%.+]] = %arg1) -> (i4) {
// CHECK-NEXT:   [[SUM:%.+]] = comb.add [[A0]], %arg1 : i4
// CHECK-NEXT:   scf.yield [[SUM]] : i4
// CHECK-NEXT: }
// CHECK-NEXT: hw.output [[RES]] : i4
