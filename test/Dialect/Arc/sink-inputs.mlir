// RUN: circt-opt %s --arc-sink-inputs | FileCheck %s

// CHECK-LABEL: arc.define @SinkSameConstantsArc(%arg0: i4)
arc.define @SinkSameConstantsArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: %c2_i4 = hw.constant 2
  // CHECK-NEXT: [[TMP:%.+]] = comb.add %arg0, %c2_i4
  // CHECK-NEXT: arc.output [[TMP]]
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK: arc.define @Foo
arc.define @Foo(%arg0: i4) -> i4 {
  // CHECK-NOT: hw.constant
  %k1 = hw.constant 2 : i4
  // CHECK: {{%.+}} = arc.call @SinkSameConstantsArc(%arg0)
  %0 = arc.call @SinkSameConstantsArc(%arg0, %k1) : (i4, i4) -> i4
  arc.output %0 : i4
}

// CHECK: hw.module @SinkSameConstants
hw.module @SinkSameConstants(%x: i4) {
  // CHECK-NOT: hw.constant
  // CHECK-NEXT: %0 = arc.state @SinkSameConstantsArc(%x)
  // CHECK-NEXT: %1 = arc.state @SinkSameConstantsArc(%x)
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.state @SinkSameConstantsArc(%x, %k1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @SinkSameConstantsArc(%x, %k2) lat 0 : (i4, i4) -> i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @DontSinkDifferentConstantsArc(%arg0: i4, %arg1: i4)
arc.define @DontSinkDifferentConstantsArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: comb.add %arg0, %arg1
  // CHECK-NEXT: arc.output
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-LABEL: hw.module @DontSinkDifferentConstants
hw.module @DontSinkDifferentConstants(%x: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %c3_i4 = hw.constant 3 : i4
  // CHECK-NEXT: %0 = arc.state @DontSinkDifferentConstantsArc(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.state @DontSinkDifferentConstantsArc(%x, %c3_i4)
  // CHECK-NEXT: hw.output
  %c2_i4 = hw.constant 2 : i4
  %c3_i4 = hw.constant 3 : i4
  %0 = arc.state @DontSinkDifferentConstantsArc(%x, %c2_i4) lat 0 : (i4, i4) -> i4
  %1 = arc.state @DontSinkDifferentConstantsArc(%x, %c3_i4) lat 0 : (i4, i4) -> i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @DontSinkDifferentConstantsArc1(%arg0: i4, %arg1: i4)
arc.define @DontSinkDifferentConstantsArc1(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: [[TMP:%.+]] = comb.add %arg0, %arg1
  // CHECK-NEXT: arc.output [[TMP]]
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK: arc.define @Bar
arc.define @Bar(%arg0: i4) -> i4 {
  // CHECK: %c1_i4 = hw.constant 1
  %k1 = hw.constant 1 : i4
  // CHECK: {{%.+}} = arc.call @DontSinkDifferentConstantsArc1(%arg0, %c1_i4)
  %0 = arc.call @DontSinkDifferentConstantsArc1(%arg0, %k1) : (i4, i4) -> i4
  arc.output %0 : i4
}

// CHECK: hw.module @DontSinkDifferentConstants1
hw.module @DontSinkDifferentConstants1(%x: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %c2_i4_0 = hw.constant 2 : i4
  // CHECK-NEXT: %0 = arc.state @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.state @DontSinkDifferentConstantsArc1(%x, %c2_i4_0)
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.state @DontSinkDifferentConstantsArc1(%x, %k1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @DontSinkDifferentConstantsArc1(%x, %k2) lat 0 : (i4, i4) -> i4
}
// CHECK-NEXT: }
