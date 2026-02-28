// RUN: circt-opt --cse %s | FileCheck %s

// CHECK-LABEL: @SymbolicValuesMustNotCSE
hw.module @SymbolicValuesMustNotCSE(out a: i32, out b: i32) {
  // CHECK: [[TMP1:%.+]] = verif.symbolic_value
  // CHECK: [[TMP2:%.+]] = verif.symbolic_value
  // CHECK: hw.output [[TMP1]], [[TMP2]]
  %0 = verif.symbolic_value : i32
  %1 = verif.symbolic_value : i32
  hw.output %0, %1 : i32, i32
}
