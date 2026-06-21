// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

module {
  // CHECK-LABEL: func.func @DynExtractPackedStructBit
  // CHECK-SAME: (%[[INPUT:.+]]: !hw.struct<a: i3, b: i2>, %[[IDX:.+]]: i3) -> i1
  func.func @DynExtractPackedStructBit(
    %input: !moore.struct<{a: i3, b: i2}>,
    %idx: !moore.i3
  ) -> !moore.i1 {
    // CHECK: %[[BITS:.+]] = hw.bitcast %[[INPUT]] : (!hw.struct<a: i3, b: i2>) -> i5
    // CHECK: %[[AMOUNT:.+]] = comb.concat {{%.+}}, %[[IDX]] : i2, i3
    // CHECK: %[[SHIFTED:.+]] = comb.shru %[[BITS]], %[[AMOUNT]] : i5
    // CHECK: %[[RESULT:.+]] = comb.extract %[[SHIFTED]] from 0 : (i5) -> i1
    // CHECK: return %[[RESULT]] : i1
    %result = moore.dyn_extract %input from %idx : !moore.struct<{a: i3, b: i2}>, !moore.i3 -> !moore.i1
    return %result : !moore.i1
  }

  // CHECK-LABEL: func.func @DynExtractPackedStructSlice
  // CHECK-SAME: (%[[INPUT:.+]]: !hw.struct<a: i3, b: i2>, %[[IDX:.+]]: i3) -> i2
  func.func @DynExtractPackedStructSlice(
    %input: !moore.struct<{a: i3, b: i2}>,
    %idx: !moore.i3
  ) -> !moore.i2 {
    // CHECK: %[[BITS:.+]] = hw.bitcast %[[INPUT]] : (!hw.struct<a: i3, b: i2>) -> i5
    // CHECK: %[[AMOUNT:.+]] = comb.concat {{%.+}}, %[[IDX]] : i2, i3
    // CHECK: %[[SHIFTED:.+]] = comb.shru %[[BITS]], %[[AMOUNT]] : i5
    // CHECK: %[[RESULT:.+]] = comb.extract %[[SHIFTED]] from 0 : (i5) -> i2
    // CHECK: return %[[RESULT]] : i2
    %result = moore.dyn_extract %input from %idx : !moore.struct<{a: i3, b: i2}>, !moore.i3 -> !moore.i2
    return %result : !moore.i2
  }

  // CHECK-LABEL: func.func @DynExtractPackedStructToStruct
  // CHECK-SAME: (%[[INPUT:.+]]: !hw.struct<a: i3, b: i2>, %[[IDX:.+]]: i3) -> !hw.struct<b: i2>
  func.func @DynExtractPackedStructToStruct(
    %input: !moore.struct<{a: i3, b: i2}>,
    %idx: !moore.i3
  ) -> !moore.struct<{b: i2}> {
    // CHECK: %[[BITS:.+]] = hw.bitcast %[[INPUT]] : (!hw.struct<a: i3, b: i2>) -> i5
    // CHECK: %[[AMOUNT:.+]] = comb.concat {{%.+}}, %[[IDX]] : i2, i3
    // CHECK: %[[SHIFTED:.+]] = comb.shru %[[BITS]], %[[AMOUNT]] : i5
    // CHECK: %[[EXTRACT:.+]] = comb.extract %[[SHIFTED]] from 0 : (i5) -> i2
    // CHECK: %[[RESULT:.+]] = hw.bitcast %[[EXTRACT]] : (i2) -> !hw.struct<b: i2>
    // CHECK: return %[[RESULT]] : !hw.struct<b: i2>
    %result = moore.dyn_extract %input from %idx : !moore.struct<{a: i3, b: i2}>, !moore.i3 -> !moore.struct<{b: i2}>
    return %result : !moore.struct<{b: i2}>
  }
}
