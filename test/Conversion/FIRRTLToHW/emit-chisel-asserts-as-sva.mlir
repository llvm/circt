// RUN: circt-opt -lower-firrtl-to-hw=emit-chisel-asserts-as-sva %s | FileCheck %s

firrtl.circuit "ifElseFatalToSVA" {
  // CHECK-LABEL: hw.module @ifElseFatalToSVA
  firrtl.module @ifElseFatalToSVA(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert0" {isConcurrent = true, format = "ifElseFatal"}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %cond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT: sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT: }
}
}
