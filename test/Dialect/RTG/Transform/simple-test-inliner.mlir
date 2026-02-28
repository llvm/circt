// RUN: circt-opt --rtg-simple-test-inliner --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-NOT: rtg.target @tgt1
rtg.target @tgt1 : !rtg.dict<imm: !rtg.isa.immediate<32>> {
  %imm = rtg.constant #rtg.isa.immediate<32, 0>
  rtg.yield %imm : !rtg.isa.immediate<32>
}

rtg.test @test1() {
  %0 = rtg.constant "Inside test1" : !rtg.string
  rtg.comment %0
}

rtg.test @test2(imm = %imm: !rtg.isa.immediate<32>) target @tgt1 {
  %0 = rtg.constant "Inside test2" : !rtg.string
  rtg.comment %0
}

rtg.test @test3(imm = %imm: !rtg.isa.immediate<32>) {
  %0 = rtg.constant "Inside test3" : !rtg.string
  rtg.comment %0
}

// CHECK-LABEL: emit.file "filename"
emit.file "filename" {
  // CHECK: rtg.isa.segment text
  // CHECK-NOT: emit.ref
  // CHECK-DAG: [[STR0:%.+]] = rtg.constant "Begin of test 'test1'" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR0]]
  // CHECK-DAG: [[STR1:%.+]] = rtg.constant "Inside test1" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR1]]
  // CHECK-DAG: [[STR2:%.+]] = rtg.constant "End of test 'test1'" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR2]]
  emit.ref @test1
  // CHECK-NOT: emit.ref
  // CHECK-DAG: [[STR3:%.+]] = rtg.constant "Begin of test 'test2'" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR3]]
  // CHECK-DAG: [[STR4:%.+]] = rtg.constant "Inside test2" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR4]]
  // CHECK-DAG: [[STR5:%.+]] = rtg.constant "End of test 'test2'" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR5]]
  emit.ref @test2
  // CHECK-NOT: emit.ref
  // CHECK-DAG: [[STR6:%.+]] = rtg.constant "Begin of test 'test3'" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR6]]
  // CHECK-DAG: [[STR7:%.+]] = rtg.constant "Inside test3" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR7]]
  // CHECK-DAG: [[STR8:%.+]] = rtg.constant "End of test 'test3'" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR8]]
  emit.ref @test3
  // CHECK-NOT: emit.ref
}

// -----

// expected-error @below {{cannot inline test with used arguments}}
rtg.test @test(imm = %imm: !rtg.isa.immediate<32>) {
  %reg = rtg.constant #rtgtest.t0
  rtgtest.rv32i.lui %reg, %imm : !rtg.isa.immediate<32>
}

emit.file "filename" {
  emit.ref @test
}

// -----

hw.module @mod() {}

emit.file "filename" {
  // expected-error @below {{invalid symbol reference: @mod}}
  emit.ref @mod
}
