// RUN: circt-opt --rtg-simple-test-inliner --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-NOT: rtg.target @tgt1
rtg.target @tgt1 : !rtg.dict<imm: !rtg.isa.immediate<32>> {
  %imm = rtg.constant #rtg.isa.immediate<32, 0>
  rtg.yield %imm : !rtg.isa.immediate<32>
}

rtg.test @test1() {
  rtg.comment "Inside test1"
}

rtg.test @test2(imm = %imm: !rtg.isa.immediate<32>) target @tgt1 {
  rtg.comment "Inside test2"
}

rtg.test @test3(imm = %imm: !rtg.isa.immediate<32>) {
  rtg.comment "Inside test3"
}

// CHECK-LABEL: emit.file "filename"
emit.file "filename" {
  // CHECK-NOT: emit.ref
  // CHECK-NEXT: rtg.comment "Begin of test 'test1'"
  // CHECK-NEXT: rtg.comment "Inside test1"
  // CHECK-NEXT: rtg.comment "End of test 'test1'"
  emit.ref @test1
  // CHECK-NOT: emit.ref
  // CHECK-NEXT: rtg.comment "Begin of test 'test2'"
  // CHECK-NEXT: rtg.comment "Inside test2"
  // CHECK-NEXT: rtg.comment "End of test 'test2'"
  emit.ref @test2
  // CHECK-NOT: emit.ref
  // CHECK-NEXT: rtg.comment "Begin of test 'test3'"
  // CHECK-NEXT: rtg.comment "Inside test3"
  // CHECK-NEXT: rtg.comment "End of test 'test3'"
  emit.ref @test3
  // CHECK-NOT: emit.ref
}

// -----

// expected-error @below {{cannot inline test with used arguments}}
rtg.test @test(imm = %imm: !rtg.isa.immediate<32>) {
  %reg = rtg.fixed_reg #rtgtest.t0
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
