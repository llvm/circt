// RUN: circt-opt --rtg-emit-isa-assembly="path=%T split-output=true" %s && FileCheck %s --input-file=%T/test0.s --check-prefix=CHECK-TEST0 && FileCheck %s --input-file=%T/test1.s --check-prefix=CHECK-TEST1
// RUN: circt-opt --rtg-emit-isa-assembly="path=%t split-output=false" %s && FileCheck %s --input-file=%t --check-prefixes=CHECK,CHECK-TEST0,CHECK-TEST1

// CHECK: Begin of test0
// CHECK-EMPTY:

rtg.test @test0 : !rtg.dict<> {
  // CHECK-TEST0: ebreak
  rtgtest.rv32i.ebreak
}

// CHECK-EMPTY:
// CHECK: End of test0
// CHECK-EMPTY:
// CHECK-NEXT: Begin of test1
// CHECK-EMPTY:

rtg.test @test1 : !rtg.dict<> {
  // CHECK-TEST1: ecall
  rtgtest.rv32i.ecall
}

// CHECK-EMPTY:
// CHECK-NEXT: End of test1
