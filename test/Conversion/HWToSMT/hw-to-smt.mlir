// RUN: circt-opt %s --convert-hw-to-smt | FileCheck %s

// CHECK-LABEL: func @test
func.func @test() {
  // CHECK: smt.bv.constant #smt.bv<42> : !smt.bv<32>
  %c42_i32 = hw.constant 42 : i32
  // CHECK: smt.bv.constant #smt.bv<-1> : !smt.bv<3>
  %c-1_i32 = hw.constant -1 : i3
  // CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<1>
  %false = hw.constant false

  return
}

// CHECK-LABEL: func.func @modA(%{{.*}}: !smt.bv<32>) -> !smt.bv<32>
hw.module @modA(in %in: i32, out out: i32) {
  // CHECK-NEXT: return
  hw.output %in : i32
}

// CHECK-LABEL: func.func @modB(%{{.*}}: !smt.bv<32>) -> !smt.bv<32>
hw.module @modB(in %in: i32, out out: i32) {
  // CHECK-NEXT: [[V:%.+]] = call @modA(%{{.*}}) : (!smt.bv<32>) -> !smt.bv<32>
  %0 = hw.instance "inst" @modA(in: %in: i32) -> (out: i32)
  // CHECK-NEXT: return [[V]] : !smt.bv<32>
  hw.output %0 : i32
}
