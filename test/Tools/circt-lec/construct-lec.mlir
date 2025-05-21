// RUN: circt-opt --construct-lec="first-module=modA0 second-module=modB0 insert-main=true" %s | FileCheck %s

hw.module @modA0(in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  hw.output %0 : i32
}

hw.module @modB0(in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.mul %in0, %in1 : i32
  hw.output %0 : i32
}

// CHECK: func.func @modA0() {
// CHECK:   [[V0:%.+]] = verif.lec first {
// CHECK:   ^bb0([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32):
// CHECK:     [[V1:%.+]] = comb.add [[ARG0]], [[ARG1]]
// CHECK:     verif.yield [[V1]]
// CHECK:   } second {
// CHECK:   ^bb0([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32):
// CHECK:     [[V2:%.+]] = comb.mul [[ARG0]], [[ARG1]]
// CHECK:     verif.yield [[V2]]
// CHECK:   }
// CHECK:   [[S0:%.+]] = llvm.mlir.addressof @"c1 == c2\0A" : !llvm.ptr
// CHECK:   [[S1:%.+]] = llvm.mlir.addressof @"c1 != c2\0A" : !llvm.ptr
// CHECK:   [[V3:%.+]] = llvm.select [[V0]], [[S0]], [[S1]]
// CHECK:   llvm.call @printf([[V3]])
// CHECK:   return

// CHECK: func.func @main(%{{.*}}: i32, %{{.*}}: !llvm.ptr) -> i32 {
// CHECK:   call @modA0() : () -> ()
// CHECK:   [[V4:%.+]] = llvm.mlir.constant(0 : i32)
// CHECK:   return [[V4]]

// CHECK: llvm.mlir.global private constant @"c1 == c2\0A"("c1 == c2\0A\00") {addr_space = 0 : i32}
// CHECK: llvm.mlir.global private constant @"c1 != c2\0A"("c1 != c2\0A\00") {addr_space = 0 : i32}


// RUN: circt-opt --construct-lec="first-module=modA1 second-module=modB1 insert-main=false" %s | FileCheck %s --check-prefix=CHECK1
// Test that using the same module twice doesn't lead to a double free
// RUN: circt-opt --construct-lec="first-module=modA1 second-module=modA1 insert-main=false" %s

hw.module @modA1() {
  hw.output
}

hw.module @modB1() {
  hw.output
}

// CHECK1: func.func @modA1() {
// CHECK1:   [[V0:%.+]] = verif.lec first {
// CHECK1:   } second {
// CHECK1:   }
// CHECK1:   [[V1:%.+]] = llvm.mlir.addressof @"c1 == c2\0A" : !llvm.ptr
// CHECK1:   [[V2:%.+]] = llvm.mlir.addressof @"c1 != c2\0A" : !llvm.ptr
// CHECK1:   [[V3:%.+]] = llvm.select [[V0]], [[V1]], [[V2]]
// CHECK1:   llvm.call @printf([[V3]])
// CHECK1:   return
// CHECK1: }

// RUN: circt-opt --construct-lec="first-module=modA0 second-module=modB0 insert-main=false insert-report=false" %s | FileCheck %s --check-prefix=CHECK2

// CHECK2:   [[V0:%.+]] = verif.lec first {
// CHECK2:   ^bb0([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32):
// CHECK2:     [[V1:%.+]] = comb.add [[ARG0]], [[ARG1]]
// CHECK2:     verif.yield [[V1]]
// CHECK2:   } second {
// CHECK2:   ^bb0([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32):
// CHECK2:     [[V2:%.+]] = comb.mul [[ARG0]], [[ARG1]]
// CHECK2:     verif.yield [[V2]]
// CHECK2:   }
