// RUN: circt-opt %s --arc-lower-arcs-to-funcs --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @EmptyArc() attributes {llvm.linkage = #llvm.linkage<internal>} {
arc.define @EmptyArc() {
  arc.output
  // CHECK-NEXT: return
}
// CHECK-NEXT: }

// CHECK-LABEL: func.func @callTest(%arg0: i32)
arc.define @callTest(%arg0: i32) -> (i32) {
    %0 = arc.call @sub1(%arg0) : (i32) -> i32
    // CHECK-NEXT: %0 = call @sub1(%arg0) : (i32) -> i32
    arc.output %0 : i32
    // CHECK-NEXT: return %0 : i32
}

// CHECK-LABEL: hw.module @callInModuleTest
hw.module @callInModuleTest(in %arg0: i32, out out0: i32) {
  %0 = arc.call @sub1(%arg0) : (i32) -> i32
  // CHECK-NEXT: %0 = func.call @sub1(%arg0) : (i32) -> i32
  hw.output %0 : i32
  // CHECK-NEXT: hw.output %0 : i32
}

// CHECK-LABEL: func.func @sub1
arc.define @sub1(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
  // CHECK-NEXT: return %arg0 : i32
}

// CHECK-LABEL: hw.module @DontConvertExecuteOps
hw.module @DontConvertExecuteOps(in %arg0: i32, out out0: i32) {
  // CHECK: arc.execute
  // CHECK: arc.output
  %0 = arc.execute (%arg0 : i32) -> (i32) {
  ^bb0(%1: i32):
    arc.output %1 : i32
  }
  hw.output %0 : i32
}
