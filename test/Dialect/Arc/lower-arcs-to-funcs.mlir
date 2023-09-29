// RUN: circt-opt %s --arc-lower-arcs-to-funcs --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @EmptyArc() attributes {llvm.linkage = #llvm.linkage<internal>} {
arc.define @EmptyArc() {
  arc.output
  // CHECK-NEXT: return
}
// CHECK-NEXT: }

// CHECK-LABEL: func.func @test1(%arg0: i32)
arc.define @test1(%arg0: i32) -> (i32) {
    %0 = arc.call @sub1(%arg0) : (i32) -> i32
    // CHECK-NEXT: %0 = call @sub1(%arg0) : (i32) -> i32
    arc.output %0 : i32
    // CHECK-NEXT: return %0 : i32
}

// CHECK-LABEL: hw.module @test2
hw.module @test2(%arg0: i32) -> (out0: i32) {
  %0 = arc.state @sub1(%arg0) lat 0 : (i32) -> i32
  // CHECK-NEXT: %0 = func.call @sub1(%arg0) : (i32) -> i32
  hw.output %0 : i32
  // CHECK-NEXT: hw.output %0 : i32
}

// CHECK-LABEL: func.func @sub1
arc.define @sub1(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
  // CHECK-NEXT: return %arg0 : i32
}
