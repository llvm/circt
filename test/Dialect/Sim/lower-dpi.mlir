// RUN: circt-opt --sim-lower-dpi-func %s | FileCheck %s

sim.func.dpi @foo(out arg0: i32, in %arg1: i32, return ret: i32)
// CHECK-LABEL:  func.func private @foo(!llvm.ptr, i32) -> i32
// CHECK-LABEL:  func.func @foo_wrapper(%arg0: i32) -> (i32, i32) {
// CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %2 = call @foo(%1, %arg0) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:    %3 = llvm.load %1 : !llvm.ptr -> i32
// CHECK-NEXT:    return %3, %2 : i32, i32
// CHECK-NEXT:   }

// CHECK-LABEL:  func.func @bar_wrapper(%arg0: i32) -> (i32, i32) {
// CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %2 = call @bar_c_name(%1, %arg0) : (!llvm.ptr, i32) -> i32
// CHECK-NEXT:    %3 = llvm.load %1 : !llvm.ptr -> i32
// CHECK-NEXT:    return %3, %2 : i32, i32
// CHECK-NEXT:   }
// CHECK-LABEL:  func.func @bar_c_name

sim.func.dpi @bar(out arg0: i32, in %arg1: i32, return ret: i32) attributes {verilogName="bar_c_name"}
func.func @bar_c_name(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
  %0 = arith.constant 0 : i32
  func.return %0 : i32
}

// CHECK-LABEL:  func.func private @baz_c_name(!llvm.ptr, i32) -> i32
// CHECK-LABEL:  func.func @baz_wrapper(%arg0: i32) -> (i32, i32)
// CHECK:     call @baz_c_name(%1, %arg0) : (!llvm.ptr, i32) -> i32
sim.func.dpi @baz(out arg0: i32, in %arg1: i32, return ret: i32) attributes {verilogName="baz_c_name"}

sim.func.dpi @qux(in %arg0: i32, inout %state: i32)
// CHECK-LABEL:  func.func private @qux(i32, !llvm.ptr)
// CHECK-LABEL:  func.func @qux_wrapper(%arg0: i32, %arg1: i32) -> i32 {
// CHECK:         llvm.store %arg1, %{{.*}} : i32, !llvm.ptr
// CHECK:         call @qux(%arg0, %{{.*}}) : (i32, !llvm.ptr) -> ()
// CHECK:         %{{.*}} = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:         return %{{.*}} : i32

sim.func.dpi @open_array_rw(in %wd: !llvm.ptr, ref %rd: !llvm.ptr)
// CHECK-LABEL:  func.func private @open_array_rw(!llvm.ptr, !llvm.ptr)
// CHECK-LABEL:  func.func @open_array_rw_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// CHECK:         call @open_array_rw(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:         return

// CHECK-LABEL: hw.module @dpi_call
hw.module @dpi_call(in %clock : !seq.clock, in %enable : i1, in %in: i32,
          out o1: i32, out o2: i32, out o3: i32, out o4: i32, out o5: i32, out o6: i32, out o7: i32) {
  // CHECK-NEXT: %0:2 = sim.func.dpi.call @foo_wrapper(%in) clock %clock : (i32) -> (i32, i32)
  // CHECK-NEXT: %1:2 = sim.func.dpi.call @bar_wrapper(%in) : (i32) -> (i32, i32)
  // CHECK-NEXT: %2:2 = sim.func.dpi.call @baz_wrapper(%in) : (i32) -> (i32, i32)
  // CHECK-NEXT: %3 = sim.func.dpi.call @qux_wrapper(%in, %in) : (i32, i32) -> i32
  // CHECK-NEXT: hw.output %0#0, %0#1, %1#0, %1#1, %2#0, %2#1, %3 : i32, i32, i32, i32, i32, i32, i32
  %0, %1 = sim.func.dpi.call @foo(%in) clock %clock : (i32) -> (i32, i32)
  %2, %3 = sim.func.dpi.call @bar(%in) : (i32) -> (i32, i32)
  %4, %5 = sim.func.dpi.call @baz(%in) : (i32) -> (i32, i32)
  %6 = sim.func.dpi.call @qux(%in, %in) : (i32, i32) -> i32

  hw.output %0, %1, %2, %3, %4, %5, %6 : i32, i32, i32, i32, i32, i32, i32
}
