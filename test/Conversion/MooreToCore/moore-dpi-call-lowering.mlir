// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// Verify that moore.func.dpi and moore.func.dpi.call lower to sim.func.dpi and
// sim.func.dpi.call during MooreToCore conversion, with types converted from
// Moore to Core.

moore.func.dpi private @dpi_add(in %a : !moore.i32, in %b : !moore.i32, out return : !moore.i32 {moore.func.explicitly_returned})

// CHECK: sim.func.dpi private @dpi_add(in %a : i32, in %b : i32, out return : i32 {sim.func.explicitly_returned})

moore.func.dpi private @dpi_out(in %val : !moore.i32, out dst : !moore.i32)

// CHECK: sim.func.dpi private @dpi_out(in %val : i32, out dst : i32)

moore.func.dpi private @dpi_inout_ret(in %val : !moore.i32, inout %state : !moore.i32, out return : !moore.i32 {moore.func.explicitly_returned})

// CHECK: sim.func.dpi private @dpi_inout_ret(in %val : i32, inout %state : i32, out return : i32 {sim.func.explicitly_returned})

moore.func.dpi private @dpi_open_array(in %wd : !moore.open_uarray<i8>, out rd : !moore.ref<open_uarray<i8>>)

// CHECK: sim.func.dpi private @dpi_open_array(in %wd : !llvm.ptr, out rd : !llvm.ptr {sim.func.dpi.byref})

// CHECK-LABEL: hw.module @DpiConvTest
moore.module @DpiConvTest(in %in_a : !moore.i32, in %in_b : !moore.i32, out result : !moore.i32) {
  // CHECK: sim.func.dpi.call @dpi_add(%in_a, %in_b) : (i32, i32) -> i32
  %0 = moore.func.dpi.call @dpi_add(%in_a, %in_b) : (!moore.i32, !moore.i32) -> !moore.i32
  moore.output %0 : !moore.i32
}

// CHECK-LABEL: hw.module @DpiOutTest
moore.module @DpiOutTest(in %val : !moore.i32, out result : !moore.i32) {
  // CHECK: sim.func.dpi.call @dpi_out(%val) : (i32) -> i32
  %0 = moore.func.dpi.call @dpi_out(%val) : (!moore.i32) -> !moore.i32
  moore.output %0 : !moore.i32
}

// CHECK-LABEL: hw.module @DpiInoutRetTest
moore.module @DpiInoutRetTest(in %val : !moore.i32, in %state : !moore.i32, out state_out : !moore.i32, out ret_out : !moore.i32) {
  // CHECK: sim.func.dpi.call @dpi_inout_ret(%val, %state) : (i32, i32) -> (i32, i32)
  %0:2 = moore.func.dpi.call @dpi_inout_ret(%val, %state) : (!moore.i32, !moore.i32) -> (!moore.i32, !moore.i32)
  moore.output %0#0, %0#1 : !moore.i32, !moore.i32
}

// CHECK-LABEL: func.func @call_dpi_open_array
func.func @call_dpi_open_array(%wd: !moore.uarray<8 x i8>, %rd: !moore.ref<uarray<8 x i8>>) {
  %0 = moore.conversion %wd : !moore.uarray<8 x i8> -> !moore.open_uarray<i8>
  %1 = moore.conversion %rd : !moore.ref<uarray<8 x i8>> -> !moore.ref<open_uarray<i8>>
  // CHECK: sim.func.dpi.call @dpi_open_array(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
  moore.func.dpi.call @dpi_open_array(%0, %1) : (!moore.open_uarray<i8>, !moore.ref<open_uarray<i8>>) -> ()
  return
}

moore.func.dpi private @dpi_void(in %val : !moore.i32)

// CHECK: sim.func.dpi private @dpi_void(in %val : i32)

// CHECK-LABEL: hw.module @DpiVoidTest
moore.module @DpiVoidTest(in %val : !moore.i32) {
  // CHECK: sim.func.dpi.call @dpi_void(%val) : (i32) -> ()
  moore.func.dpi.call @dpi_void(%val) : (!moore.i32) -> ()
  moore.output
}