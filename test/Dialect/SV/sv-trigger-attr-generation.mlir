//RUN: circt-opt -sv-trigger-attr-gen %s | FileCheck %s

// Test trigger attribute generation in the modules arguments.
// CHECK-LABEL: hw.module @test_case_0(
// CHECK-SAME: %[[VAL_0:.*]]: i1,
// CHECK-SAME: %[[VAL_1:.*]]: i1) attributes {sv.trigger = [0 : i8, 1 : i8]} {
hw.module @test_case_0(%arg0: i1, %arg1: i1){
  sv.always posedge %arg0 {}
  sv.always posedge %arg0, posedge %arg1 {}
  hw.output
}

// Test secondary trigger attribute generation in the operations attributes part.
// CHECK-LABEL: hw.module @test_case_1(
hw.module @test_case_1(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) {
  // CHECK-NEXT: %[[VAL_4:.*]] = hw.instance "instance_0" @extern_module_1(arg0: %arg0: i1) -> (out0: i1) {sv.trigger = [1 : i8]}
  %instance_0.out_0 = hw.instance "instance_0" @extern_module_1(arg0: %arg0: i1) -> (out0: i1)
  
  // CHECK-NEXT: %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "instance_1" @extern_module_0(arg0: %arg0: i1, arg1: %arg1: i1, arg2: %arg2: i1, arg3: %arg3: i1) -> (out0: i1, out1: i1, out2: i1) {sv.trigger = [4 : i8, 5 : i8, 6 : i8]}
  %instance_1.out0, %instance_1.out1, %instance_1.out2 = hw.instance "instance_1" @extern_module_0(arg0: %arg0: i1, arg1: %arg1: i1, arg2: %arg2: i1, arg3: %arg3: i1) -> (out0: i1, out1: i1, out2: i1)

  sv.always posedge %instance_0.out_0 {}
  sv.always posedge %instance_1.out1, negedge %instance_0.out_0 {}
  sv.always posedge %instance_1.out0, negedge %instance_1.out1, edge %instance_1.out2 {}
  hw.output
}

hw.module.extern private @extern_module_1(%arg0: i1) -> (out0: i1)
hw.module.extern private @extern_module_0(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) -> (out0: i1, out1: i1, out2: i1)
