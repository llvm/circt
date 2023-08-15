//RUN: circt-opt -sv-trigger-attr-gen=reset-trigger-value %s | FileCheck %s

// Test for resetting the trigger attribute in the module 

llvm.mlir.global internal @_Struct_test_case_1() {addr_space = 0 : i32} : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> {
  %0 = llvm.mlir.constant(false) : i1
  %1 = llvm.mlir.constant(false) : i1
  %2 = llvm.mlir.constant(false) : i1
  %3 = llvm.mlir.constant(false) : i1
  %4 = llvm.mlir.constant(false) : i1
  %5 = llvm.mlir.constant(false) : i1
  %6 = llvm.mlir.constant(false) : i1
  %7 = llvm.mlir.constant(false) : i1
  %8 = llvm.mlir.null : !llvm.ptr<struct<"_Struct_extern_module_1", opaque>>
  %9 = llvm.mlir.null : !llvm.ptr<struct<"_Struct_extern_module_0", opaque>>
  %10 = llvm.mlir.undef : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>
  %11 = llvm.insertvalue %0, %10[0] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %12 = llvm.insertvalue %1, %11[1] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %13 = llvm.insertvalue %2, %12[2] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %14 = llvm.insertvalue %3, %13[3] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %15 = llvm.insertvalue %4, %14[4] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %16 = llvm.insertvalue %5, %15[5] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %17 = llvm.insertvalue %6, %16[6] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %18 = llvm.insertvalue %7, %17[7] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %19 = llvm.insertvalue %8, %18[8] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  %20 = llvm.insertvalue %9, %19[9] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)> 
  llvm.return %20 : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>
}
llvm.mlir.global internal @_Struct_test_case_0() {addr_space = 0 : i32} : !llvm.struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)> {
  %0 = llvm.mlir.constant(false) : i1
  %1 = llvm.mlir.constant(false) : i1
  %2 = llvm.mlir.constant(false) : i1
  %3 = llvm.mlir.constant(false) : i1
  %4 = llvm.mlir.undef : !llvm.struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)>
  %5 = llvm.insertvalue %0, %4[0] : !llvm.struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)> 
  %6 = llvm.insertvalue %2, %5[1] : !llvm.struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)> 
  %7 = llvm.insertvalue %1, %6[2] : !llvm.struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)> 
  %8 = llvm.insertvalue %3, %7[3] : !llvm.struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)> 
  llvm.return %8 : !llvm.struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)>
}

// CHECK-LABEL: hw.module @test_case_0(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)>>) {
hw.module @test_case_0(%ptr_struct_test_case_0: !llvm.ptr<struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)>>) attributes {sv.trigger = [0 : i8, 1 : i8]} {

  // CHECK: llvm.getelementptr inbounds %ptr_struct_test_case_0{{\[}}%0, 0] : (!llvm.ptr<struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.getelementptr inbounds %ptr_struct_test_case_0[%0, 0] : (!llvm.ptr<struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>

  // CHECK: llvm.load %1 {sv.trigger = [0 : i8]} : !llvm.ptr<i1>
  %2 = llvm.load %1 : !llvm.ptr<i1>

  %3 = llvm.getelementptr inbounds %ptr_struct_test_case_0[%0, 1] : (!llvm.ptr<struct<"_Struct_test_case_0", packed (i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  %4 = llvm.load %3 : !llvm.ptr<i1>
  sv.always posedge %2 {
  }
  sv.always posedge %2, posedge %4 {
  }
  hw.output 
}

// CHECK-LABEL: hw.module @test_case_1(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>>) {
hw.module @test_case_1(%ptr_struct_test_case_1: !llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>>) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.getelementptr inbounds %ptr_struct_test_case_1[%0, 0] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>>, i32) -> !llvm.ptr<i1>
  %2 = llvm.load %1 : !llvm.ptr<i1>
  %3 = llvm.getelementptr inbounds %ptr_struct_test_case_1[%0, 1] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>>, i32) -> !llvm.ptr<i1>
  %4 = llvm.load %3 : !llvm.ptr<i1>
  %5 = llvm.getelementptr inbounds %ptr_struct_test_case_1[%0, 2] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>>, i32) -> !llvm.ptr<i1>
  %6 = llvm.load %5 : !llvm.ptr<i1>
  %7 = llvm.getelementptr inbounds %ptr_struct_test_case_1[%0, 3] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>>, i32) -> !llvm.ptr<i1>
  %8 = llvm.load %7 : !llvm.ptr<i1>
  %9 = llvm.getelementptr inbounds %ptr_struct_test_case_1[%0, 8] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>>, i32) -> !llvm.ptr<ptr<struct<"_Struct_extern_module_1", opaque>>>
  %10 = llvm.load %9 : !llvm.ptr<ptr<struct<"_Struct_extern_module_1", opaque>>>
  %11 = llvm.bitcast %10 : !llvm.ptr<struct<"_Struct_extern_module_1", opaque>> to !llvm.ptr<struct<packed (i1, i1)>>
  %12 = llvm.getelementptr inbounds %11[%0, 0] : (!llvm.ptr<struct<packed (i1, i1)>>, i32) -> !llvm.ptr<i1>
  llvm.store %2, %12 : !llvm.ptr<i1>

  // CHECK: hw.instance "instance_0" @extern_module_1(ptr_struct_extern_module_1: %10: !llvm.ptr<struct<"_Struct_extern_module_1", opaque>>) -> ()
  hw.instance "instance_0" @extern_module_1(ptr_struct_extern_module_1: %10: !llvm.ptr<struct<"_Struct_extern_module_1", opaque>>) -> () {sv.trigger = [1 : i8]}


  // CHECK: llvm.bitcast %10 : !llvm.ptr<struct<"_Struct_extern_module_1", opaque>> to !llvm.ptr<struct<packed (i1, i1)>>
  %13 = llvm.bitcast %10 : !llvm.ptr<struct<"_Struct_extern_module_1", opaque>> to !llvm.ptr<struct<packed (i1, i1)>>

  %14 = llvm.mlir.constant(1 : i32) : i32

  // CHECK: llvm.getelementptr inbounds %13{{\[}}%0, 1] : (!llvm.ptr<struct<packed (i1, i1)>>, i32) -> !llvm.ptr<i1>
  %15 = llvm.getelementptr inbounds %13[%0, 1] : (!llvm.ptr<struct<packed (i1, i1)>>, i32) -> !llvm.ptr<i1>

  // CHECK: llvm.load %15 {sv.trigger = [0 : i8]} : !llvm.ptr<i1>
  %16 = llvm.load %15 : !llvm.ptr<i1>

  %17 = llvm.getelementptr inbounds %ptr_struct_test_case_1[%0, 9] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_extern_module_1", opaque>>, ptr<struct<"_Struct_extern_module_0", opaque>>)>>, i32) -> !llvm.ptr<ptr<struct<"_Struct_extern_module_0", opaque>>>
  %18 = llvm.load %17 : !llvm.ptr<ptr<struct<"_Struct_extern_module_0", opaque>>>
  %19 = llvm.bitcast %18 : !llvm.ptr<struct<"_Struct_extern_module_0", opaque>> to !llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>
  %20 = llvm.getelementptr inbounds %19[%0, 0] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  llvm.store %2, %20 : !llvm.ptr<i1>
  %21 = llvm.getelementptr inbounds %19[%0, 1] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  llvm.store %4, %21 : !llvm.ptr<i1>
  %22 = llvm.getelementptr inbounds %19[%0, 2] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  llvm.store %6, %22 : !llvm.ptr<i1>
  %23 = llvm.getelementptr inbounds %19[%0, 3] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  llvm.store %8, %23 : !llvm.ptr<i1>

  // CHECK: hw.instance "instance_1" @extern_module_0(ptr_struct_extern_module_0: %18: !llvm.ptr<struct<"_Struct_extern_module_0", opaque>>) -> ()
  hw.instance "instance_1" @extern_module_0(ptr_struct_extern_module_0: %18: !llvm.ptr<struct<"_Struct_extern_module_0", opaque>>) -> () {sv.trigger = [4 : i8, 5 : i8, 6 : i8]}

  // CHECK: llvm.bitcast %18 : !llvm.ptr<struct<"_Struct_extern_module_0", opaque>> to !llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>
  %24 = llvm.bitcast %18 : !llvm.ptr<struct<"_Struct_extern_module_0", opaque>> to !llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>
  %25 = llvm.mlir.constant(4 : i32) : i32

  // CHECK: llvm.getelementptr inbounds %24{{\[}}%0, 4] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  // CHECK: llvm.load %26 {sv.trigger = [1 : i8]} : !llvm.ptr<i1>
  %26 = llvm.getelementptr inbounds %24[%0, 4] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  %27 = llvm.load %26 : !llvm.ptr<i1>

  %28 = llvm.mlir.constant(5 : i32) : i32

  // CHECK: llvm.getelementptr inbounds %24{{\[}}%0, 5] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  // CHECK: llvm.load %29 {sv.trigger = [2 : i8]} : !llvm.ptr<i1>
  %29 = llvm.getelementptr inbounds %24[%0, 5] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  %30 = llvm.load %29 : !llvm.ptr<i1>

  %31 = llvm.mlir.constant(6 : i32) : i32

  // CHECK: llvm.getelementptr inbounds %24{{\[}}%0, 6] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  // CHECK: llvm.load %32 {sv.trigger = [3 : i8]} : !llvm.ptr<i1>
  %32 = llvm.getelementptr inbounds %24[%0, 6] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
  %33 = llvm.load %32 : !llvm.ptr<i1>

  sv.always posedge %16 {
  }
  sv.always posedge %30, negedge %16 {
  }
  sv.always posedge %27, negedge %30, edge %33 {
  }
  hw.output
}

hw.module.extern private @extern_module_1(%ptr_struct_extern_module_1: !llvm.ptr<struct<"_Struct_extern_module_1", opaque>>)
hw.module.extern private @extern_module_0(%ptr_struct_extern_module_0: !llvm.ptr<struct<"_Struct_extern_module_0", opaque>>)
