// RUN: circt-opt %s --convert-hw-to-llvm=spill-arrays-early=false | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<i65, dense<128> : vector<2xi64>>,
  #dlti.dl_entry<!llvm.array<2 x i65>, dense<128> : vector<2xi64>>
>} {

  // CHECK-LABEL: func.func @wideArrayGet
  func.func @wideArrayGet(%idx : i1, %arr: !hw.array<2xi65>) -> i65 {
    // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi65> to !llvm.array<2 x i65>
    // CHECK: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[ALLOCA:%.+]] = llvm.alloca [[ONE]] x !llvm.array<2 x i65> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    // CHECK: llvm.store [[CAST]], [[ALLOCA]] : !llvm.array<2 x i65>, !llvm.ptr
    // CHECK: [[IDX:%.+]] = llvm.zext %arg0 : i1 to i2
    // CHECK: [[GEP:%.+]] = llvm.getelementptr [[ALLOCA]][0, [[IDX]]] : (!llvm.ptr, i2) -> !llvm.ptr, !llvm.array<2 x i65>
    // CHECK: [[VAL:%.+]] = llvm.load [[GEP]] : !llvm.ptr -> i65
    %0 = hw.array_get %arr[%idx] : !hw.array<2xi65>, i1
    // CHECK: return [[VAL]] : i65
    return %0 : i65
  }

  // CHECK-LABEL: func.func @wideArrayInject
  func.func @wideArrayInject(%arr: !hw.array<2xi65>, %idx: i1, %elt: i65) -> !hw.array<2xi65> {
    // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !hw.array<2xi65> to !llvm.array<2 x i65>
    // CHECK: [[ONE:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[ZEXT:%.+]] = llvm.zext %arg1 : i1 to i2
    // CHECK: [[ALLOCA:%.+]] = llvm.alloca [[ONE]] x !llvm.array<2 x i65> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    // CHECK: llvm.store [[CAST]], [[ALLOCA]] : !llvm.array<2 x i65>, !llvm.ptr
    // CHECK: [[GEP:%.+]] = llvm.getelementptr [[ALLOCA]][0, [[ZEXT]]] : (!llvm.ptr, i2) -> !llvm.ptr, !llvm.array<2 x i65>
    // CHECK: llvm.store %arg2, [[GEP]] : i65, !llvm.ptr
    // CHECK: [[OUT:%.+]] = llvm.load [[ALLOCA]] : !llvm.ptr -> !llvm.array<2 x i65>
    %0 = hw.array_inject %arr[%idx], %elt : !hw.array<2xi65>, i1
    // CHECK: [[CASTOUT:%.+]] = builtin.unrealized_conversion_cast [[OUT]] : !llvm.array<2 x i65> to !hw.array<2xi65>
    // CHECK: return [[CASTOUT]] : !hw.array<2xi65>
    return %0 : !hw.array<2xi65>
  }
}
