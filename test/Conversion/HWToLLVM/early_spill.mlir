// RUN: circt-opt %s --allow-unregistered-dialect --convert-hw-to-llvm=spill-arrays-early=true --reconcile-unrealized-casts | FileCheck %s


// CHECK-LABEL: func.func @spillNonHWGet
func.func @spillNonHWGet(%idx0 : i4, %idx1 : i4) -> (i32, i32) {
    // CHECK: [[ARRVAL:%.+]] = "foo.some_array"
    // CHECK: [[LLVAL:%.+]]  = builtin.unrealized_conversion_cast [[ARRVAL]] : !hw.array<16xi32> to !llvm.array<16 x i32>
    // CHECK: [[CST1:%.+]]   = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[ALLOCA:%.+]] = llvm.alloca [[CST1]] x !llvm.array<16 x i32>
    // CHECK: llvm.store [[LLVAL]], [[ALLOCA]]
    %arr = "foo.some_array" () : () -> (!hw.array<16xi32>)
    // CHECK-NOT: llvm.alloca
    // CHECK: [[IDX0:%.+]] = llvm.zext %arg0 : i4 to i5
    // CHECK: [[GEP0:%.+]] = llvm.getelementptr [[ALLOCA]][0, [[IDX0]]]
    // CHECK: [[LD0:%.+]]  = llvm.load [[GEP0]]
    %get0 = hw.array_get %arr[%idx0] : !hw.array<16xi32>, i4
    // CHECK-NOT: llvm.alloca
    // CHECK: [[IDX1:%.+]] = llvm.zext %arg1 : i4 to i5
    // CHECK: [[GEP1:%.+]] = llvm.getelementptr [[ALLOCA]][0, [[IDX1]]]
    // CHECK: [[LD1:%.+]]  = llvm.load [[GEP1]]
    %get1 = hw.array_get %arr[%idx1] : !hw.array<16xi32>, i4
    // CHECK: return [[LD0]], [[LD1]] : i32, i32
    return %get0, %get1 : i32, i32
}

// CHECK-LABEL: func.func @spillArgumentGet
func.func @spillArgumentGet(%idx : i4, %arr : !hw.array<16xi32>) -> (i32) {
    // CHECK: [[LLVAL:%.+]]  = builtin.unrealized_conversion_cast %arg1 : !hw.array<16xi32> to !llvm.array<16 x i32>
    // CHECK: [[CST1:%.+]]   = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[ALLOCA:%.+]] = llvm.alloca [[CST1]] x !llvm.array<16 x i32>
    // CHECK: llvm.store [[LLVAL]], [[ALLOCA]]
    // CHECK: "foo.bar"
    "foo.bar" () : () -> ()
    // CHECK-NOT: llvm.alloca
    // CHECK: [[IDX:%.+]] = llvm.zext %arg0 : i4 to i5
    // CHECK: [[GEP:%.+]] = llvm.getelementptr [[ALLOCA]][0, [[IDX]]]
    // CHECK: [[LD:%.+]]  = llvm.load [[GEP]]
    %get = hw.array_get %arr[%idx] : !hw.array<16xi32>, i4
    // CHECK: return [[LD]] : i32
    return %get : i32
}

// CHECK-LABEL: func.func @dontSpillWithoutSpillingUser
func.func @dontSpillWithoutSpillingUser() -> () {
    // CHECK-NOT: llvm.alloca
    %arr = "foo.some_array_def" () : () -> (!hw.array<16xi32>)
    "foo.some_array_use" (%arr) : (!hw.array<16xi32>) -> ()
    return
}

// CHECK-LABEL: func.func @dontSpillConstantGet
func.func @dontSpillConstantGet(%idx : i2) -> (i8) {
    // CHECK-NOT: llvm.alloca
    // CHECK: [[PTR:%.+]] = llvm.mlir.addressof
    // CHECK-NOT: llvm.alloca
    %cst = hw.aggregate_constant [3 : i8, 2 : i8, 1 : i8, 0 : i8]  : !hw.array<4xi8>
    // CHECK: [[GEP:%.+]] = llvm.getelementptr [[PTR]]
    // CHECK-NEXT: [[GET:%.+]] = llvm.load [[GEP]]
    // CHECK-NOT: llvm.alloca
    %get = hw.array_get %cst[%idx] : !hw.array<4xi8>, i2
    // CHECK: return [[GET]] : i8
    return %get : i8
}

// CHECK-LABEL: func.func @dontSpillSlicedConstant
func.func @dontSpillSlicedConstant(%sliceIdx : i2, %getIdx : i1) -> (i8) {
    // CHECK-NOT: llvm.alloca
    // CHECK: [[PTR:%.+]] = llvm.mlir.addressof
    // CHECK-NOT: llvm.alloca
    %cst = hw.aggregate_constant [3 : i8, 2 : i8, 1 : i8, 0 : i8]  : !hw.array<4xi8>
    // CHECK: [[BPTR:%.+]] = llvm.getelementptr [[PTR]]
    // CHECK-NOT: llvm.alloca
    // CHECK: [[SPTR:%.+]] = llvm.getelementptr [[BPTR]]
    // CHECK-NOT: llvm.alloca
    %slice = hw.array_slice %cst[%sliceIdx] : (!hw.array<4xi8>) -> !hw.array<2xi8>
    // CHECK: [[GET:%.+]] = llvm.load [[SPTR]]
    // CHECK-NOT: llvm.alloca
    %get   = hw.array_get %slice[%getIdx] :  !hw.array<2xi8>, i1
    // CHECK: return [[GET]] : i8
    return %get : i8
}

// CHECK-LABEL: func.func @dontRespillInject
func.func @dontRespillInject(%injectIdx : i2, %getIdx : i2) -> (i8) {
    // CHECK: [[ICST:%.+]] = llvm.mlir.constant
    %injectCst = hw.constant 0xCA : i8
    %arrCst = hw.aggregate_constant [3 : i8, 2 : i8, 1 : i8, 0 : i8]  : !hw.array<4xi8>
    // CHECK: [[ALLOCA:%.+]] = llvm.alloca
    // CHECK: [[IPTR:%.+]] = llvm.getelementptr [[ALLOCA]]
    // CHECK-NEXT: llvm.store [[ICST]], [[IPTR]]
    // CHECK-NOT: llvm.alloca
    %injected = hw.array_inject %arrCst[%injectIdx], %injectCst : !hw.array<4xi8>, i2
    // CHECK: [[GPTR:%.+]] = llvm.getelementptr [[ALLOCA]]
    // CHECK-NEXT: [[GET:%.+]] = llvm.load [[GPTR]]
    %get = hw.array_get %injected[%getIdx] : !hw.array<4xi8>, i2
    // CHECK: return [[GET]] : i8
    return %get : i8
}

// CHECK-LABEL: func.func @dontRespillConcat
func.func @dontRespillConcat(%getIdx0 : i2, %getIdx1 : i2) -> (i8, i8) {
    %arrCst = hw.aggregate_constant [0xCA : i8, 0xFE : i8]  : !hw.array<2xi8>
    // CHECK:      [[ALLOCA:%.+]] = llvm.alloca
    // CHECK-NEXT: llvm.store %{{.+}}, [[ALLOCA]]
    // CHECK-NOT:  llvm.alloca
    %concat = hw.array_concat %arrCst, %arrCst : !hw.array<2xi8>, !hw.array<2xi8>
    // CHECK: [[GPTR0:%.+]] = llvm.getelementptr [[ALLOCA]]
    // CHECK-NEXT: [[GET0:%.+]] = llvm.load [[GPTR0]]
    // CHECK-NOT:  llvm.alloca
    %get0 = hw.array_get %concat[%getIdx0] : !hw.array<4xi8>, i2
    // CHECK: [[GPTR1:%.+]] = llvm.getelementptr [[ALLOCA]]
    // CHECK-NEXT: [[GET1:%.+]] = llvm.load [[GPTR1]]
    // CHECK-NOT:  llvm.alloca
    %get1 = hw.array_get %concat[%getIdx1] : !hw.array<4xi8>, i2
    // CHECK: return [[GET0]], [[GET1]] : i8, i8
    return %get0, %get1 : i8, i8
}
