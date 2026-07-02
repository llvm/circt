// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

moore.class.classdecl @AggregateZeroClass {
}

// CHECK-LABEL: func.func @ArrayOfChandleVariable
func.func @ArrayOfChandleVariable() {
  // CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[ARRAY:%.+]] = hw.array_create [[NULL]], [[NULL]] : !llvm.ptr
  // CHECK: llhd.sig [[ARRAY]] : !hw.array<2x!llvm.ptr>
  %0 = moore.variable : <!moore.uarray<2 x chandle>>
  return
}

// CHECK-LABEL: func.func @ArrayOfClassVariable
func.func @ArrayOfClassVariable() {
  // CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[ARRAY:%.+]] = hw.array_create [[NULL]], [[NULL]] : !llvm.ptr
  // CHECK: llhd.sig [[ARRAY]] : !hw.array<2x!llvm.ptr>
  %0 = moore.variable : <!moore.uarray<2 x class<@AggregateZeroClass>>>
  return
}

// CHECK-LABEL: func.func @StructOfStringAndChandleVariable
func.func @StructOfStringAndChandleVariable() {
  // CHECK: [[STRING:%.+]] = sim.string.literal ""
  // CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[STRUCT:%.+]] = hw.struct_create ([[STRING]], [[NULL]]) : !hw.struct<s: !sim.dstring, h: !llvm.ptr>
  // CHECK: llhd.sig [[STRUCT]] : !hw.struct<s: !sim.dstring, h: !llvm.ptr>
  %0 = moore.variable : <!moore.ustruct<{s: string, h: chandle}>>
  return
}

// CHECK-LABEL: func.func @ArrayOfRealVariable
func.func @ArrayOfRealVariable() {
  // CHECK: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f64
  // CHECK: [[ARRAY:%.+]] = hw.array_create [[ZERO]], [[ZERO]] : f64
  // CHECK: llhd.sig [[ARRAY]] : !hw.array<2xf64>
  %0 = moore.variable : <!moore.uarray<2 x f64>>
  return
}

// CHECK-LABEL: func.func @StructOfTimeAndRealVariable
func.func @StructOfTimeAndRealVariable() {
  // CHECK: [[TIME:%.+]] = llhd.constant_time <0ns, 0d, 0e>
  // CHECK: [[REAL:%.+]] = arith.constant 0.000000e+00 : f64
  // CHECK: [[STRUCT:%.+]] = hw.struct_create ([[TIME]], [[REAL]]) : !hw.struct<t: !llhd.time, r: f64>
  // CHECK: llhd.sig [[STRUCT]] : !hw.struct<t: !llhd.time, r: f64>
  %0 = moore.variable : <!moore.ustruct<{t: time, r: f64}>>
  return
}

// CHECK-LABEL: func.func @ArrayOfQueueVariable
func.func @ArrayOfQueueVariable() {
  // CHECK: [[QUEUE:%.+]] = sim.queue.empty : <i32, 4>
  // CHECK: [[ARRAY:%.+]] = hw.array_create [[QUEUE]], [[QUEUE]] : !sim.queue<i32, 4>
  // CHECK: llhd.sig [[ARRAY]] : !hw.array<2x!sim.queue<i32, 4>>
  %0 = moore.variable : <!moore.uarray<2 x queue<i32, 4>>>
  return
}
