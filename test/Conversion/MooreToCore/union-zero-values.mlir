// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

moore.class.classdecl @UnionZeroClass {
}

// CHECK-LABEL: func.func @UnionStringVariable
func.func @UnionStringVariable() {
  // CHECK: [[STRING:%.+]] = sim.string.literal ""
  // CHECK: [[UNION:%.+]] = hw.union_create "s", [[STRING]] : !hw.union<s: !sim.dstring, bits: i32>
  // CHECK: llhd.sig [[UNION]] : !hw.union<s: !sim.dstring, bits: i32>
  %0 = moore.variable : <!moore.uunion<{s: string, bits: i32}>>
  return
}

// CHECK-LABEL: func.func @UnionChandleVariable
func.func @UnionChandleVariable() {
  // CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[UNION:%.+]] = hw.union_create "h", [[NULL]] : !hw.union<h: !llvm.ptr, bits: i32>
  // CHECK: llhd.sig [[UNION]] : !hw.union<h: !llvm.ptr, bits: i32>
  %0 = moore.variable : <!moore.uunion<{h: chandle, bits: i32}>>
  return
}

// CHECK-LABEL: func.func @UnionClassVariable
func.func @UnionClassVariable() {
  // CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[UNION:%.+]] = hw.union_create "c", [[NULL]] : !hw.union<c: !llvm.ptr, bits: i32>
  // CHECK: llhd.sig [[UNION]] : !hw.union<c: !llvm.ptr, bits: i32>
  %0 = moore.variable : <!moore.uunion<{c: class<@UnionZeroClass>, bits: i32}>>
  return
}

// CHECK-LABEL: func.func @UnionIntegerFirstVariable
func.func @UnionIntegerFirstVariable() {
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i32
  // CHECK: [[UNION:%.+]] = hw.union_create "bits", [[ZERO]] : !hw.union<bits: i32, s: !sim.dstring>
  // CHECK: llhd.sig [[UNION]] : !hw.union<bits: i32, s: !sim.dstring>
  %0 = moore.variable : <!moore.uunion<{bits: i32, s: string}>>
  return
}

// CHECK-LABEL: func.func @UnionRealVariable
func.func @UnionRealVariable() {
  // CHECK: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f64
  // CHECK: [[UNION:%.+]] = hw.union_create "r", [[ZERO]] : !hw.union<r: f64, bits: i32>
  // CHECK: llhd.sig [[UNION]] : !hw.union<r: f64, bits: i32>
  %0 = moore.variable : <!moore.uunion<{r: f64, bits: i32}>>
  return
}

// CHECK-LABEL: func.func @UnionTimeVariable
func.func @UnionTimeVariable() {
  // CHECK: [[TIME:%.+]] = llhd.constant_time <0ns, 0d, 0e>
  // CHECK: [[UNION:%.+]] = hw.union_create "t", [[TIME]] : !hw.union<t: !llhd.time, bits: i32>
  // CHECK: llhd.sig [[UNION]] : !hw.union<t: !llhd.time, bits: i32>
  %0 = moore.variable : <!moore.uunion<{t: time, bits: i32}>>
  return
}

// CHECK-LABEL: func.func @UnionQueueVariable
func.func @UnionQueueVariable() {
  // CHECK: [[QUEUE:%.+]] = sim.queue.empty : <i32, 4>
  // CHECK: [[UNION:%.+]] = hw.union_create "q", [[QUEUE]] : !hw.union<q: !sim.queue<i32, 4>, bits: i32>
  // CHECK: llhd.sig [[UNION]] : !hw.union<q: !sim.queue<i32, 4>, bits: i32>
  %0 = moore.variable : <!moore.uunion<{q: queue<i32, 4>, bits: i32}>>
  return
}
