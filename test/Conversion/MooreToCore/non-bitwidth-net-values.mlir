// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

moore.class.classdecl @NetClass {
}

// CHECK-LABEL: func.func @StringWire
func.func @StringWire() {
  // CHECK: %[[EMPTY:.*]] = sim.string.literal ""
  // CHECK: llhd.sig %[[EMPTY]] : !sim.dstring
  %net = moore.net wire : <!moore.string>
  return
}

// CHECK-LABEL: func.func @ChandleWire
func.func @ChandleWire() {
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: llhd.sig %[[NULL]] : !llvm.ptr
  %net = moore.net wire : <!moore.chandle>
  return
}

// CHECK-LABEL: func.func @ClassWire
func.func @ClassWire() {
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: llhd.sig %[[NULL]] : !llvm.ptr
  %net = moore.net wire : <!moore.class<@NetClass>>
  return
}

// CHECK-LABEL: func.func @StringWireWithAssignment
// CHECK-SAME: (%arg0: !sim.dstring)
func.func @StringWireWithAssignment(%value: !moore.string) {
  // CHECK: %[[EMPTY:.*]] = sim.string.literal ""
  // CHECK: %[[SIG:.*]] = llhd.sig %[[EMPTY]] : !sim.dstring
  %net = moore.net wire %value : <!moore.string>
  // CHECK: %[[DELAY:.*]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %[[SIG]], %arg0 after %[[DELAY]] : !sim.dstring
  return
}

// CHECK-LABEL: func.func @ClassWireWithAssignment
// CHECK-SAME: (%arg0: !llvm.ptr)
func.func @ClassWireWithAssignment(%value: !moore.class<@NetClass>) {
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[SIG:.*]] = llhd.sig %[[NULL]] : !llvm.ptr
  %net = moore.net wire %value : <!moore.class<@NetClass>>
  // CHECK: %[[DELAY:.*]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %[[SIG]], %arg0 after %[[DELAY]] : !llvm.ptr
  return
}

// CHECK-LABEL: func.func @QueueWire
func.func @QueueWire() {
  // CHECK: %[[EMPTY:.*]] = sim.queue.empty : <i32, 4>
  // CHECK: llhd.sig %[[EMPTY]] : !sim.queue<i32, 4>
  %net = moore.net wire : <!moore.queue<i32, 4>>
  return
}

// CHECK-LABEL: func.func @QueueClassWire
func.func @QueueClassWire() {
  // CHECK: %[[EMPTY:.*]] = sim.queue.empty : <!llvm.ptr, 4>
  // CHECK: llhd.sig %[[EMPTY]] : !sim.queue<!llvm.ptr, 4>
  %net = moore.net wire : <!moore.queue<class<@NetClass>, 4>>
  return
}

// CHECK-LABEL: func.func @QueueRealWire
func.func @QueueRealWire() {
  // CHECK: %[[EMPTY:.*]] = sim.queue.empty : <f64, 4>
  // CHECK: llhd.sig %[[EMPTY]] : !sim.queue<f64, 4>
  %net = moore.net wire : <!moore.queue<f64, 4>>
  return
}

// CHECK-LABEL: func.func @QueueTimeWire
func.func @QueueTimeWire() {
  // CHECK: %[[EMPTY:.*]] = sim.queue.empty : <!llhd.time, 4>
  // CHECK: llhd.sig %[[EMPTY]] : !sim.queue<!llhd.time, 4>
  %net = moore.net wire : <!moore.queue<time, 4>>
  return
}

// CHECK-LABEL: func.func @RealWire
func.func @RealWire() {
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
  // CHECK: llhd.sig %[[ZERO]] : f64
  %net = moore.net wire : <!moore.f64>
  return
}

// CHECK-LABEL: func.func @TimeWire
func.func @TimeWire() {
  // CHECK: %[[ZERO:.*]] = llhd.constant_time <0ns, 0d, 0e>
  // CHECK: llhd.sig %[[ZERO]] : !llhd.time
  %net = moore.net wire : <!moore.time>
  return
}

// CHECK-LABEL: func.func @RealWireWithAssignment
// CHECK-SAME: (%arg0: f64)
func.func @RealWireWithAssignment(%value: !moore.f64) {
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
  // CHECK: %[[SIG:.*]] = llhd.sig %[[ZERO]] : f64
  %net = moore.net wire %value : <!moore.f64>
  // CHECK: %[[DELAY:.*]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %[[SIG]], %arg0 after %[[DELAY]] : f64
  return
}

// CHECK-LABEL: func.func @TimeWireWithAssignment
// CHECK-SAME: (%arg0: !llhd.time)
func.func @TimeWireWithAssignment(%value: !moore.time) {
  // CHECK: %[[ZERO:.*]] = llhd.constant_time <0ns, 0d, 0e>
  // CHECK: %[[SIG:.*]] = llhd.sig %[[ZERO]] : !llhd.time
  %net = moore.net wire %value : <!moore.time>
  // CHECK: %[[DELAY:.*]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %[[SIG]], %arg0 after %[[DELAY]] : !llhd.time
  return
}
