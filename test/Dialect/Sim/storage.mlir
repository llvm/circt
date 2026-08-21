// RUN: circt-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: hw.module @assoc_array_storage
hw.module @assoc_array_storage() {
  // CHECK: sv.reg : !hw.inout<!sim.assoc_array<i32, i32>>
  %0 = sv.reg : !hw.inout<!sim.assoc_array<i32, i32>>
}

// CHECK-LABEL: hw.module @queue_storage
hw.module @queue_storage() {
  // CHECK: sv.reg : !hw.inout<!sim.queue<i8, 0>>
  %0 = sv.reg : !hw.inout<!sim.queue<i8, 0>>
}

// CHECK-LABEL: hw.module @dstring_storage
hw.module @dstring_storage() {
  // CHECK: sv.reg : !hw.inout<!sim.dstring>
  %0 = sv.reg : !hw.inout<!sim.dstring>
}
