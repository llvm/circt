// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-full-reset))' --split-input-file %s | FileCheck %s

// Exclude annotation is consumed; registers stay reset-less.
// CHECK-LABEL: firrtl.module @Excluded
// CHECK-NOT: ExcludeFromFullResetAnnotation
// CHECK: %reg = firrtl.reg %clock
firrtl.circuit "Excluded" {
  firrtl.module @Excluded(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>)
      attributes {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %reg, %in : !firrtl.uint<8>
  }
}

// -----
// Sync full-reset domains keep comb mems.
// CHECK-LABEL: firrtl.module @SyncDomainMem
// CHECK: firrtl.mem
// CHECK-NOT: firrtl.reg
firrtl.circuit "SyncDomainMem" {
  firrtl.module @SyncDomainMem(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.uint<1>
          [{class = "circt.FullResetAnnotation", resetType = "sync"}]) {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 4 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}
