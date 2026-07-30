// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-full-reset))' --split-input-file %s | FileCheck %s

// Basic async full-reset: reset-less register becomes regreset.
// CHECK-LABEL: firrtl.module @AsyncFullReset
firrtl.circuit "AsyncFullReset" {
  firrtl.module @AsyncFullReset(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset
          [{class = "circt.FullResetAnnotation", resetType = "async"}],
      in %in: !firrtl.uint<8>) {
    // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %reg, %in : !firrtl.uint<8>
  }
}

// -----
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
// Child inherits async domain; reset is wired through an added port.
// CHECK-LABEL: firrtl.module @Child
// CHECK-SAME: in %reset: !firrtl.asyncreset
// CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
// CHECK-LABEL: firrtl.module @Nested
// CHECK: firrtl.matchingconnect %child_reset, %reset
firrtl.circuit "Nested" {
  firrtl.module @Child(in %clock: !firrtl.clock) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  firrtl.module @Nested(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset
          [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %child_clock = firrtl.instance child @Child(in clock: !firrtl.clock)
    firrtl.matchingconnect %child_clock, %clock : !firrtl.clock
  }
}

// -----
// Comb mems in async full-reset domains become resettable registers.
// CHECK-LABEL: firrtl.module @AsyncDomainMem
// CHECK-NOT: firrtl.mem
// CHECK: firrtl.regreset
firrtl.circuit "AsyncDomainMem" {
  firrtl.module @AsyncDomainMem(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset
          [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
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
