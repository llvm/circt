// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-full-reset))' --split-input-file %s | FileCheck %s

// Combinational memories in asynchronous full-reset domains become registers.
// CHECK-LABEL: firrtl.circuit "AsyncDomainConvertsMem"
firrtl.circuit "AsyncDomainConvertsMem" {
  firrtl.module public @AsyncDomainConvertsMem(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    // CHECK-NOT: firrtl.mem
    // CHECK: firrtl.regreset
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

// -----
// Synchronous full-reset domains must not convert comb mems.
// CHECK-LABEL: firrtl.circuit "SyncDomainKeepsMem"
firrtl.circuit "SyncDomainKeepsMem" {
  firrtl.module public @SyncDomainKeepsMem(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.uint<1> [{class = "circt.FullResetAnnotation", resetType = "sync"}]) {
    // CHECK: firrtl.mem
    // CHECK-NOT: firrtl.reg
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

// -----
// ExcludeFromFullReset cuts the domain: child keeps its memory.
// CHECK-LABEL: firrtl.circuit "ExcludeKeepsMem"
firrtl.circuit "ExcludeKeepsMem" {
  // CHECK-LABEL: firrtl.module @Child
  // CHECK: firrtl.mem
  firrtl.module @Child(in %clock: !firrtl.clock) attributes {
    annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]
  } {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 4 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
  // CHECK-LABEL: firrtl.module @ExcludeKeepsMem
  // CHECK-NOT: firrtl.mem
  // CHECK: firrtl.regreset
  firrtl.module @ExcludeKeepsMem(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 4 : i64,
      name = "topmem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    firrtl.instance c @Child(in clock: !firrtl.clock)
  }
}

// -----
// No FullReset annotation: memory stays a memory (FullReset is a no-op).
// CHECK-LABEL: firrtl.circuit "NoAnnoKeepsMem"
firrtl.circuit "NoAnnoKeepsMem" {
  firrtl.module public @NoAnnoKeepsMem(in %clock: !firrtl.clock) {
    // CHECK: firrtl.mem
    // CHECK-NOT: firrtl.reg
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

// -----
// Nested children inherit the async domain and convert mems.
// CHECK-LABEL: firrtl.circuit "NestedAsyncConvertsMem"
firrtl.circuit "NestedAsyncConvertsMem" {
  // CHECK-LABEL: firrtl.module @Child
  // CHECK-NOT: firrtl.mem
  // CHECK: firrtl.regreset
  firrtl.module @Child(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 4 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
  firrtl.module @NestedAsyncConvertsMem(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %c_clock, %c_reset = firrtl.instance c @Child(in clock: !firrtl.clock, in reset: !firrtl.asyncreset)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_reset, %reset : !firrtl.asyncreset
  }
}
