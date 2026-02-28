// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-infer-rw)))' %s | FileCheck %s

firrtl.circuit "TLRAM" {
// Test the case when the enable is a simple not of write enable.
// CHECK-LABEL: firrtl.module @TLRAM
    firrtl.module @TLRAM(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %index: !firrtl.uint<4>, in %index2: !firrtl.uint<4>, in %data_0: !firrtl.uint<8>, in %wen: !firrtl.uint<1>, in %_T_29: !firrtl.uint<1>, out %auto_0: !firrtl.uint<8>, out %dbg_0: !firrtl.probe<vector<uint<8>, 16>>) {
      %mem_MPORT_en = firrtl.wire  : !firrtl.uint<1>
      %mem_MPORT_data_0 = firrtl.wire  : !firrtl.uint<8>
      %debug, %mem_0_MPORT, %mem_0_MPORT_1 = firrtl.mem Undefined  {depth = 16 : i64, name = "mem_0", portNames = ["dbgs", "MPORT", "MPORT_1"], prefix = "foo_", readLatency = 1 : i32, writeLatency = 1 : i32} :  !firrtl.probe<vector<uint<8>, 16>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      firrtl.ref.define %dbg_0, %debug : !firrtl.probe<vector<uint<8>, 16>>
      %0 = firrtl.subfield %mem_0_MPORT[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      firrtl.connect %0, %index2 : !firrtl.uint<4>, !firrtl.uint<4>
      %1 = firrtl.subfield %mem_0_MPORT[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      firrtl.connect %1, %mem_MPORT_en : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %mem_0_MPORT[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      firrtl.connect %2, %clock : !firrtl.clock, !firrtl.clock
      %3 = firrtl.subfield %mem_0_MPORT[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      firrtl.connect %mem_MPORT_data_0, %3 : !firrtl.uint<8>, !firrtl.uint<8>
      %4 = firrtl.subfield %mem_0_MPORT_1[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      firrtl.connect %4, %index : !firrtl.uint<4>, !firrtl.uint<4>
      %5 = firrtl.subfield %mem_0_MPORT_1[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      firrtl.connect %5, %wen : !firrtl.uint<1>, !firrtl.uint<1>
      %6 = firrtl.subfield %mem_0_MPORT_1[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      firrtl.connect %6, %clock : !firrtl.clock, !firrtl.clock
      %7 = firrtl.subfield %mem_0_MPORT_1[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      firrtl.connect %7, %data_0 : !firrtl.uint<8>, !firrtl.uint<8>
      %8 = firrtl.subfield %mem_0_MPORT_1[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      firrtl.connect %8, %_T_29 : !firrtl.uint<1>, !firrtl.uint<1>
      %9 = firrtl.not %wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %mem_MPORT_en, %9 : !firrtl.uint<1>, !firrtl.uint<1>
      %REG = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
      firrtl.connect %REG, %9 : !firrtl.uint<1>, !firrtl.uint<1>
      %r_0 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<8>
      %10 = firrtl.mux(%REG, %mem_MPORT_data_0, %r_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
      firrtl.connect %r_0, %10 : !firrtl.uint<8>, !firrtl.uint<8>
      %11 = firrtl.mux(%REG, %mem_MPORT_data_0, %r_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
      firrtl.connect %auto_0, %11 : !firrtl.uint<8>, !firrtl.uint<8>

// CHECK: %mem_0_dbgs, %mem_0_rw = firrtl.mem  Undefined  {depth = 16 : i64, name = "mem_0", portNames = ["dbgs", "rw"], prefix = "foo_", readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<8>, 16>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>
// CHECK:  %[[v7:.+]] = firrtl.mux(%[[writeEnable:.+]], %[[writeAddr:.+]], %[[readAddr:.+]]) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK:  firrtl.matchingconnect %[[v0:.+]], %[[v7]] : !firrtl.uint<4>
// CHECK:  %[[v8:.+]] = firrtl.or %[[readEnable:.+]], %[[writeEnable]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.matchingconnect %[[v1:.+]], %[[v8]] : !firrtl.uint<1>
// CHECK:  firrtl.ref.define %dbg_0, %mem_0_dbgs : !firrtl.probe<vector<uint<8>, 16>>
// CHECK:  firrtl.connect %[[readAddr]], %[[index2:.+]] : !firrtl.uint<4>
// CHECK:  firrtl.connect %[[readEnable]], %mem_MPORT_en : !firrtl.uint<1>
// CHECK:  firrtl.connect %[[writeAddr]], %index : !firrtl.uint<4>
// CHECK:  firrtl.connect %[[writeEnable]], %wen : !firrtl.uint<1>
// CHECK:  %[[v10:.+]] = firrtl.not %wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %mem_MPORT_en, %[[v10]] : !firrtl.uint<1>
// CHECK:  firrtl.matchingconnect %[[v4:.+]], %wen : !firrtl.uint<1>
    }

// Test the pattern of enable  with Mux (sel, high, 0)
// CHECK-LABEL: firrtl.module @memTest4t
  firrtl.module @memTest4t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<11>, in %io_ren: !firrtl.uint<1>, in %io_wen: !firrtl.uint<1>, in %io_dataIn: !firrtl.uint<32>, out %io_dataOut: !firrtl.uint<32>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %mem__T_14, %mem__T_22 = firrtl.mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
// CHECK:    %mem_rw = firrtl.mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    %0 = firrtl.subfield %mem__T_14[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %1 = firrtl.subfield %mem__T_14[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %2 = firrtl.subfield %mem__T_14[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %3 = firrtl.subfield %mem__T_14[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %4 = firrtl.subfield %mem__T_14[mask] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = firrtl.subfield %mem__T_22[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %6 = firrtl.subfield %mem__T_22[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %7 = firrtl.subfield %mem__T_22[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %8 = firrtl.subfield %mem__T_22[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %0, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.connect %1, %io_wen : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %2, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %4, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %3, %io_dataIn : !firrtl.uint<32>, !firrtl.uint<32>
    %9 = firrtl.not %io_wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = firrtl.mux(%9, %io_ren, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %6, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %5, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.connect %7, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %io_dataOut, %8 : !firrtl.uint<32>, !firrtl.uint<32>
    // CHECK:   firrtl.matchingconnect %4, %io_wen : !firrtl.uint<1>
  }

// Test the pattern of enable  with an And tree and Mux (sel, high, 0)
// CHECK-LABEL: firrtl.module @memTest6t
  firrtl.module @memTest6t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<11>, in %io_valid: !firrtl.uint<1>, in %io_write: !firrtl.uint<1>, in %io_dataIn: !firrtl.uint<32>, out %io_dataOut: !firrtl.uint<32>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %mem__T_14, %mem__T_22 = firrtl.mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
// CHECK: %mem_rw = firrtl.mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
// CHECK:   %[[v7:.+]] = firrtl.mux(%writeEnable, %writeAddr, %readAddr) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<11>) -> !firrtl.uint<11>
// CHECK:   firrtl.matchingconnect %[[v0:.+]], %[[v7]]
// CHECK:   %[[v8:.+]] = firrtl.or %readEnable, %writeEnable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %0 = firrtl.subfield %mem__T_14[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %1 = firrtl.subfield %mem__T_14[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %2 = firrtl.subfield %mem__T_14[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %3 = firrtl.subfield %mem__T_14[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %4 = firrtl.subfield %mem__T_14[mask] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = firrtl.subfield %mem__T_22[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %6 = firrtl.subfield %mem__T_22[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %7 = firrtl.subfield %mem__T_22[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %8 = firrtl.subfield %mem__T_22[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %9 = firrtl.and %io_valid, %io_write : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = firrtl.not %io_write : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %11 = firrtl.and %io_valid, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %0, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.connect %1, %9 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %2, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %4, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %3, %io_dataIn : !firrtl.uint<32>, !firrtl.uint<32>
    %12 = firrtl.not %9 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %13 = firrtl.mux(%12, %11, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %6, %13 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %5, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.connect %7, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %io_dataOut, %8 : !firrtl.uint<32>, !firrtl.uint<32>
    // CHECK:  firrtl.matchingconnect %4, %io_write : !firrtl.uint<1>
  }

// Cannot merge read and write, since the pattern is enable = Mux (sel, high, 1)
// CHECK-LABEL: firrtl.module @memTest7t
  firrtl.module @memTest7t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<11>, in %io_valid: !firrtl.uint<1>, in %io_write: !firrtl.uint<1>, in %io_dataIn: !firrtl.uint<32>, out %io_dataOut: !firrtl.uint<32>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %mem__T_14, %mem__T_22 = firrtl.mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
// CHECK: %mem__T_14, %mem__T_22 = firrtl.mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %0 = firrtl.subfield %mem__T_14[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %1 = firrtl.subfield %mem__T_14[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %2 = firrtl.subfield %mem__T_14[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %3 = firrtl.subfield %mem__T_14[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %4 = firrtl.subfield %mem__T_14[mask] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = firrtl.subfield %mem__T_22[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %6 = firrtl.subfield %mem__T_22[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %7 = firrtl.subfield %mem__T_22[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %8 = firrtl.subfield %mem__T_22[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %9 = firrtl.and %io_valid, %io_write : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = firrtl.not %io_write : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %11 = firrtl.and %io_valid, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %0, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.connect %1, %9 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %2, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %4, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %3, %io_dataIn : !firrtl.uint<32>, !firrtl.uint<32>
    %12 = firrtl.not %9 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %13 = firrtl.mux(%12, %11, %c1_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %6, %13 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %5, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.connect %7, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %io_dataOut, %8 : !firrtl.uint<32>, !firrtl.uint<32>
  }

// Cannot merge, since the clocks are different.
// CHECK-LABEL: firrtl.module @memTest5t
  firrtl.module @memTest5t(in %clk1: !firrtl.clock, in %clk2: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_en: !firrtl.uint<1>, in %io_wen: !firrtl.uint<1>, in %io_waddr: !firrtl.uint<8>, in %io_wdata: !firrtl.uint<32>, in %io_raddr: !firrtl.uint<8>, out %io_rdata: !firrtl.uint<32>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %mem_T_3, %mem_T_5 = firrtl.mem Undefined  {depth = 128 : i64, name = "mem", portNames = ["T_3", "T_5"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    // CHECK:    %mem_T_3, %mem_T_5 = firrtl.mem Undefined  {depth = 128 : i64, name = "mem", portNames = ["T_3", "T_5"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %0 = firrtl.subfield %mem_T_3[addr] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %1 = firrtl.subfield %mem_T_3[en] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %2 = firrtl.subfield %mem_T_3[clk] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %3 = firrtl.subfield %mem_T_3[data] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %4 = firrtl.subfield %mem_T_5[addr] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = firrtl.subfield %mem_T_5[en] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %6 = firrtl.subfield %mem_T_5[clk] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %7 = firrtl.subfield %mem_T_5[data] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %8 = firrtl.subfield %mem_T_5[mask] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %9 = firrtl.not %io_wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = firrtl.and %io_en, %9 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %1, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    %11 = firrtl.bits %io_raddr 6 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<7>
    firrtl.connect %0, %11 : !firrtl.uint<7>, !firrtl.uint<7>
    firrtl.connect %2, %clk1 : !firrtl.clock, !firrtl.clock
    firrtl.connect %io_rdata, %3 : !firrtl.uint<32>, !firrtl.uint<32>
    %12 = firrtl.and %io_en, %io_wen : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %13 = firrtl.bits %io_waddr 6 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<7>
    firrtl.connect %4, %13 : !firrtl.uint<7>, !firrtl.uint<7>
    firrtl.connect %5, %12 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %6, %clk2 : !firrtl.clock, !firrtl.clock
    firrtl.connect %8, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %7, %io_wdata : !firrtl.uint<32>, !firrtl.uint<32>
  }

// Check for a complement term in the And expression tree.
// CHECK-LABEL: firrtl.module @memTest3t
  firrtl.module @memTest3t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_en: !firrtl.uint<1>, in %io_wen: !firrtl.uint<1>, in %io_waddr: !firrtl.uint<8>, in %io_wdata: !firrtl.uint<32>, in %io_raddr: !firrtl.uint<8>, out %io_rdata: !firrtl.uint<32>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %dbg0, %mem_T_3, %mem_T_5, %dbg = firrtl.mem Undefined  {depth = 128 : i64, name = "mem", portNames = ["dbg0", "T_3", "T_5", "dbg"], readLatency = 1 : i32, writeLatency = 1 : i32} :  !firrtl.probe<vector<uint<32>,128>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.probe<vector<uint<32>,128>>
// CHECK: %mem_dbg0, %mem_dbg, %mem_rw = firrtl.mem  Undefined  {depth = 128 : i64, name = "mem", portNames = ["dbg0", "dbg", "rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<32>, 128>>, !firrtl.probe<vector<uint<32>, 128>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    %0 = firrtl.subfield %mem_T_3[addr] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %1 = firrtl.subfield %mem_T_3[en] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %2 = firrtl.subfield %mem_T_3[clk] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %3 = firrtl.subfield %mem_T_3[data] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %4 = firrtl.subfield %mem_T_5[addr] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = firrtl.subfield %mem_T_5[en] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %6 = firrtl.subfield %mem_T_5[clk] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %7 = firrtl.subfield %mem_T_5[data] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %8 = firrtl.subfield %mem_T_5[mask] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %9 = firrtl.not %io_wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = firrtl.and %io_en, %9 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %1, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    %11 = firrtl.bits %io_raddr 6 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<7>
    firrtl.connect %0, %11 : !firrtl.uint<7>, !firrtl.uint<7>
    firrtl.connect %2, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %io_rdata, %3 : !firrtl.uint<32>, !firrtl.uint<32>
    %12 = firrtl.and %io_en, %io_wen : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %13 = firrtl.bits %io_waddr 6 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<7>
    firrtl.connect %4, %13 : !firrtl.uint<7>, !firrtl.uint<7>
    firrtl.connect %5, %12 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %6, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %8, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %7, %io_wdata : !firrtl.uint<32>, !firrtl.uint<32>
    // CHECK:  firrtl.matchingconnect %4, %io_wen : !firrtl.uint<1>
  }

// Check for indirect connection to clock
// CHECK-LABEL: firrtl.module @memTest2t
  firrtl.module @memTest2t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<11>, in %io_ren: !firrtl.uint<1>, in %io_wen: !firrtl.uint<1>, in %io_dataIn: !firrtl.uint<32>, out %io_dataOut: !firrtl.uint<32>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %mem__T_14, %mem__T_22 = firrtl.mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
// CHECK:    %mem_rw = firrtl.mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    %0 = firrtl.subfield %mem__T_14[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %1 = firrtl.subfield %mem__T_14[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %2 = firrtl.subfield %mem__T_14[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %3 = firrtl.subfield %mem__T_14[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %4 = firrtl.subfield %mem__T_14[mask] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = firrtl.subfield %mem__T_22[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %6 = firrtl.subfield %mem__T_22[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %7 = firrtl.subfield %mem__T_22[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %8 = firrtl.subfield %mem__T_22[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %0, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.connect %1, %io_wen : !firrtl.uint<1>, !firrtl.uint<1>
    %xc = firrtl.wire : !firrtl.clock
    firrtl.connect %2, %xc : !firrtl.clock, !firrtl.clock
    firrtl.connect %xc, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %4, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %3, %io_dataIn : !firrtl.uint<32>, !firrtl.uint<32>
    %9 = firrtl.not %io_wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = firrtl.mux(%9, %io_ren, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %6, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %5, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.connect %7, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %io_dataOut, %8 : !firrtl.uint<32>, !firrtl.uint<32>
  }

    firrtl.module @constMask(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %index: !firrtl.uint<4>, in %index2: !firrtl.uint<4>, in %data_0: !firrtl.uint<8>, in %wen: !firrtl.uint<1>, in %_T_29: !firrtl.uint<1>, out %auto_0: !firrtl.uint<8>) {
      %mem_MPORT_en = firrtl.wire  : !firrtl.uint<1>
      %mem_MPORT_data_0 = firrtl.wire  : !firrtl.uint<8>
      %mem_0_MPORT, %mem_0_MPORT_1 = firrtl.mem Undefined  {depth = 16 : i64, name = "mem_0", portNames = ["MPORT", "MPORT_1"], prefix = "Foo_", readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>

      // CHECK: %mem_0_rw = firrtl.mem Undefined  {depth = 16 : i64, name = "mem_0", portNames = ["rw"], prefix = "Foo_", readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>
      // CHECK: %[[v6:.+]] = firrtl.subfield %mem_0_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>
      // CHECK: %[[c1_ui1:.+]] = firrtl.constant 1 : !firrtl.uint<1>
      // CHECK: firrtl.matchingconnect %[[v6]], %[[c1_ui1]]
      %0 = firrtl.subfield %mem_0_MPORT[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      firrtl.connect %0, %index2 : !firrtl.uint<4>, !firrtl.uint<4>
      %1 = firrtl.subfield %mem_0_MPORT[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      firrtl.connect %1, %mem_MPORT_en : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %mem_0_MPORT[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      firrtl.connect %2, %clock : !firrtl.clock, !firrtl.clock
      %3 = firrtl.subfield %mem_0_MPORT[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      firrtl.connect %mem_MPORT_data_0, %3 : !firrtl.uint<8>, !firrtl.uint<8>
      %4 = firrtl.subfield %mem_0_MPORT_1[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      firrtl.connect %4, %index : !firrtl.uint<4>, !firrtl.uint<4>
      %5 = firrtl.subfield %mem_0_MPORT_1[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      firrtl.connect %5, %wen : !firrtl.uint<1>, !firrtl.uint<1>
      %6 = firrtl.subfield %mem_0_MPORT_1[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      firrtl.connect %6, %clock : !firrtl.clock, !firrtl.clock
      %7 = firrtl.subfield %mem_0_MPORT_1[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      firrtl.connect %7, %data_0 : !firrtl.uint<8>, !firrtl.uint<8>
      %8 = firrtl.subfield %mem_0_MPORT_1[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      %c1 = firrtl.constant 3 : !firrtl.uint<2>
      firrtl.connect %8, %c1 : !firrtl.uint<2>, !firrtl.uint<2>
      %9 = firrtl.not %wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %mem_MPORT_en, %9 : !firrtl.uint<1>, !firrtl.uint<1>
      %REG = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
      firrtl.connect %REG, %9 : !firrtl.uint<1>, !firrtl.uint<1>
      %r_0 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<8>
      %10 = firrtl.mux(%REG, %mem_MPORT_data_0, %r_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
      firrtl.connect %r_0, %10 : !firrtl.uint<8>, !firrtl.uint<8>
      %11 = firrtl.mux(%REG, %mem_MPORT_data_0, %r_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
      firrtl.connect %auto_0, %11 : !firrtl.uint<8>, !firrtl.uint<8>

    }

    // CHECK-LABEL: firrtl.module @SimplifyWMODE
    firrtl.module @SimplifyWMODE(in %rwPort_enable: !firrtl.uint<1>, in %rwPort_isWrite: !firrtl.uint<1>) attributes {} {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      %18 = firrtl.mux(%rwPort_enable, %rwPort_isWrite, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK: %[[c1_ui1:.+]] = firrtl.constant 1 : !firrtl.uint<1>
      // CHECK: %[[v7:.+]] = firrtl.mux(%[[c1_ui1]], %rwPort_isWrite, %c0_ui1)
      %mem_rwPort_readData_rw = firrtl.mem Undefined {depth = 64 : i64, name = "t", portNames = ["rw"], prefix = "", readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
      %mem_rwPort_readData_rw_wmode = firrtl.wire : !firrtl.uint<1>
      %0 = firrtl.subfield %mem_rwPort_readData_rw[addr] : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
      %1 = firrtl.subfield %mem_rwPort_readData_rw[en] : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
      firrtl.matchingconnect %1, %rwPort_enable : !firrtl.uint<1>
      %2 = firrtl.subfield %mem_rwPort_readData_rw[clk] : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
      %3 = firrtl.subfield %mem_rwPort_readData_rw[rdata] : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
      %6 = firrtl.subfield %mem_rwPort_readData_rw[wmode] : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
      firrtl.matchingconnect %6, %mem_rwPort_readData_rw_wmode : !firrtl.uint<1>
      %7 = firrtl.subfield %mem_rwPort_readData_rw[wdata] : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
      %9 = firrtl.subfield %mem_rwPort_readData_rw[wmask] : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
      firrtl.matchingconnect %mem_rwPort_readData_rw_wmode, %18 : !firrtl.uint<1>
    }

  // CHECK: firrtl.module @InferUnmasked
  firrtl.module @InferUnmasked(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>} {
    %readwritePortA_isWrite_2 = firrtl.wire {name = "readwritePortA_isWrite"} : !firrtl.uint<1>
    %syncreadmem_singleport_readwritePortA_readData_rw = firrtl.mem Undefined {depth = 64 : i64, name = "syncreadmem_singleport", portNames = ["readwritePortA_readData_rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
    // CHECK:  %syncreadmem_singleport_readwritePortA_readData_rw = firrtl.mem Undefined {depth = 64 : i64, name = "syncreadmem_singleport", portNames = ["readwritePortA_readData_rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<1>>
    %syncreadmem_singleport_readwritePortA_readData_rw_wmask_x = firrtl.wire : !firrtl.uint<1>
    %syncreadmem_singleport_readwritePortA_readData_rw_wmask_y = firrtl.wire : !firrtl.uint<1>
    %9 = firrtl.subfield %syncreadmem_singleport_readwritePortA_readData_rw[wmask] : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<5>>
    %10 = firrtl.cat %syncreadmem_singleport_readwritePortA_readData_rw_wmask_y, %syncreadmem_singleport_readwritePortA_readData_rw_wmask_x : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %11 = firrtl.bits %10 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    %12 = firrtl.cat %11, %11 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %13 = firrtl.cat %11, %12 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
    %14 = firrtl.bits %10 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    %15 = firrtl.cat %14, %13 : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<4>
    %16 = firrtl.cat %14, %15 : (!firrtl.uint<1>, !firrtl.uint<4>) -> !firrtl.uint<5>
    firrtl.matchingconnect %9, %16 : !firrtl.uint<5>
    firrtl.connect %syncreadmem_singleport_readwritePortA_readData_rw_wmask_x, %readwritePortA_isWrite_2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %syncreadmem_singleport_readwritePortA_readData_rw_wmask_y, %readwritePortA_isWrite_2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Test that this pass walks into layers.
  // CHECK-LABEL: firrtl.module @Layers
  firrtl.layer @A bind {}
  firrtl.module @Layers() {
    firrtl.layerblock @A {
      %mem_w = firrtl.mem Undefined {
        depth = 2 : i64,
        name = "mem",
        portNames = ["w"],
        readLatency = 1 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<2>, mask: uint<2>>
      // CHECK:      %mem_w = firrtl.mem
      // CHECK-SAME:   mask: uint<1>
      %0 = firrtl.subfield %mem_w[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<2>, mask: uint<2>>
      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
      %1 = firrtl.cat %c1_ui1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
      firrtl.matchingconnect %0, %1 : !firrtl.uint<2>
    }
  }

}
