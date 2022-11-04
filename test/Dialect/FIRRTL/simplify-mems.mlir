// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s


firrtl.circuit "ReadOnlyMemory" {
  // CHECK-LABEL: firrtl.module public @ReadOnlyMemory
  firrtl.module public @ReadOnlyMemory(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>) {

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %Memory_r = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>


    // CHECK-NOT: firrtl.mem
    // CHECK: firrtl.wire : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %2 = firrtl.subfield %Memory_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<4>
    firrtl.connect %2, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = firrtl.subfield %Memory_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<1>
    firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.subfield %Memory_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.clock
    firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
  }
}

firrtl.circuit "WriteOnlyMemory" {
  // CHECK-LABEL: firrtl.module public @WriteOnlyMemory
  firrtl.module public @WriteOnlyMemory(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>, in %indata: !firrtl.uint<42>) {

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %Memory_write = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    // CHECK-NOT: firrtl.mem
    // CHECK: firrtl.wire : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    %10 = firrtl.subfield %Memory_write(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<4>
    firrtl.connect %10, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %11 = firrtl.subfield %Memory_write(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %11, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %12 = firrtl.subfield %Memory_write(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.clock
    firrtl.connect %12, %clock : !firrtl.clock, !firrtl.clock
    %13 = firrtl.subfield %Memory_write(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<42>
    firrtl.connect %13, %indata : !firrtl.uint<42>, !firrtl.uint<42>
    %14 = firrtl.subfield %Memory_write(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %14, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

firrtl.circuit "ReadWriteToWrite" {
  firrtl.module public @ReadWriteToWrite(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>, in %indata: !firrtl.uint<42>, out %result: !firrtl.uint<42>) {

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_rw, %Memory_r = firrtl.mem  Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>

    %Memory_rw, %Memory_r = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["rw", "r"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>


    // CHECK: [[WIRE:%.+]] = firrtl.wire   : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    // CHECK: [[PORT_ADDR:%.+]] = firrtl.subfield %Memory_rw(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<4>
    // CHECK: [[GET_ADDR:%.+]] = firrtl.subfield [[WIRE]](0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<4>
    // CHECK: firrtl.strictconnect [[PORT_ADDR]], [[GET_ADDR]] : !firrtl.uint<4>
    // CHECK: [[PORT_EN:%.+]] = firrtl.subfield %Memory_rw(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    // CHECK: [[GET_EN:%.+]] = firrtl.subfield [[WIRE]](1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.strictconnect [[PORT_EN]], [[GET_EN]] : !firrtl.uint<1>
    // CHECK: [[PORT_CLK:%.+]] = firrtl.subfield %Memory_rw(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.clock
    // CHECK: [[GET_CLK:%.+]] = firrtl.subfield [[WIRE]](2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.clock
    // CHECK: firrtl.strictconnect [[PORT_CLK]], [[GET_CLK]] : !firrtl.clock
    // CHECK: [[PORT_DATA:%.+]] = firrtl.subfield %Memory_rw(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<42>
    // CHECK: [[GET_DATA:%.+]] = firrtl.subfield [[WIRE]](5) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<42>
    // CHECK: firrtl.strictconnect [[PORT_DATA]], [[GET_DATA]] : !firrtl.uint<42>
    // CHECK: [[PORT_MASK:%.+]] = firrtl.subfield %Memory_rw(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    // CHECK: [[GET_MASK:%.+]] = firrtl.subfield [[WIRE]](6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.strictconnect [[PORT_MASK]], [[GET_MASK]] : !firrtl.uint<1>
    // CHECK: [[SET_ADDR:%.+]] = firrtl.subfield [[WIRE]](0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<4>
    // CHECK: firrtl.strictconnect [[SET_ADDR]], %addr : !firrtl.uint<4>
    // CHECK: [[SET_EN:%.+]] = firrtl.subfield [[WIRE]](1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.strictconnect [[SET_EN]], %c1_ui1 : !firrtl.uint<1>
    // CHECK: [[SET_CLK:%.+]] = firrtl.subfield [[WIRE]](2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.clock
    // CHECK: firrtl.strictconnect [[SET_CLK]], %clock : !firrtl.clock
    // CHECK: [[SET_WMODE:%.+]] = firrtl.subfield [[WIRE]](4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.strictconnect [[SET_WMODE]], %c1_ui1 : !firrtl.uint<1>
    // CHECK: [[SET_WDATA:%.+]] = firrtl.subfield [[WIRE]](5) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<42>
    // CHECK: firrtl.strictconnect [[SET_WDATA]], %indata : !firrtl.uint<42>
    // CHECK: [[SET_WMASK:%.+]] = firrtl.subfield [[WIRE]](6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.strictconnect [[SET_WMASK]], %c1_ui1 : !firrtl.uint<1>

    %0 = firrtl.subfield %Memory_rw(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<4>
    firrtl.connect %0, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %1 = firrtl.subfield %Memory_rw(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %Memory_rw(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.clock
    firrtl.connect %2, %clock : !firrtl.clock, !firrtl.clock
    %3 = firrtl.subfield %Memory_rw(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.subfield %Memory_rw(5) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<42>
    firrtl.connect %4, %indata : !firrtl.uint<42>, !firrtl.uint<42>
    %5 = firrtl.subfield %Memory_rw(6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %5, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %6 = firrtl.subfield %Memory_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<4>
    firrtl.connect %6, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %7 = firrtl.subfield %Memory_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<1>
    firrtl.connect %7, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %8 = firrtl.subfield %Memory_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.clock
    firrtl.connect %8, %clock : !firrtl.clock, !firrtl.clock
    %9 = firrtl.subfield %Memory_r(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<42>
    firrtl.connect %result, %9 : !firrtl.uint<42>, !firrtl.uint<42>
  }
}

firrtl.circuit "UnusedPorts" {
  firrtl.module public @UnusedPorts(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<4>,
      in %in_data: !firrtl.uint<42>,
      in %wmode_rw: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<42>,
      out %result_rw: !firrtl.uint<42>,
      out %result_pinned: !firrtl.uint<42>) {

    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_pinned = firrtl.mem  Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    // CHECK: firrtl.wire   : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    // CHECK: firrtl.wire   : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    // CHECK: firrtl.wire   : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    %Memory_read, %Memory_rw, %Memory_write, %Memory_pinned = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read", "rw", "write", "pinned"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>


    %read_addr = firrtl.subfield %Memory_read(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<4>
    firrtl.connect %read_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %read_en = firrtl.subfield %Memory_read(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<1>
    firrtl.connect %read_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = firrtl.subfield %Memory_read(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.clock
    firrtl.connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = firrtl.subfield %Memory_read(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<42>
    firrtl.connect %result_read, %read_data : !firrtl.uint<42>, !firrtl.uint<42>

    %rw_addr = firrtl.subfield %Memory_rw(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<4>
    firrtl.connect %rw_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %rw_en = firrtl.subfield %Memory_rw(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %rw_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_clk = firrtl.subfield %Memory_rw(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.clock
    firrtl.connect %rw_clk, %clock : !firrtl.clock, !firrtl.clock
    %rw_rdata = firrtl.subfield %Memory_rw(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<42>
    firrtl.connect %result_rw, %rw_rdata : !firrtl.uint<42>, !firrtl.uint<42>
    %rw_wmode = firrtl.subfield %Memory_rw(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %rw_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_wdata = firrtl.subfield %Memory_rw(5) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<42>
    firrtl.connect %rw_wdata, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %rw_wmask = firrtl.subfield %Memory_rw(6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %rw_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %write_addr = firrtl.subfield %Memory_write(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<4>
    firrtl.connect %write_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %write_en = firrtl.subfield %Memory_write(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %write_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = firrtl.subfield %Memory_write(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.clock
    firrtl.connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = firrtl.subfield %Memory_write(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<42>
    firrtl.connect %write_data, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %write_mask = firrtl.subfield %Memory_write(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %pinned_addr = firrtl.subfield %Memory_pinned(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<4>
    firrtl.connect %pinned_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %pinned_en = firrtl.subfield %Memory_pinned(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %pinned_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %pinned_clk = firrtl.subfield %Memory_pinned(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.clock
    firrtl.connect %pinned_clk, %clock : !firrtl.clock, !firrtl.clock
    %pinned_rdata = firrtl.subfield %Memory_pinned(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<42>
    firrtl.connect %result_pinned, %pinned_rdata : !firrtl.uint<42>, !firrtl.uint<42>
    %pinned_wmode = firrtl.subfield %Memory_pinned(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %pinned_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %pinned_wdata = firrtl.subfield %Memory_pinned(5) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<42>
    firrtl.connect %pinned_wdata, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %pinned_wmask = firrtl.subfield %Memory_pinned(6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %pinned_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
