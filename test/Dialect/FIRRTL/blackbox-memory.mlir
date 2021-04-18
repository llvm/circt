// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-blackbox-memory{emit-wrapper=true})' %s | FileCheck --check-prefix=WRAPPER %s
// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-blackbox-memory{emit-wrapper=false})' %s | FileCheck --check-prefix=INLINE %s

firrtl.circuit "Read" {
  firrtl.module @Read() {
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>

    %0 = firrtl.mem Undefined {depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>

    %1 = firrtl.subfield %0("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.sint<8>
    %2 = firrtl.subfield %0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %2, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
    %3 = firrtl.subfield %0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %3, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %4 = firrtl.subfield %0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<clock>
  }
}

// WRAPPER-LABEL: firrtl.circuit "Read" {
// WRAPPER-NEXT:   firrtl.extmodule @ReadMemory_ext(!firrtl.uint<4> {firrtl.name = "R0_addr"}, !firrtl.uint<1> {firrtl.name = "R0_en"}, !firrtl.clock {firrtl.name = "R0_clk"}, !firrtl.flip<sint<8>> {firrtl.name = "R0_data"}) attributes {depth = 16 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// WRAPPER-NEXT:   firrtl.module @ReadMemory(%read0: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) {
// WRAPPER-NEXT:     %ReadMemory_R0_addr, %ReadMemory_R0_en, %ReadMemory_R0_clk, %ReadMemory_R0_data = firrtl.instance @ReadMemory_ext {name = "ReadMemory", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<8>
// WRAPPER-NEXT:     %0 = firrtl.subfield %read0("addr") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.uint<4>
// WRAPPER-NEXT:     firrtl.connect %ReadMemory_R0_addr, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
// WRAPPER-NEXT:     %1 = firrtl.subfield %read0("en") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %ReadMemory_R0_en, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %read0("clk") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %ReadMemory_R0_clk, %2 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %3 = firrtl.subfield %read0("data") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.flip<sint<8>>
// WRAPPER-NEXT:     firrtl.connect %3, %ReadMemory_R0_data : !firrtl.flip<sint<8>>, !firrtl.sint<8>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT:   firrtl.module @Read() {
// WRAPPER-NEXT:     %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
// WRAPPER-NEXT:     %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
// WRAPPER-NEXT:     %ReadMemory_read0 = firrtl.instance @ReadMemory {name = "ReadMemory", portNames = ["read0"]} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// WRAPPER-NEXT:     %0 = firrtl.subfield %ReadMemory_read0("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.sint<8>
// WRAPPER-NEXT:     %1 = firrtl.subfield %ReadMemory_read0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<4>>
// WRAPPER-NEXT:     firrtl.connect %1, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %ReadMemory_read0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<1>>
// WRAPPER-NEXT:     firrtl.connect %2, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %3 = firrtl.subfield %ReadMemory_read0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<clock>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT: }

// INLINE-LABEL: firrtl.circuit "Read" {
// INLINE-NEXT:   firrtl.extmodule @ReadMemory_ext(!firrtl.uint<4> {firrtl.name = "R0_addr"}, !firrtl.uint<1> {firrtl.name = "R0_en"}, !firrtl.clock {firrtl.name = "R0_clk"}, !firrtl.flip<sint<8>> {firrtl.name = "R0_data"}) attributes {depth = 16 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// INLINE-NEXT:   firrtl.module @Read() {
// INLINE-NEXT:     %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
// INLINE-NEXT:     %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
// INLINE-NEXT:     %ReadMemory_R0_addr, %ReadMemory_R0_en, %ReadMemory_R0_clk, %ReadMemory_R0_data = firrtl.instance @ReadMemory_ext {name = "ReadMemory", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<8>
// INLINE-NEXT:     %0 = firrtl.wire : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// INLINE-NEXT:     %1 = firrtl.subfield %0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %ReadMemory_R0_addr, %1 : !firrtl.flip<uint<4>>, !firrtl.flip<uint<4>>
// INLINE-NEXT:     %2 = firrtl.subfield %0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %ReadMemory_R0_en, %2 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %3 = firrtl.subfield %0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %ReadMemory_R0_clk, %3 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %4 = firrtl.subfield %0("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.sint<8>
// INLINE-NEXT:     firrtl.connect %4, %ReadMemory_R0_data : !firrtl.sint<8>, !firrtl.sint<8>
// INLINE-NEXT:     %5 = firrtl.subfield %0("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.sint<8>
// INLINE-NEXT:     %6 = firrtl.subfield %0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %6, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
// INLINE-NEXT:     %7 = firrtl.subfield %0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %7, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// INLINE-NEXT:     %8 = firrtl.subfield %0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<clock>
// INLINE-NEXT:   }
// INLINE-NEXT: }

firrtl.circuit "Write" {
  firrtl.module @Write() {
    %0 = firrtl.mem Undefined {depth = 1 : i64, name = "WriteMemory", portNames = ["write0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
  }
}

// WRAPPER-LABEL: firrtl.circuit "Write" {
// WRAPPER-NEXT:   firrtl.extmodule @WriteMemory_ext(!firrtl.uint<1> {firrtl.name = "W0_addr"}, !firrtl.uint<1> {firrtl.name = "W0_en"}, !firrtl.clock {firrtl.name = "W0_clk"}, !firrtl.sint<8> {firrtl.name = "W0_data"}, !firrtl.uint<1> {firrtl.name = "W0_mask"}) attributes {depth = 1 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// WRAPPER-NEXT:   firrtl.module @WriteMemory(%write0: !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) {
// WRAPPER-NEXT:     %WriteMemory_W0_addr, %WriteMemory_W0_en, %WriteMemory_W0_clk, %WriteMemory_W0_data, %WriteMemory_W0_mask = firrtl.instance @WriteMemory_ext {name = "WriteMemory", portNames = ["W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<8>>, !firrtl.flip<uint<1>>
// WRAPPER-NEXT:     %0 = firrtl.subfield %write0("addr") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_addr, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %1 = firrtl.subfield %write0("en") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_en, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %write0("clk") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_clk, %2 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %3 = firrtl.subfield %write0("data") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.sint<8>
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_data, %3 : !firrtl.flip<sint<8>>, !firrtl.sint<8>
// WRAPPER-NEXT:     %4 = firrtl.subfield %write0("mask") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_mask, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT:   firrtl.module @Write() {
// WRAPPER-NEXT:     %WriteMemory_write0 = firrtl.instance @WriteMemory {name = "WriteMemory", portNames = ["write0"]} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT: }

// INLINE-LABEL: firrtl.circuit "Write" {
// INLINE-NEXT:   firrtl.extmodule @WriteMemory_ext(!firrtl.uint<1> {firrtl.name = "W0_addr"}, !firrtl.uint<1> {firrtl.name = "W0_en"}, !firrtl.clock {firrtl.name = "W0_clk"}, !firrtl.sint<8> {firrtl.name = "W0_data"}, !firrtl.uint<1> {firrtl.name = "W0_mask"}) attributes {depth = 1 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// INLINE-NEXT:   firrtl.module @Write() {
// INLINE-NEXT:     %WriteMemory_W0_addr, %WriteMemory_W0_en, %WriteMemory_W0_clk, %WriteMemory_W0_data, %WriteMemory_W0_mask = firrtl.instance @WriteMemory_ext {name = "WriteMemory", portNames = ["W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<8>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %0 = firrtl.wire  : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// INLINE-NEXT:     %1 = firrtl.subfield %0("addr") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_addr, %1 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %2 = firrtl.subfield %0("en") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_en, %2 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %3 = firrtl.subfield %0("clk") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_clk, %3 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %4 = firrtl.subfield %0("data") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<sint<8>>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_data, %4 : !firrtl.flip<sint<8>>, !firrtl.flip<sint<8>>
// INLINE-NEXT:     %5 = firrtl.subfield %0("mask") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_mask, %5 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:   }
// INLINE-NEXT: }

// generated from:
// circuit MemSimple :
//     module MemSimple :
//        input clock1  : Clock
//        input clock2  : Clock
//        input inpred  : UInt<1>
//        input indata  : SInt<42>
//        output result : SInt<42>
// 
//        mem _M : @[Decoupled.scala 209:27]
//              data-type => SInt<42>
//              depth => 12
//              read-latency => 0
//              write-latency => 1
//              reader => read
//              writer => write
//              read-under-write => undefined
// 
//        result <= _M.read.data
// 
//        _M.read.addr <= UInt<1>("h0")
//        _M.read.en <= UInt<1>("h1")
//        _M.read.clk <= clock1
//        _M.write.addr <= validif(inpred, UInt<3>("h0"))
//        _M.write.en <= mux(inpred, UInt<1>("h1"), UInt<1>("h0"))
//        _M.write.clk <= validif(inpred, clock2)
//        _M.write.data <= validif(inpred, indata)
//        _M.write.mask <= validif(inpred, UInt<1>("h1"))

firrtl.circuit "MemSimple" {
  firrtl.module @MemSimple(%clock1: !firrtl.clock, %clock2: !firrtl.clock, %inpred: !firrtl.uint<1>, %indata: !firrtl.sint<42>, %result: !firrtl.flip<sint<42>>) {
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
    %c0_ui3 = firrtl.constant(0 : ui3) : !firrtl.uint<3>
    %_M_read, %_M_write = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>, !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>
    %0 = firrtl.subfield %_M_read("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.sint<42>
    firrtl.connect %result, %0 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
    %1 = firrtl.subfield %_M_read("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %1, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
    %2 = firrtl.subfield %_M_read("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %2, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %3 = firrtl.subfield %_M_read("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<clock>
    firrtl.connect %3, %clock1 : !firrtl.flip<clock>, !firrtl.clock
    %4 = firrtl.subfield %_M_write("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<4>>
    %5 = firrtl.invalidvalue : !firrtl.uint<3>
    %6 = firrtl.mux(%inpred, %c0_ui3, %5) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %4, %6 : !firrtl.flip<uint<4>>, !firrtl.uint<3>
    %7 = firrtl.subfield %_M_write("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
    %8 = firrtl.mux(%inpred, %c1_ui1, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %7, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %9 = firrtl.subfield %_M_write("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<clock>
    %10 = firrtl.invalidvalue : !firrtl.clock
    %11 = firrtl.mux(%inpred, %clock2, %10) : (!firrtl.uint<1>, !firrtl.clock, !firrtl.clock) -> !firrtl.clock
    firrtl.connect %9, %11 : !firrtl.flip<clock>, !firrtl.clock
    %12 = firrtl.subfield %_M_write("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<sint<42>>
    %13 = firrtl.invalidvalue : !firrtl.sint<42>
    %14 = firrtl.mux(%inpred, %indata, %13) : (!firrtl.uint<1>, !firrtl.sint<42>, !firrtl.sint<42>) -> !firrtl.sint<42>
    firrtl.connect %12, %14 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
    %15 = firrtl.subfield %_M_write("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
    %16 = firrtl.invalidvalue : !firrtl.uint<1>
    %17 = firrtl.mux(%inpred, %c1_ui1, %16) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %15, %17 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  }
}

// WRAPPER-LABEL: firrtl.circuit "MemSimple" {
// WRAPPER-NEXT:   firrtl.extmodule @_M_ext(!firrtl.uint<4> {firrtl.name = "R0_addr"}, !firrtl.uint<1> {firrtl.name = "R0_en"}, !firrtl.clock {firrtl.name = "R0_clk"}, !firrtl.flip<sint<42>> {firrtl.name = "R0_data"}, !firrtl.uint<4> {firrtl.name = "W0_addr"}, !firrtl.uint<1> {firrtl.name = "W0_en"}, !firrtl.clock {firrtl.name = "W0_clk"}, !firrtl.sint<42> {firrtl.name = "W0_data"}, !firrtl.uint<1> {firrtl.name = "W0_mask"}) attributes {depth = 12 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// WRAPPER-NEXT:   firrtl.module @_M(%read: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<42>>>, %write: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) {
// WRAPPER-NEXT:     %_M_R0_addr, %_M_R0_en, %_M_R0_clk, %_M_R0_data, %_M_W0_addr, %_M_W0_en, %_M_W0_clk, %_M_W0_data, %_M_W0_mask = firrtl.instance @_M_ext {name = "_M", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data", "W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<42>, !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<42>>, !firrtl.flip<uint<1>>
// WRAPPER-NEXT:     %0 = firrtl.subfield %read("addr") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<42>>>) -> !firrtl.uint<4>
// WRAPPER-NEXT:     firrtl.connect %_M_R0_addr, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
// WRAPPER-NEXT:     %1 = firrtl.subfield %read("en") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<42>>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %_M_R0_en, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %read("clk") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<42>>>) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %_M_R0_clk, %2 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %3 = firrtl.subfield %read("data") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<42>>>) -> !firrtl.flip<sint<42>>
// WRAPPER-NEXT:     firrtl.connect %3, %_M_R0_data : !firrtl.flip<sint<42>>, !firrtl.sint<42>
// WRAPPER-NEXT:     %4 = firrtl.subfield %write("addr") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<4>
// WRAPPER-NEXT:     firrtl.connect %_M_W0_addr, %4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
// WRAPPER-NEXT:     %5 = firrtl.subfield %write("en") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %_M_W0_en, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %6 = firrtl.subfield %write("clk") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %_M_W0_clk, %6 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %7 = firrtl.subfield %write("data") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.sint<42>
// WRAPPER-NEXT:     firrtl.connect %_M_W0_data, %7 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
// WRAPPER-NEXT:     %8 = firrtl.subfield %write("mask") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %_M_W0_mask, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT:   firrtl.module @MemSimple(%clock1: !firrtl.clock, %clock2: !firrtl.clock, %inpred: !firrtl.uint<1>, %indata: !firrtl.sint<42>, %result: !firrtl.flip<sint<42>>) {
// WRAPPER-NEXT:     %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
// WRAPPER-NEXT:     %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
// WRAPPER-NEXT:     %c0_ui3 = firrtl.constant(0 : ui3) : !firrtl.uint<3>
// WRAPPER-NEXT:     %_M_read, %_M_write = firrtl.instance @_M {name = "_M", portNames = ["read", "write"]} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>, !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>
// WRAPPER-NEXT:     %0 = firrtl.subfield %_M_read("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.sint<42>
// WRAPPER-NEXT:     firrtl.connect %result, %0 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
// WRAPPER-NEXT:     %1 = firrtl.subfield %_M_read("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<4>>
// WRAPPER-NEXT:     firrtl.connect %1, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %_M_read("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<1>>
// WRAPPER-NEXT:     firrtl.connect %2, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %3 = firrtl.subfield %_M_read("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<clock>
// WRAPPER-NEXT:     firrtl.connect %3, %clock1 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %4 = firrtl.subfield %_M_write("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<4>>
// WRAPPER-NEXT:     %5 = firrtl.invalidvalue : !firrtl.uint<3>
// WRAPPER-NEXT:     %6 = firrtl.mux(%inpred, %c0_ui3, %5) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
// WRAPPER-NEXT:     firrtl.connect %4, %6 : !firrtl.flip<uint<4>>, !firrtl.uint<3>
// WRAPPER-NEXT:     %7 = firrtl.subfield %_M_write("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// WRAPPER-NEXT:     %8 = firrtl.mux(%inpred, %c1_ui1, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %7, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %9 = firrtl.subfield %_M_write("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<clock>
// WRAPPER-NEXT:     %10 = firrtl.invalidvalue : !firrtl.clock
// WRAPPER-NEXT:     %11 = firrtl.mux(%inpred, %clock2, %10) : (!firrtl.uint<1>, !firrtl.clock, !firrtl.clock) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %9, %11 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %12 = firrtl.subfield %_M_write("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<sint<42>>
// WRAPPER-NEXT:     %13 = firrtl.invalidvalue : !firrtl.sint<42>
// WRAPPER-NEXT:     %14 = firrtl.mux(%inpred, %indata, %13) : (!firrtl.uint<1>, !firrtl.sint<42>, !firrtl.sint<42>) -> !firrtl.sint<42>
// WRAPPER-NEXT:     firrtl.connect %12, %14 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
// WRAPPER-NEXT:     %15 = firrtl.subfield %_M_write("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// WRAPPER-NEXT:     %16 = firrtl.invalidvalue : !firrtl.uint<1>
// WRAPPER-NEXT:     %17 = firrtl.mux(%inpred, %c1_ui1, %16) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %15, %17 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT: }

// INLINE-LABEL: firrtl.circuit "MemSimple" {
// INLINE-NEXT:   firrtl.extmodule @_M_ext(!firrtl.uint<4> {firrtl.name = "R0_addr"}, !firrtl.uint<1> {firrtl.name = "R0_en"}, !firrtl.clock {firrtl.name = "R0_clk"}, !firrtl.flip<sint<42>> {firrtl.name = "R0_data"}, !firrtl.uint<4> {firrtl.name = "W0_addr"}, !firrtl.uint<1> {firrtl.name = "W0_en"}, !firrtl.clock {firrtl.name = "W0_clk"}, !firrtl.sint<42> {firrtl.name = "W0_data"}, !firrtl.uint<1> {firrtl.name = "W0_mask"}) attributes {depth = 12 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// INLINE-NEXT:   firrtl.module @MemSimple(%clock1: !firrtl.clock, %clock2: !firrtl.clock, %inpred: !firrtl.uint<1>, %indata: !firrtl.sint<42>, %result: !firrtl.flip<sint<42>>) {
// INLINE-NEXT:     %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
// INLINE-NEXT:     %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
// INLINE-NEXT:     %c0_ui3 = firrtl.constant(0 : ui3) : !firrtl.uint<3>
// INLINE-NEXT:     %_M_R0_addr, %_M_R0_en, %_M_R0_clk, %_M_R0_data, %_M_W0_addr, %_M_W0_en, %_M_W0_clk, %_M_W0_data, %_M_W0_mask = firrtl.instance @_M_ext {name = "_M", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data", "W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<42>, !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<42>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %0 = firrtl.wire  : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>
// INLINE-NEXT:     %1 = firrtl.subfield %0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %_M_R0_addr, %1 : !firrtl.flip<uint<4>>, !firrtl.flip<uint<4>>
// INLINE-NEXT:     %2 = firrtl.subfield %0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %_M_R0_en, %2 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %3 = firrtl.subfield %0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %_M_R0_clk, %3 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %4 = firrtl.subfield %0("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.sint<42>
// INLINE-NEXT:     firrtl.connect %4, %_M_R0_data : !firrtl.sint<42>, !firrtl.sint<42>
// INLINE-NEXT:     %5 = firrtl.wire  : !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>
// INLINE-NEXT:     %6 = firrtl.subfield %5("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %_M_W0_addr, %6 : !firrtl.flip<uint<4>>, !firrtl.flip<uint<4>>
// INLINE-NEXT:     %7 = firrtl.subfield %5("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %_M_W0_en, %7 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %8 = firrtl.subfield %5("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %_M_W0_clk, %8 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %9 = firrtl.subfield %5("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<sint<42>>
// INLINE-NEXT:     firrtl.connect %_M_W0_data, %9 : !firrtl.flip<sint<42>>, !firrtl.flip<sint<42>>
// INLINE-NEXT:     %10 = firrtl.subfield %5("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %_M_W0_mask, %10 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %11 = firrtl.subfield %0("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.sint<42>
// INLINE-NEXT:     firrtl.connect %result, %11 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
// INLINE-NEXT:     %12 = firrtl.subfield %0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %12, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
// INLINE-NEXT:     %13 = firrtl.subfield %0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %13, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// INLINE-NEXT:     %14 = firrtl.subfield %0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %14, %clock1 : !firrtl.flip<clock>, !firrtl.clock
// INLINE-NEXT:     %15 = firrtl.subfield %5("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     %16 = firrtl.invalidvalue : !firrtl.uint<3>
// INLINE-NEXT:     %17 = firrtl.mux(%inpred, %c0_ui3, %16) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
// INLINE-NEXT:     firrtl.connect %15, %17 : !firrtl.flip<uint<4>>, !firrtl.uint<3>
// INLINE-NEXT:     %18 = firrtl.subfield %5("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     %19 = firrtl.mux(%inpred, %c1_ui1, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// INLINE-NEXT:     firrtl.connect %18, %19 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// INLINE-NEXT:     %20 = firrtl.subfield %5("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     %21 = firrtl.invalidvalue : !firrtl.clock
// INLINE-NEXT:     %22 = firrtl.mux(%inpred, %clock2, %21) : (!firrtl.uint<1>, !firrtl.clock, !firrtl.clock) -> !firrtl.clock
// INLINE-NEXT:     firrtl.connect %20, %22 : !firrtl.flip<clock>, !firrtl.clock
// INLINE-NEXT:     %23 = firrtl.subfield %5("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<sint<42>>
// INLINE-NEXT:     %24 = firrtl.invalidvalue : !firrtl.sint<42>
// INLINE-NEXT:     %25 = firrtl.mux(%inpred, %indata, %24) : (!firrtl.uint<1>, !firrtl.sint<42>, !firrtl.sint<42>) -> !firrtl.sint<42>
// INLINE-NEXT:     firrtl.connect %23, %25 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
// INLINE-NEXT:     %26 = firrtl.subfield %5("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     %27 = firrtl.invalidvalue : !firrtl.uint<1>
// INLINE-NEXT:     %28 = firrtl.mux(%inpred, %c1_ui1, %27) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// INLINE-NEXT:     firrtl.connect %26, %28 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// INLINE-NEXT:   }
// INLINE-NEXT: }

firrtl.circuit "NameCollision" {
  // Check for name NameCollision with a generated module
  firrtl.module @NameCollisionMemory_ext() {
    %0 = firrtl.mem Undefined {depth = 16 : i64, name = "NameCollisionMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
  }
  firrtl.module @NameCollision() {
    %0 = firrtl.mem Undefined {depth = 16 : i64, name = "NameCollisionMemory", portNames = ["write0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
  }
}

// WRAPPER-LABEL: firrtl.circuit "NameCollision" {
// WRAPPER-NEXT:   firrtl.extmodule @NameCollisionMemory_ext_1(!firrtl.uint<4> {firrtl.name = "W0_addr"}, !firrtl.uint<1> {firrtl.name = "W0_en"}, !firrtl.clock {firrtl.name = "W0_clk"}, !firrtl.sint<8> {firrtl.name = "W0_data"}, !firrtl.uint<1> {firrtl.name = "W0_mask"}) attributes {depth = 16 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// WRAPPER-NEXT:   firrtl.module @NameCollisionMemory_0(%write0: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) {
// WRAPPER-NEXT:     %NameCollisionMemory_W0_addr, %NameCollisionMemory_W0_en, %NameCollisionMemory_W0_clk, %NameCollisionMemory_W0_data, %NameCollisionMemory_W0_mask = firrtl.instance @NameCollisionMemory_ext_1 {name = "NameCollisionMemory", portNames = ["W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<8>>, !firrtl.flip<uint<1>>
// WRAPPER-NEXT:     %0 = firrtl.subfield %write0("addr") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<4>
// WRAPPER-NEXT:     firrtl.connect %NameCollisionMemory_W0_addr, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
// WRAPPER-NEXT:     %1 = firrtl.subfield %write0("en") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %NameCollisionMemory_W0_en, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %write0("clk") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %NameCollisionMemory_W0_clk, %2 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %3 = firrtl.subfield %write0("data") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.sint<8>
// WRAPPER-NEXT:     firrtl.connect %NameCollisionMemory_W0_data, %3 : !firrtl.flip<sint<8>>, !firrtl.sint<8>
// WRAPPER-NEXT:     %4 = firrtl.subfield %write0("mask") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %NameCollisionMemory_W0_mask, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT:   firrtl.extmodule @NameCollisionMemory_ext_0(!firrtl.uint<4> {firrtl.name = "R0_addr"}, !firrtl.uint<1> {firrtl.name = "R0_en"}, !firrtl.clock {firrtl.name = "R0_clk"}, !firrtl.flip<sint<8>> {firrtl.name = "R0_data"}) attributes {depth = 16 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// WRAPPER-NEXT:   firrtl.module @NameCollisionMemory(%read0: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) {
// WRAPPER-NEXT:     %NameCollisionMemory_R0_addr, %NameCollisionMemory_R0_en, %NameCollisionMemory_R0_clk, %NameCollisionMemory_R0_data = firrtl.instance @NameCollisionMemory_ext_0 {name = "NameCollisionMemory", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<8>
// WRAPPER-NEXT:     %0 = firrtl.subfield %read0("addr") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.uint<4>
// WRAPPER-NEXT:     firrtl.connect %NameCollisionMemory_R0_addr, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
// WRAPPER-NEXT:     %1 = firrtl.subfield %read0("en") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %NameCollisionMemory_R0_en, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %read0("clk") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %NameCollisionMemory_R0_clk, %2 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %3 = firrtl.subfield %read0("data") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.flip<sint<8>>
// WRAPPER-NEXT:     firrtl.connect %3, %NameCollisionMemory_R0_data : !firrtl.flip<sint<8>>, !firrtl.sint<8>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT:   firrtl.module @NameCollisionMemory_ext() {
// WRAPPER-NEXT:     %NameCollisionMemory_read0 = firrtl.instance @NameCollisionMemory {name = "NameCollisionMemory", portNames = ["read0"]} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT:   firrtl.module @NameCollision() {
// WRAPPER-NEXT:     %NameCollisionMemory_write0 = firrtl.instance @NameCollisionMemory_0 {name = "NameCollisionMemory", portNames = ["write0"]} : !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT: }

// INLINE-LABEL: firrtl.circuit "NameCollision" {
// INLINE-NEXT:   firrtl.extmodule @NameCollisionMemory_ext_1(!firrtl.uint<4> {firrtl.name = "W0_addr"}, !firrtl.uint<1> {firrtl.name = "W0_en"}, !firrtl.clock {firrtl.name = "W0_clk"}, !firrtl.sint<8> {firrtl.name = "W0_data"}, !firrtl.uint<1> {firrtl.name = "W0_mask"}) attributes {depth = 16 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// INLINE-NEXT:   firrtl.extmodule @NameCollisionMemory_ext_0(!firrtl.uint<4> {firrtl.name = "R0_addr"}, !firrtl.uint<1> {firrtl.name = "R0_en"}, !firrtl.clock {firrtl.name = "R0_clk"}, !firrtl.flip<sint<8>> {firrtl.name = "R0_data"}) attributes {depth = 16 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// INLINE-NEXT:   firrtl.module @NameCollisionMemory_ext() {
// INLINE-NEXT:     %NameCollisionMemory_R0_addr, %NameCollisionMemory_R0_en, %NameCollisionMemory_R0_clk, %NameCollisionMemory_R0_data = firrtl.instance @NameCollisionMemory_ext_0 {name = "NameCollisionMemory", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<8>
// INLINE-NEXT:     %0 = firrtl.wire  : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// INLINE-NEXT:     %1 = firrtl.subfield %0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %NameCollisionMemory_R0_addr, %1 : !firrtl.flip<uint<4>>, !firrtl.flip<uint<4>>
// INLINE-NEXT:     %2 = firrtl.subfield %0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %NameCollisionMemory_R0_en, %2 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %3 = firrtl.subfield %0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %NameCollisionMemory_R0_clk, %3 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %4 = firrtl.subfield %0("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.sint<8>
// INLINE-NEXT:     firrtl.connect %4, %NameCollisionMemory_R0_data : !firrtl.sint<8>, !firrtl.sint<8>
// INLINE-NEXT:   }
// INLINE-NEXT:   firrtl.module @NameCollision() {
// INLINE-NEXT:     %NameCollisionMemory_W0_addr, %NameCollisionMemory_W0_en, %NameCollisionMemory_W0_clk, %NameCollisionMemory_W0_data, %NameCollisionMemory_W0_mask = firrtl.instance @NameCollisionMemory_ext_1 {name = "NameCollisionMemory", portNames = ["W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<8>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %0 = firrtl.wire  : !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// INLINE-NEXT:     %1 = firrtl.subfield %0("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %NameCollisionMemory_W0_addr, %1 : !firrtl.flip<uint<4>>, !firrtl.flip<uint<4>>
// INLINE-NEXT:     %2 = firrtl.subfield %0("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %NameCollisionMemory_W0_en, %2 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %3 = firrtl.subfield %0("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %NameCollisionMemory_W0_clk, %3 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %4 = firrtl.subfield %0("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<sint<8>>
// INLINE-NEXT:     firrtl.connect %NameCollisionMemory_W0_data, %4 : !firrtl.flip<sint<8>>, !firrtl.flip<sint<8>>
// INLINE-NEXT:     %5 = firrtl.subfield %0("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %NameCollisionMemory_W0_mask, %5 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:   }
// INLINE-NEXT: }

firrtl.circuit "Duplicate" {
  firrtl.module @Duplicate() {
    %r0 = firrtl.mem Undefined {depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
    %w0 = firrtl.mem Undefined {depth = 1 : i64, name = "WriteMemory", portNames = ["write0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
    %r1 = firrtl.mem Undefined {depth = 16 : i64, name = "ReadMemory1", portNames = ["read1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
    %w1 = firrtl.mem Undefined {depth = 1 : i64, name = "WriteMemory1", portNames = ["write1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
    %r2 = firrtl.mem Undefined {depth = 16 : i64, name = "ReadMemory2", portNames = ["read2"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
    %w2 = firrtl.mem Undefined {depth = 1 : i64, name = "WriteMemory2", portNames = ["write2"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
  }
}

// WRAPPER-LABEL: firrtl.circuit "Duplicate" {
// WRAPPER-NEXT:   firrtl.extmodule @WriteMemory_ext(!firrtl.uint<1> {firrtl.name = "W0_addr"}, !firrtl.uint<1> {firrtl.name = "W0_en"}, !firrtl.clock {firrtl.name = "W0_clk"}, !firrtl.sint<8> {firrtl.name = "W0_data"}, !firrtl.uint<1> {firrtl.name = "W0_mask"}) attributes {depth = 1 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// WRAPPER-NEXT:   firrtl.module @WriteMemory(%write0: !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) {
// WRAPPER-NEXT:     %WriteMemory_W0_addr, %WriteMemory_W0_en, %WriteMemory_W0_clk, %WriteMemory_W0_data, %WriteMemory_W0_mask = firrtl.instance @WriteMemory_ext {name = "WriteMemory", portNames = ["W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<8>>, !firrtl.flip<uint<1>>
// WRAPPER-NEXT:     %0 = firrtl.subfield %write0("addr") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_addr, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %1 = firrtl.subfield %write0("en") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_en, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %write0("clk") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_clk, %2 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %3 = firrtl.subfield %write0("data") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.sint<8>
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_data, %3 : !firrtl.flip<sint<8>>, !firrtl.sint<8>
// WRAPPER-NEXT:     %4 = firrtl.subfield %write0("mask") : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %WriteMemory_W0_mask, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT:   firrtl.extmodule @ReadMemory_ext(!firrtl.uint<4> {firrtl.name = "R0_addr"}, !firrtl.uint<1> {firrtl.name = "R0_en"}, !firrtl.clock {firrtl.name = "R0_clk"}, !firrtl.flip<sint<8>> {firrtl.name = "R0_data"}) attributes {depth = 16 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// WRAPPER-NEXT:   firrtl.module @ReadMemory(%read0: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) {
// WRAPPER-NEXT:     %ReadMemory_R0_addr, %ReadMemory_R0_en, %ReadMemory_R0_clk, %ReadMemory_R0_data = firrtl.instance @ReadMemory_ext {name = "ReadMemory", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<8>
// WRAPPER-NEXT:     %0 = firrtl.subfield %read0("addr") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.uint<4>
// WRAPPER-NEXT:     firrtl.connect %ReadMemory_R0_addr, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
// WRAPPER-NEXT:     %1 = firrtl.subfield %read0("en") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.uint<1>
// WRAPPER-NEXT:     firrtl.connect %ReadMemory_R0_en, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// WRAPPER-NEXT:     %2 = firrtl.subfield %read0("clk") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.clock
// WRAPPER-NEXT:     firrtl.connect %ReadMemory_R0_clk, %2 : !firrtl.flip<clock>, !firrtl.clock
// WRAPPER-NEXT:     %3 = firrtl.subfield %read0("data") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<sint<8>>>) -> !firrtl.flip<sint<8>>
// WRAPPER-NEXT:     firrtl.connect %3, %ReadMemory_R0_data : !firrtl.flip<sint<8>>, !firrtl.sint<8>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT:   firrtl.module @Duplicate() {
// WRAPPER-NEXT:     %ReadMemory_read0 = firrtl.instance @ReadMemory {name = "ReadMemory", portNames = ["read0"]} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// WRAPPER-NEXT:     %WriteMemory_write0 = firrtl.instance @WriteMemory {name = "WriteMemory", portNames = ["write0"]} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// WRAPPER-NEXT:     %ReadMemory1_read0 = firrtl.instance @ReadMemory {name = "ReadMemory1", portNames = ["read0"]} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// WRAPPER-NEXT:     %WriteMemory1_write0 = firrtl.instance @WriteMemory {name = "WriteMemory1", portNames = ["write0"]} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// WRAPPER-NEXT:     %ReadMemory2_read0 = firrtl.instance @ReadMemory {name = "ReadMemory2", portNames = ["read0"]} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// WRAPPER-NEXT:     %WriteMemory2_write0 = firrtl.instance @WriteMemory {name = "WriteMemory2", portNames = ["write0"]} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// WRAPPER-NEXT:   }
// WRAPPER-NEXT: }

// INLINE-LABEL: firrtl.circuit "Duplicate" {
// INLINE-NEXT:   firrtl.extmodule @WriteMemory_ext(!firrtl.uint<1> {firrtl.name = "W0_addr"}, !firrtl.uint<1> {firrtl.name = "W0_en"}, !firrtl.clock {firrtl.name = "W0_clk"}, !firrtl.sint<8> {firrtl.name = "W0_data"}, !firrtl.uint<1> {firrtl.name = "W0_mask"}) attributes {depth = 1 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// INLINE-NEXT:   firrtl.extmodule @ReadMemory_ext(!firrtl.uint<4> {firrtl.name = "R0_addr"}, !firrtl.uint<1> {firrtl.name = "R0_en"}, !firrtl.clock {firrtl.name = "R0_clk"}, !firrtl.flip<sint<8>> {firrtl.name = "R0_data"}) attributes {depth = 16 : i64, generator = "FIRRTLMemory", readLatency = 1 : i32, ruw = 0 : i32, writeLatency = 1 : i32}
// INLINE-NEXT:   firrtl.module @Duplicate() {
// INLINE-NEXT:     %ReadMemory_R0_addr, %ReadMemory_R0_en, %ReadMemory_R0_clk, %ReadMemory_R0_data = firrtl.instance @ReadMemory_ext {name = "ReadMemory", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<8>
// INLINE-NEXT:     %0 = firrtl.wire  : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// INLINE-NEXT:     %1 = firrtl.subfield %0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %ReadMemory_R0_addr, %1 : !firrtl.flip<uint<4>>, !firrtl.flip<uint<4>>
// INLINE-NEXT:     %2 = firrtl.subfield %0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %ReadMemory_R0_en, %2 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %3 = firrtl.subfield %0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %ReadMemory_R0_clk, %3 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %4 = firrtl.subfield %0("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.sint<8>
// INLINE-NEXT:     firrtl.connect %4, %ReadMemory_R0_data : !firrtl.sint<8>, !firrtl.sint<8>
// INLINE-NEXT:     %WriteMemory_W0_addr, %WriteMemory_W0_en, %WriteMemory_W0_clk, %WriteMemory_W0_data, %WriteMemory_W0_mask = firrtl.instance @WriteMemory_ext {name = "WriteMemory", portNames = ["W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<8>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %5 = firrtl.wire  : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// INLINE-NEXT:     %6 = firrtl.subfield %5("addr") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_addr, %6 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %7 = firrtl.subfield %5("en") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_en, %7 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %8 = firrtl.subfield %5("clk") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_clk, %8 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %9 = firrtl.subfield %5("data") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<sint<8>>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_data, %9 : !firrtl.flip<sint<8>>, !firrtl.flip<sint<8>>
// INLINE-NEXT:     %10 = firrtl.subfield %5("mask") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory_W0_mask, %10 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %ReadMemory1_R0_addr, %ReadMemory1_R0_en, %ReadMemory1_R0_clk, %ReadMemory1_R0_data = firrtl.instance @ReadMemory_ext {name = "ReadMemory1", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<8>
// INLINE-NEXT:     %11 = firrtl.wire  : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// INLINE-NEXT:     %12 = firrtl.subfield %11("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %ReadMemory1_R0_addr, %12 : !firrtl.flip<uint<4>>, !firrtl.flip<uint<4>>
// INLINE-NEXT:     %13 = firrtl.subfield %11("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %ReadMemory1_R0_en, %13 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %14 = firrtl.subfield %11("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %ReadMemory1_R0_clk, %14 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %15 = firrtl.subfield %11("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.sint<8>
// INLINE-NEXT:     firrtl.connect %15, %ReadMemory1_R0_data : !firrtl.sint<8>, !firrtl.sint<8>
// INLINE-NEXT:     %WriteMemory1_W0_addr, %WriteMemory1_W0_en, %WriteMemory1_W0_clk, %WriteMemory1_W0_data, %WriteMemory1_W0_mask = firrtl.instance @WriteMemory_ext {name = "WriteMemory1", portNames = ["W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<8>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %16 = firrtl.wire  : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// INLINE-NEXT:     %17 = firrtl.subfield %16("addr") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory1_W0_addr, %17 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %18 = firrtl.subfield %16("en") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory1_W0_en, %18 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %19 = firrtl.subfield %16("clk") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %WriteMemory1_W0_clk, %19 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %20 = firrtl.subfield %16("data") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<sint<8>>
// INLINE-NEXT:     firrtl.connect %WriteMemory1_W0_data, %20 : !firrtl.flip<sint<8>>, !firrtl.flip<sint<8>>
// INLINE-NEXT:     %21 = firrtl.subfield %16("mask") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory1_W0_mask, %21 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %ReadMemory2_R0_addr, %ReadMemory2_R0_en, %ReadMemory2_R0_clk, %ReadMemory2_R0_data = firrtl.instance @ReadMemory_ext {name = "ReadMemory2", portNames = ["R0_addr", "R0_en", "R0_clk", "R0_data"]} : !firrtl.flip<uint<4>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.sint<8>
// INLINE-NEXT:     %22 = firrtl.wire  : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>
// INLINE-NEXT:     %23 = firrtl.subfield %22("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<4>>
// INLINE-NEXT:     firrtl.connect %ReadMemory2_R0_addr, %23 : !firrtl.flip<uint<4>>, !firrtl.flip<uint<4>>
// INLINE-NEXT:     %24 = firrtl.subfield %22("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %ReadMemory2_R0_en, %24 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %25 = firrtl.subfield %22("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %ReadMemory2_R0_clk, %25 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %26 = firrtl.subfield %22("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<8>>) -> !firrtl.sint<8>
// INLINE-NEXT:     firrtl.connect %26, %ReadMemory2_R0_data : !firrtl.sint<8>, !firrtl.sint<8>
// INLINE-NEXT:     %WriteMemory2_W0_addr, %WriteMemory2_W0_en, %WriteMemory2_W0_clk, %WriteMemory2_W0_data, %WriteMemory2_W0_mask = firrtl.instance @WriteMemory_ext {name = "WriteMemory2", portNames = ["W0_addr", "W0_en", "W0_clk", "W0_data", "W0_mask"]} : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>, !firrtl.flip<clock>, !firrtl.flip<sint<8>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %27 = firrtl.wire  : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>
// INLINE-NEXT:     %28 = firrtl.subfield %27("addr") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory2_W0_addr, %28 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %29 = firrtl.subfield %27("en") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory2_W0_en, %29 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:     %30 = firrtl.subfield %27("clk") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<clock>
// INLINE-NEXT:     firrtl.connect %WriteMemory2_W0_clk, %30 : !firrtl.flip<clock>, !firrtl.flip<clock>
// INLINE-NEXT:     %31 = firrtl.subfield %27("data") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<sint<8>>
// INLINE-NEXT:     firrtl.connect %WriteMemory2_W0_data, %31 : !firrtl.flip<sint<8>>, !firrtl.flip<sint<8>>
// INLINE-NEXT:     %32 = firrtl.subfield %27("mask") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
// INLINE-NEXT:     firrtl.connect %WriteMemory2_W0_mask, %32 : !firrtl.flip<uint<1>>, !firrtl.flip<uint<1>>
// INLINE-NEXT:   }
// INLINE-NEXT: }