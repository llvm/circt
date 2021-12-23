// RUN: circt-reduce %s --test %S/test.sh --test-arg cat --test-arg "firrtl.module @Basic" --keep-best=0 --include memory-stubber | FileCheck %s

firrtl.circuit "Basic"   {

//  module Basic :
//    input clock: Clock
//    input wAddr: UInt<4>
//    input wEn: UInt<1>
//    input wMask: {a: UInt<1>, b: UInt<1>}
//    input wData: {a: UInt<8>, b: SInt<8>}
//
//    output out0: {a: UInt<8>, b: SInt<8>}
//    output out1: {a: UInt<8>, b: SInt<8>}
//
//    mem memory:
//      data-type => {a: UInt<8>, b: SInt<8>}
//      depth => 16
//      writer => w0
//      writer => w1
//      reader => r0
//      reader => r1
//      read-latency => 0
//      write-latency => 1
//      read-under-write => undefined
//
//    memory.w0.clk <= clock
//    memory.w0.en <= wEn
//    memory.w0.addr <= wAddr
//    memory.w0.mask <= wMask
//    memory.w0.data <= wData
//
//    memory.w1.clk <= clock
//    memory.w1.en <= wEn
//    memory.w1.addr <= wAddr
//    memory.w1.mask <= wMask
//    memory.w1.data <= wData
//
//    memory.r0.clk <= clock
//    memory.r0.addr <= wAddr
//    out0 <= memory.r0.data
//
//    memory.r1.clk <= clock
//    memory.r1.addr <= wAddr
//    out1 <= memory.r0.data

    firrtl.module @Basic(in %clock: !firrtl.clock, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %wData: !firrtl.bundle<a: uint<8>, b: sint<8>>, out %out0: !firrtl.bundle<a: uint<8>, b: sint<8>>, out %out1: !firrtl.bundle<a: uint<8>, b: sint<8>>) {
      %memory_r0, %memory_r1, %memory_w0, %memory_w1 = firrtl.mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r0", "r1", "w0", "w1"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
      %0 = firrtl.subfield %memory_r1(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.uint<4>
      %1 = firrtl.subfield %memory_r1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.clock
      %2 = firrtl.subfield %memory_r0(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
      %3 = firrtl.subfield %memory_r0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.uint<4>
      %4 = firrtl.subfield %memory_r0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.clock
      %5 = firrtl.subfield %memory_w1(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
      %6 = firrtl.subfield %memory_w1(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
      %7 = firrtl.subfield %memory_w1(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
      %8 = firrtl.subfield %memory_w1(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
      %9 = firrtl.subfield %memory_w1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
      %10 = firrtl.subfield %memory_w0(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
      %11 = firrtl.subfield %memory_w0(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
      %12 = firrtl.subfield %memory_w0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
      %13 = firrtl.subfield %memory_w0(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
      %14 = firrtl.subfield %memory_w0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
      firrtl.connect %14, %clock : !firrtl.clock, !firrtl.clock
      firrtl.connect %13, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %12, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
      firrtl.connect %11, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
      firrtl.connect %10, %wData : !firrtl.bundle<a: uint<8>, b: sint<8>>, !firrtl.bundle<a: uint<8>, b: sint<8>>
      firrtl.connect %9, %clock : !firrtl.clock, !firrtl.clock
      firrtl.connect %8, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %7, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
      firrtl.connect %6, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
      firrtl.connect %5, %wData : !firrtl.bundle<a: uint<8>, b: sint<8>>, !firrtl.bundle<a: uint<8>, b: sint<8>>
      firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
      firrtl.connect %3, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
      firrtl.connect %out0, %2 : !firrtl.bundle<a: uint<8>, b: sint<8>>, !firrtl.bundle<a: uint<8>, b: sint<8>>
      firrtl.connect %1, %clock : !firrtl.clock, !firrtl.clock
      firrtl.connect %0, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
      firrtl.connect %out1, %2 : !firrtl.bundle<a: uint<8>, b: sint<8>>, !firrtl.bundle<a: uint<8>, b: sint<8>>
    }
    
// CHECK-LABEL: firrtl.module @Basic(in %clock: !firrtl.clock, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %wData: !firrtl.bundle<a: uint<8>, b: sint<8>>, out %out0: !firrtl.bundle<a: uint<8>, b: sint<8>>, out %out1: !firrtl.bundle<a: uint<8>, b: sint<8>>) {
// CHECK-NEXT:   %memory_r0 = firrtl.wire  : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>
// CHECK-NEXT:   %0 = firrtl.subfield %memory_r0(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %1 = firrtl.subfield %0(0) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.uint<8>
// CHECK-NEXT:   %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %1, %invalid_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:   %2 = firrtl.subfield %0(1) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.sint<8>
// CHECK-NEXT:   %invalid_si8 = firrtl.invalidvalue : !firrtl.sint<8>
// CHECK-NEXT:   firrtl.connect %2, %invalid_si8 : !firrtl.sint<8>, !firrtl.sint<8>
// CHECK-NEXT:   %3 = firrtl.subfield %memory_r0(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %4 = firrtl.subfield %memory_r0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.uint<4>
// CHECK-NEXT:   %5 = firrtl.subfield %memory_r0(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %6 = firrtl.xor %4, %5 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<4>
// CHECK-NEXT:   %memory_r1 = firrtl.wire  : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>
// CHECK-NEXT:   %7 = firrtl.subfield %memory_r1(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %8 = firrtl.subfield %7(0) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %8, %invalid_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:   %9 = firrtl.subfield %7(1) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.sint<8>
// CHECK-NEXT:   firrtl.connect %9, %invalid_si8 : !firrtl.sint<8>, !firrtl.sint<8>
// CHECK-NEXT:   %10 = firrtl.subfield %memory_r1(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %11 = firrtl.subfield %memory_r1(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.uint<4>
// CHECK-NEXT:   %12 = firrtl.xor %6, %11 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK-NEXT:   %13 = firrtl.subfield %memory_r1(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %14 = firrtl.xor %12, %13 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<4>
// CHECK-NEXT:   %memory_w0 = firrtl.wire  : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
// CHECK-NEXT:   %15 = firrtl.subfield %memory_w0(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %16 = firrtl.subfield %memory_w0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
// CHECK-NEXT:   %17 = firrtl.xor %14, %16 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK-NEXT:   %18 = firrtl.subfield %memory_w0(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %19 = firrtl.xor %17, %18 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<4>
// CHECK-NEXT:   %20 = firrtl.subfield %memory_w0(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
// CHECK-NEXT:   %21 = firrtl.subfield %20(0) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %22 = firrtl.xor %19, %21 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<4>
// CHECK-NEXT:   %23 = firrtl.subfield %20(1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %24 = firrtl.xor %22, %23 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<4>
// CHECK-NEXT:   %25 = firrtl.subfield %15(0) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.uint<8>
// CHECK-NEXT:   %26 = firrtl.xor %24, %25 : (!firrtl.uint<4>, !firrtl.uint<8>) -> !firrtl.uint<8>
// CHECK-NEXT:   %27 = firrtl.subfield %15(1) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.sint<8>
// CHECK-NEXT:   %28 = firrtl.asUInt %27 : (!firrtl.sint<8>) -> !firrtl.uint<8>
// CHECK-NEXT:   %29 = firrtl.xor %26, %28 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
// CHECK-NEXT:   %memory_w1 = firrtl.wire  : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
// CHECK-NEXT:   %30 = firrtl.subfield %memory_w1(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %31 = firrtl.subfield %memory_w1(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
// CHECK-NEXT:   %32 = firrtl.xor %29, %31 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<8>
// CHECK-NEXT:   %33 = firrtl.subfield %memory_w1(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %34 = firrtl.xor %32, %33 : (!firrtl.uint<8>, !firrtl.uint<1>) -> !firrtl.uint<8>
// CHECK-NEXT:   %35 = firrtl.subfield %memory_w1(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
// CHECK-NEXT:   %36 = firrtl.subfield %35(0) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %37 = firrtl.xor %34, %36 : (!firrtl.uint<8>, !firrtl.uint<1>) -> !firrtl.uint<8>
// CHECK-NEXT:   %38 = firrtl.subfield %35(1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %39 = firrtl.xor %37, %38 : (!firrtl.uint<8>, !firrtl.uint<1>) -> !firrtl.uint<8>
// CHECK-NEXT:   %40 = firrtl.subfield %30(0) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.uint<8>
// CHECK-NEXT:   %41 = firrtl.xor %39, %40 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
// CHECK-NEXT:   %42 = firrtl.subfield %30(1) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.sint<8>
// CHECK-NEXT:   %43 = firrtl.asUInt %42 : (!firrtl.sint<8>) -> !firrtl.uint<8>
// CHECK-NEXT:   %44 = firrtl.xor %41, %43 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
// CHECK-NEXT:   %45 = firrtl.subfield %3(0) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %45, %44 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:   %46 = firrtl.subfield %3(1) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.sint<8>
// CHECK-NEXT:   %47 = firrtl.asSInt %44 : (!firrtl.uint<8>) -> !firrtl.sint<8>
// CHECK-NEXT:   firrtl.connect %46, %47 : !firrtl.sint<8>, !firrtl.sint<8>
// CHECK-NEXT:   %48 = firrtl.subfield %10(0) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %48, %44 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:   %49 = firrtl.subfield %10(1) : (!firrtl.bundle<a: uint<8>, b: sint<8>>) -> !firrtl.sint<8>
// CHECK-NEXT:   %50 = firrtl.asSInt %44 : (!firrtl.uint<8>) -> !firrtl.sint<8>
// CHECK-NEXT:   firrtl.connect %49, %50 : !firrtl.sint<8>, !firrtl.sint<8>
// CHECK-NEXT:   %51 = firrtl.subfield %memory_r1(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.uint<4>
// CHECK-NEXT:   %52 = firrtl.subfield %memory_r1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.clock
// CHECK-NEXT:   %53 = firrtl.subfield %memory_r0(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %54 = firrtl.subfield %memory_r0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.uint<4>
// CHECK-NEXT:   %55 = firrtl.subfield %memory_r0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>) -> !firrtl.clock
// CHECK-NEXT:   %56 = firrtl.subfield %memory_w1(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %57 = firrtl.subfield %memory_w1(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
// CHECK-NEXT:   %58 = firrtl.subfield %memory_w1(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
// CHECK-NEXT:   %59 = firrtl.subfield %memory_w1(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %60 = firrtl.subfield %memory_w1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
// CHECK-NEXT:   %61 = firrtl.subfield %memory_w0(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   %62 = firrtl.subfield %memory_w0(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
// CHECK-NEXT:   %63 = firrtl.subfield %memory_w0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
// CHECK-NEXT:   %64 = firrtl.subfield %memory_w0(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %65 = firrtl.subfield %memory_w0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
// CHECK-NEXT:   firrtl.connect %65, %clock : !firrtl.clock, !firrtl.clock
// CHECK-NEXT:   firrtl.connect %64, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %63, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %62, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
// CHECK-NEXT:   firrtl.connect %61, %wData : !firrtl.bundle<a: uint<8>, b: sint<8>>, !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   firrtl.connect %60, %clock : !firrtl.clock, !firrtl.clock
// CHECK-NEXT:   firrtl.connect %59, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %58, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %57, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
// CHECK-NEXT:   firrtl.connect %56, %wData : !firrtl.bundle<a: uint<8>, b: sint<8>>, !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   firrtl.connect %55, %clock : !firrtl.clock, !firrtl.clock
// CHECK-NEXT:   firrtl.connect %54, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %out0, %53 : !firrtl.bundle<a: uint<8>, b: sint<8>>, !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT:   firrtl.connect %52, %clock : !firrtl.clock, !firrtl.clock
// CHECK-NEXT:   firrtl.connect %51, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %out1, %53 : !firrtl.bundle<a: uint<8>, b: sint<8>>, !firrtl.bundle<a: uint<8>, b: sint<8>>
// CHECK-NEXT: }

}
