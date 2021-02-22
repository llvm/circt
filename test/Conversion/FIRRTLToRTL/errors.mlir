// RUN: circt-opt -lower-firrtl-to-rtl -split-input-file -verify-diagnostics %s

  // module MemAggregate :
  //    input clock1 : Clock
  //    input clock2 : Clock
  //
  //    mem _M : @[Decoupled.scala 209:24]
  //          data-type => { id : UInt<4>, other: SInt<8> }
  //          depth => 20
  //          read-latency => 0
  //          write-latency => 1
  //          reader => read
  //          writer => write
  //          read-under-write => undefined
  // COM: This is a memory with aggregates which is currently not
  // supported.
  rtl.module @MemAggregate(%clock1: i1, %clock2: i1) {
      %0 = firrtl.stdIntCast %clock1 : (i1) -> !firrtl.clock
      %1 = firrtl.stdIntCast %clock2 : (i1) -> !firrtl.clock
      // expected-error @+1 {{'firrtl.mem' op should have already been lowered}}
      %_M_read, %_M_write = firrtl.mem Undefined {depth = 20 : i64, name = "_M", portNames = ["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<5>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<id: uint<4>, other: sint<8>>>, !firrtl.flip<bundle<addr: uint<5>, en: uint<1>, clk: clock, data: bundle<id: uint<4>, other: sint<8>>, mask: bundle<id: uint<1>, other: uint<1>>>>
      rtl.output
    }

  // module MemOne :
  //   mem _M : @[Decoupled.scala 209:24]
  //         data-type => { id : UInt<4>, other: SInt<8> }
  //         depth => 1
  //         read-latency => 0
  //         write-latency => 1
  //         reader => read
  //         writer => write
  //         read-under-write => undefined
  // COM: This is an aggregate memory which is not supported.
  rtl.module @MemOne() {
      // expected-error @+1 {{'firrtl.mem' op should have already been lowered}}
    %_M_read, %_M_write = firrtl.mem Undefined {depth = 1 : i64, name = "_M", portNames=["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<1>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<id: uint<4>, other: sint<8>>>, !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<id: uint<4>, other: sint<8>>, mask: bundle<id: uint<1>, other: uint<1>>>>
    rtl.output
  }
