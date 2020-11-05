; RUN: firtool %s --format=fir              | circt-opt | FileCheck %s 

circuit foo :

  module foo :
    input clock : Clock
    input reset : UInt<1>

    mem bar : @[Decoupled.scala 218:16]
      data-type => UInt<3>
      depth => 1
      read-latency => 0
      write-latency => 1
      reader => io_deq_bits_MPORT
      writer => MPORT
      read-under-write => undefined
    
; CHECK:         %bar = firrtl.mem "Undefined" {depth = 1 : i64, name = "bar", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<MPORT: bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<3>, mask: uint<1>>, io_deq_bits_MPORT: bundle<addr: uint<1>, en: uint<1>, clk: clock, data: flip<uint<3>>>>

