; RUN: firtool %s --format=fir              | circt-opt | FileCheck %s --check-prefix=PARSE
; RUN: firtool %s --format=fir -disable-opt | circt-opt -pass-pipeline='firrtl.circuit(lower-firrtl-to-rtl)' | FileCheck %s --check-prefix=RTL

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
    
    bar.io_deq_bits_MPORT.addr <= UInt<1>("h0")
; PARSE:       firrtl.connect %1, %2 : !firrtl.uint<0>, !firrtl.uint<0>
; RTL:      rtl.connect %3, %2 : i0
