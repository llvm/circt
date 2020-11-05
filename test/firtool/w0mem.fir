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
    
    bar.io_deq_bits_MPORT.addr <= UInt<1>("h0")
; CHECK-NOT:       firrtl.connect
