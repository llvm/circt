; RUN: firtool %s --inline-mem | FileCheck %s --implicit-check-not 'memory0_8x32' --implicit-check-not 'memory1_8x32'
FIRRTL version 3.0.0
circuit Foo: %[[
  {
    "class": "firrtl.annotations.MemoryFileInlineAnnotation",
    "target": "~Foo|Foo>memory1",
    "filename": "memory1.txt",
    "hexOrBinary": "h"
  },
  {
    "class": "firrtl.annotations.LoadMemoryAnnotation",
    "target": "~Foo|Foo>memory2",
    "fileName": "mememory2.txt",
    "hexOrBinary": "b",
    "originalMemoryNameOpt": "m"
  } 
]]
  type reader = {addr: UInt<3>, en: UInt<1>, clk: Clock, flip data: UInt<32>}
  type writer = {addr: UInt<3>, en: UInt<1>, clk: Clock, data: UInt<32>, mask: UInt<1>} 
  module Foo:
    
; CHECK: module memory2_8x32

    input r0: reader
    input w0: writer

    input r1: reader
    input w1: writer

    input r2: reader
    input w2: writer

    mem memory0 :
      data-type => UInt<32>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => r
      writer => w
      read-under-write => undefined

; CHECK: reg [31:0] memory0_ext_Memory[0:7];

    connect memory0.r, r0
    connect memory0.w, w0

    mem memory1 :
      data-type => UInt<32>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => r
      writer => w
      read-under-write => undefined

; CHECK: reg [31:0] memory1_ext_Memory[0:7];

    connect memory1.r, r1
    connect memory1.w, w1

    mem memory2 :
      data-type => UInt<32>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => r
      writer => w
      read-under-write => undefined

    connect memory2.r, r2
    connect memory2.w, w2

; CHECK memory2_8x32 memory2_ext