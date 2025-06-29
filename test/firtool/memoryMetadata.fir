; RUN: firtool %s --annotation-file %s.anno.json --repl-seq-mem --repl-seq-mem-file="dutModule.conf" |  FileCheck %s


FIRRTL version 4.0.0
circuit test:
  module tbMemModule1:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem tbMemoryKind1:
      data-type => UInt<1>
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    connect tbMemoryKind1.r.clk, clock
    connect tbMemoryKind1.r.en, rEn
    connect tbMemoryKind1.r.addr, rAddr
    connect rData, tbMemoryKind1.r.data
    connect tbMemoryKind1.w.clk, clock
    connect tbMemoryKind1.w.en, rEn
    connect tbMemoryKind1.w.addr, rAddr
    connect tbMemoryKind1.w.mask, wMask
    connect tbMemoryKind1.w.data, wData

; CHECK-LABEL: module tbMemModule1
; CHECK:         tbMemoryKind1 tbMemoryKind1
; CHECK:       endmodule

  module dutModule2:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem dutMemory:
      data-type => UInt<2>
      depth => 32
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    connect dutMemory.r.clk, clock
    connect dutMemory.r.en, rEn
    connect dutMemory.r.addr, rAddr
    connect rData, dutMemory.r.data
    connect dutMemory.w.clk, clock
    connect dutMemory.w.en, rEn
    connect dutMemory.w.addr, rAddr
    connect dutMemory.w.mask, wMask
    connect dutMemory.w.data, wData

; CHECK-LABEL: module dutModule2
; CHECK:  dutMemory dutMemory

  module hier1:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem tbMemoryKind1:
      data-type => UInt<3>
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    connect tbMemoryKind1.r.clk, clock
    connect tbMemoryKind1.r.en, rEn
    connect tbMemoryKind1.r.addr, rAddr
    connect rData, tbMemoryKind1.r.data
    connect tbMemoryKind1.w.clk, clock
    connect tbMemoryKind1.w.en, rEn
    connect tbMemoryKind1.w.addr, rAddr
    connect tbMemoryKind1.w.mask, wMask
    connect tbMemoryKind1.w.data, wData

; CHECK-LABEL: module hier1
; CHECK:  tbMemoryKind1_0 tbMemoryKind1


  module hier2:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>


    inst m of tbMemModule1
    connect m.clock, clock
    connect m.rAddr, rAddr
    connect m.rEn, rEn
    connect rData, m.rData
    connect m.wMask, wMask
    connect m.wData, wData

  public module test:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>[3]
    input wMask: UInt<1>
    input wData: UInt<8>


    inst h1 of hier1
    connect h1.clock, clock
    connect h1.rAddr, rAddr
    connect h1.rEn, rEn
    connect rData[0], h1.rData
    connect h1.wMask, wMask
    connect h1.wData, wData

    inst h2 of hier2
    connect h2.clock, clock
    connect h2.rAddr, rAddr
    connect h2.rEn, rEn
    connect rData[1], h2.rData
    connect h2.wMask, wMask
    connect h2.wData, wData

    inst m2 of dutModule
    connect m2.clock, clock
    connect m2.rAddr, rAddr
    connect m2.rEn, rEn
    connect rData[2], m2.rData
    connect m2.wMask, wMask
    connect m2.wData, wData

; CHECK-LABEL: module test
; CHECK: hier1 [[h1:.+]] (
; CHECK: hier2 [[h2:.+]] (
; CHECK: dutModule [[m2:.+]] (

  module dutModule:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>


    inst m of dutModule2
    connect m.clock, clock
    connect m.rAddr, rAddr
    connect m.rEn, rEn
    connect rData, m.rData
    connect m.wMask, wMask
    connect m.wData, wData

; CHECK-LABEL: module dutModule
; CHECK: dutModule2 [[m:.+]] (


; CHECK-NOT: module tbMemoryKind1_ext
; CHECK-NOT: module dutMemory_ext
; CHECK-NOT: module tbMemoryKind1_0_ext


; CHECK-LABEL: FILE "metadata{{[/\]}}seq_mems.json"
; CHECK:      [
; CHECK-NEXT:   {
; CHECK-NEXT:     "module_name": "dutMemory_ext",
; CHECK-NEXT:     "depth": 32,
; CHECK-NEXT:     "width": 2,
; CHECK-NEXT:     "masked": false,
; CHECK-NEXT:     "read": 1,
; CHECK-NEXT:     "write": 1,
; CHECK-NEXT:     "readwrite": 0,
; CHECK-NEXT:     "ruw_behavior": "Undefined",
; CHECK-NEXT:     "extra_ports": [],
; CHECK-NEXT:     "hierarchy": [
; CHECK-NEXT:       "dutModule.hello.dutMemory.dutMemory_ext"
; CHECK-NEXT:     ]
; CHECK-NEXT:   }
; CHECK-NEXT: ]


; CHECK-LABEL: FILE "dutModule.conf"
; CHECK: name tbMemoryKind1_ext depth 16 width 1 ports write,read
; CHECK: name dutMemory_ext depth 32 width 2 ports write,read
; CHECK: name tbMemoryKind1_0_ext depth 16 width 3 ports write,read
