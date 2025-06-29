; RUN: firtool %s -repl-seq-mem -repl-seq-mem-file="test.conf" -split-input-file |  FileCheck %s

FIRRTL version 4.0.0
circuit test: %[[
  {
    "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target": "test.test"
  }
]]
  public module test:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>[4]
    input wMask: UInt<1>[4]
    input wData: UInt<8>[4]

    mem memory:
      data-type => UInt<8>[4]
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    ; All of these are unified together
    connect memory.r.clk, clock
    connect memory.r.en, rEn
    connect memory.r.addr, rAddr
    connect rData, memory.r.data

    connect memory.w.clk, clock
    connect memory.w.en, rEn
    connect memory.w.addr, rAddr
    ; These two are split
    connect memory.w.mask, wMask
    connect memory.w.data, wData

; CHECK-NOT: module memory_ext

; CHECK-LABEL: FILE "metadata{{[/\]}}seq_mems.json"
; CHECK:      [
; CHECK-NEXT:   {
; CHECK-NEXT:     "module_name": "memory_ext",
; CHECK-NEXT:     "depth": 16,
; CHECK-NEXT:     "width": 32,
; CHECK-NEXT:     "masked": true,
; CHECK-NEXT:     "read": 1,
; CHECK-NEXT:     "write": 1,
; CHECK-NEXT:     "readwrite": 0,
; CHECK-NEXT:     "ruw_behavior": "Undefined",
; CHECK-NEXT:     "mask_granularity": 8,
; CHECK-NEXT:     "extra_ports": [],
; CHECK-NEXT:     "hierarchy": [
; CHECK-NEXT:       "test.memory.memory_ext"
; CHECK-NEXT:     ]
; CHECK-NEXT:   }
; CHECK-NEXT: ]

; CHECK: test.conf

; CHECK-NOT: name memory_ext depth 16 width 32 ports mwrite,read mask_gran 8 ruw
; CHECK:     name memory_ext depth 16 width 32 ports mwrite,read mask_gran 8

;// -----

FIRRTL version 4.0.0
circuit test: %[[
  {
    "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target": "test.test"
  }
]]
  public module test:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>[4]
    input wMask: UInt<1>[4]
    input wData: UInt<8>[4]

    mem memory:
      data-type => UInt<8>[4]
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => new

    ; All of these are unified together
    connect memory.r.clk, clock
    connect memory.r.en, rEn
    connect memory.r.addr, rAddr
    connect rData, memory.r.data

    connect memory.w.clk, clock
    connect memory.w.en, rEn
    connect memory.w.addr, rAddr
    ; These two are split
    connect memory.w.mask, wMask
    connect memory.w.data, wData

; CHECK-LABEL: seq_mems.json
; CHECK:       "ruw_behavior": "New"

; CHECK: test.conf
; CHECK: name memory_ext depth 16 width 32 ports mwrite,read mask_gran 8 ruw New

;// -----

FIRRTL version 4.0.0
circuit test: %[[
  {
    "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target": "test.test"
  }
]]
  public module test:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>[4]
    input wMask: UInt<1>[4]
    input wData: UInt<8>[4]

    mem memory:
      data-type => UInt<8>[4]
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => old

    ; All of these are unified together
    connect memory.r.clk, clock
    connect memory.r.en, rEn
    connect memory.r.addr, rAddr
    connect rData, memory.r.data

    connect memory.w.clk, clock
    connect memory.w.en, rEn
    connect memory.w.addr, rAddr
    ; These two are split
    connect memory.w.mask, wMask
    connect memory.w.data, wData

; CHECK-LABEL: seq_mems.json
; CHECK:       "ruw_behavior": "Old"

; CHECK: test.conf
; CHECK: name memory_ext depth 16 width 32 ports mwrite,read mask_gran 8 ruw Old
