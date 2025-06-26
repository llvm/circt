; RUN: firtool %s --format=fir  --annotation-file %s.anno.json --verilog |  FileCheck %s
; XFAIL: true

FIRRTL version 4.0.0
circuit test:
  module memoryTest1:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem memory:
      data-type => UInt<8>
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

  module memoryTest2:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem memory:
      data-type => UInt<8>
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


  public module test:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>


    inst m of memoryTest1
    connect m.clock, clock
    connect m.rAddr, rAddr
    connect m.rEn, rEn
    connect rData, m.rData
    connect m.wMask, wMask
    connect m.wData, wData

    inst signed of memoryP
    connect signed.clock, clock
    connect signed.rAddr, rAddr
    connect signed.rEn, rEn
    connect rData, signed.rData
    connect signed.wMask, wMask
    connect signed.wData, wData


  module memoryP:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>


    inst m of memoryTest2
    connect m.clock, clock
    connect m.rAddr, rAddr
    connect m.rEn, rEn
    connect rData, m.rData
    connect m.wMask, wMask
    connect m.wData, wData

; CHECK-LABEL: module test
; CHECK: memoryP [[signed:.+]] (

; CHECK-LABEL: module memoryP
; CHECK: memoryTest2 [[m:.+]] (

; CHECK:      FILE "metadata/tb_seq_mems.json"
; CHECK:      [{"module_name":"memory","depth":16,"width":8,"masked":"true",
; CHECK-SAME: "read":"true","write":"true","readwrite":"false",
; CHECK-SAME: "ruw_behavior": "Undefined","mask_granularity":8,"extra_ports":[],
; CHECK-SAME: "hierarchy":["test.[[m]]"]}]

; CHECK:       FILE "metadata/seq_mems.json"
; CHECK:       [{"module_name":"memory","depth":16,"width":8,"masked":"true",
; CHECK-SAME:  "read":"true","write":"true","readwrite":"false",
; CHECK-SAME:  "ruw_behavior": "Undefined","mask_granularity":8,"extra_ports":[],
; CHECK-SAME:  "hierarchy":["test.[[signed]].[[m]]"],
; CHECK-SAME:  "verification_only_data":{
; CHECK-SAME:  "test.[[signed]].[[m]]":{"baseAddress":1073741824,
; CHECK-SAME:  "dataBits":8,"eccBits":0,"eccIndices":[],"eccScheme":"none"}}}]
