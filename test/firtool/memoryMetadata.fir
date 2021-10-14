; RUN: firtool %s --format=fir  --annotation-file %s.anno.json --repl-seq-mem --verilog |  FileCheck %s

; CHECK-LABEL: external module FIRRTLMem_1_1_0_8_16_1_1_1_0_1_a

; CHECK-LABEL: external module FIRRTLMem_1_1_0_8_32_1_1_1_0_1_a


circuit test:
  module tbMemModule1:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem tbMemoryKind1:
      data-type => UInt<8>
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    tbMemoryKind1.r.clk <= clock
    tbMemoryKind1.r.en <= rEn
    tbMemoryKind1.r.addr <= rAddr
    rData <= tbMemoryKind1.r.data
    tbMemoryKind1.w.clk <= clock
    tbMemoryKind1.w.en <= rEn
    tbMemoryKind1.w.addr <= rAddr
    tbMemoryKind1.w.mask <= wMask
    tbMemoryKind1.w.data <= wData

; CHECK-LABEL: module tbMemModule1
; CHECK:  FIRRTLMem_1_1_0_8_16_1_1_1_0_1_a tbMemoryKind1

  module dutModule2:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem dutMemory:
      data-type => UInt<8>
      depth => 32
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    dutMemory.r.clk <= clock
    dutMemory.r.en <= rEn
    dutMemory.r.addr <= rAddr
    rData <= dutMemory.r.data
    dutMemory.w.clk <= clock
    dutMemory.w.en <= rEn
    dutMemory.w.addr <= rAddr
    dutMemory.w.mask <= wMask
    dutMemory.w.data <= wData

; CHECK-LABEL: module dutModule2
; CHECK:  FIRRTLMem_1_1_0_8_32_1_1_1_0_1_a

  module hier1: 
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem tbMemoryKind1:
      data-type => UInt<8>
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    tbMemoryKind1.r.clk <= clock
    tbMemoryKind1.r.en <= rEn
    tbMemoryKind1.r.addr <= rAddr
    rData <= tbMemoryKind1.r.data
    tbMemoryKind1.w.clk <= clock
    tbMemoryKind1.w.en <= rEn
    tbMemoryKind1.w.addr <= rAddr
    tbMemoryKind1.w.mask <= wMask
    tbMemoryKind1.w.data <= wData

; CHECK=LABEL: module hier1
; CHECK:  FIRRTLMem_1_1_0_8_16_1_1_1_0_1_a tbMemoryKind1


  module hier2: 
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>


    inst m of tbMemModule1
    m.clock <= clock
    m.rAddr <= rAddr
    m.rEn <= rEn
    rData <= m.rData
    m.wMask <= wMask
    m.wData <= wData

  module test: 
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>


    inst h1 of hier1
    h1.clock <= clock
    h1.rAddr <= rAddr
    h1.rEn <= rEn
    rData <= h1.rData
    h1.wMask <= wMask
    h1.wData <= wData

    inst h2 of hier2
    h2.clock <= clock
    h2.rAddr <= rAddr
    h2.rEn <= rEn
    rData <= h2.rData
    h2.wMask <= wMask
    h2.wData <= wData

    inst m2 of dutModule
    m2.clock <= clock
    m2.rAddr <= rAddr
    m2.rEn <= rEn
    rData <= m2.rData
    m2.wMask <= wMask
    m2.wData <= wData

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
    m.clock <= clock
    m.rAddr <= rAddr
    m.rEn <= rEn
    rData <= m.rData
    m.wMask <= wMask
    m.wData <= wData



; CHECK-LABEL: module dutModule
; CHECK: dutModule2 [[m:.+]] (

; CHECK-LABEL:      FILE "metadata/tb_seq_mems.json"
; CHECK:      [{"module_name":"FIRRTLMem_1_1_0_8_16_1_1_1_0_1_a","depth":16,"width":8,"masked":"true",
; CHECK-SAME: "read":"true","write":"true","readwrite":"false",
; CHECK_SAME: "mask_granularity":8,"extra_ports":[],
; CHECK-SAME: hierarchy":["test.[[h2]].[[m]]","test.[[h1]]"

; CHECK-LABEL:       FILE "metadata/seq_mems.json"
; CHECK:       [{"module_name":"FIRRTLMem_1_1_0_8_32_1_1_1_0_1_a","depth":32,"width":8,"masked":"true",
; CHECK-SAME:  "read":"true","write":"true","readwrite":"false",
; CHECK_SAME:  "mask_granularity":8,"extra_ports":[],
; CHECK-SAME:  "hierarchy":["test.[[m2]].[[m]]"

; CHECK-LABEL: FILE "metadata/dutModule.conf"
; CHECK: name FIRRTLMem_1_1_0_8_32_1_1_1_0_1_a depth 32 width 8 ports mwrite,read mask_gran 8
