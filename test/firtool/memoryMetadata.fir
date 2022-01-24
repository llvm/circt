; RUN: firtool %s --format=fir  --annotation-file %s.anno.json --repl-seq-mem --repl-seq-mem-file="metadata/dutModule.conf" --verilog |  FileCheck %s


; CHECK-LABEL: external module tbMemoryKind1_ext

; CHECK-LABEL: external module dutMemory_ext

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
; CHECK:   tbMemoryKind1_ext tbMemoryKind1

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
; CHECK:  dutMemory_ext dutMemory

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
; CHECK:  tbMemoryKind1_ext tbMemoryKind1


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
; CHECK: [{"module_name":"tbMemoryKind1_ext","depth":16,"width":8,
; CHECK:          "masked":"true","read":"true","write":"true","readwrite":"false"
; CHECK:          ,"mask_granularity":8,"extra_ports":[],"hierarchy":
; CHECK:          ["test.h2.m.tbMemoryKind1","test.h1.tbMemoryKind1"]}

; CHECK-LABEL:       FILE "metadata/seq_mems.json"
; CHECK:  [{"module_name":"dutMemory_ext","depth":32,"width":8,"masked":"true","read":"true",
; CHECK:  "write":"true","readwrite":"false","mask_granularity":8,"extra_ports":[],
; CHECK:  "hierarchy":["test.m2.m.dutMemory"]}]

; CHECK-LABEL: FILE "metadata/dutModule.conf"
; CHECK: name dutMemory_ext depth 32 width 8 ports mwrite,read mask_gran 8
; CHECK: name tbMemoryKind1_ext depth 16 width 8 ports mwrite,read mask_gran 8
