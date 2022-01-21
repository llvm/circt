; RUN: firtool %s --format=fir  --annotation-file %s.anno.json -emit-metadata -repl-seq-mem -repl-seq-mem-file="metadata/test.conf" --verilog |  FileCheck %s

circuit test:
  module test:
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
    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    rData <= memory.r.data

    memory.w.clk <= clock
    memory.w.en <= rEn
    memory.w.addr <= rAddr
    ; These two are split
    memory.w.mask <= wMask
    memory.w.data <= wData

; CHECK: external module memory_ext

; CHECK: metadata/seq_mems.json

; CHECK: [{"module_name":"memory_ext","depth":16,"width":32,"masked":"true","read":"true","write":"true","readwrite":"false","mask_granularity":8,"extra_ports":[],"hierarchy":[]}]

; CHECK: metadata/test.conf

; CHECK: name memory_ext depth 16 width 32 ports mwrite,read mask_gran 8
