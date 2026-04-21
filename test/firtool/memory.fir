; RUN: firtool %s --format=fir -disable-mem-randomization | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_MEM
; RUN: firtool %s --format=fir -disable-reg-randomization | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_REG
; RUN: firtool %s --format=fir -disable-mem-randomization --disable-reg-randomization | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_MEM --implicit-check-not RANDOMIZE_REG
; RUN: firtool %s --format=fir -disable-all-randomization | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_MEM --implicit-check-not RANDOMIZE_REG

FIRRTL version 4.0.0
circuit Qux: %[[{
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~Qux|Qux"
  },
  {
    "class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"
  }]]
  public module Qux:
    input clock: Clock
    input rAddr: UInt<2>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wAddr: UInt<2>
    input wEn: UInt<1>
    input wMask: UInt<1>
    input wData: UInt<8>
    input rwEn: UInt<1>
    input rwMode: UInt<1>
    input rwAddr: UInt<2>
    input rwMask: UInt<1>
    input rwDataIn: UInt<8>
    output rwDataOut: UInt<8>

    mem memory:
      data-type => UInt<8>
      depth => 4
      reader => r
      writer => w
      readwriter => rw
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    ; All of these are unified together
    connect memory.r.clk, clock
    connect memory.r.en, rEn
    connect memory.r.addr, rAddr
    ; This is split
    connect rData, memory.r.data

    ; All of these are unified together
    connect memory.w.clk, clock
    connect memory.w.en, wEn
    connect memory.w.addr, wAddr
    ; These two are split
    connect memory.w.mask, wMask
    connect memory.w.data, wData

    ; All of these are unified together
    connect memory.rw.clk, clock
    connect memory.rw.en, rwEn
    connect memory.rw.addr, rwAddr
    connect memory.rw.wmode, rwMode
    ; All of these are split
    connect memory.rw.wmask, rwMask
    connect memory.rw.wdata, rwDataIn
    connect rwDataOut, memory.rw.rdata

; COMMON-LABEL: module Qux
