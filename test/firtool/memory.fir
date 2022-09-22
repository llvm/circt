; RUN: firtool %s --format=fir --ir-fir | FileCheck %s

circuit Qux: %[[{
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~Qux|Qux"
  }, 
  {
    "class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"
  }]]
  module Qux: 
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
    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    ; This is split
    rData <= memory.r.data

    ; All of these are unified together
    memory.w.clk <= clock
    memory.w.en <= wEn
    memory.w.addr <= wAddr
    ; These two are split
    memory.w.mask <= wMask
    memory.w.data <= wData

    ; All of these are unified together
    memory.rw.clk <= clock
    memory.rw.en <= rwEn
    memory.rw.addr <= rwAddr
    memory.rw.wmode <= rwMode
    ; All of these are split
    memory.rw.wmask <= rwMask
    memory.rw.wdata <= rwDataIn
    rwDataOut <= memory.rw.rdata



; This test is quite fragile, both as written, and in that it depends on
; multiple passes.  It should be replaced with a narrower test.

; CHECK:      %memory = firrtl.reg  %clock  {firrtl.random_init_start = 0 : ui64, preserve_type} : !firrtl.vector<uint<8>, 4>
; CHECK:      %addr = firrtl.reg  %clock  {firrtl.random_init_start = 33 : ui64} : !firrtl.uint<2>
; CHECK:      %[[v4:.+]] = firrtl.subaccess %memory[%addr] : !firrtl.vector<uint<8>, 4>, !firrtl.uint<2>
; CHECK:      %[[v5:.+]] = firrtl.subaccess %memory[%rwAddr] : !firrtl.vector<uint<8>, 4>, !firrtl.uint<2>
; CHECK:      %[[v43:.+]] = firrtl.andr %wAddr : (!firrtl.uint<2>) -> !firrtl.uint<1>
; CHECK:      %[[v44:.+]] = firrtl.subindex %memory[3] : !firrtl.vector<uint<8>, 4>
; CHECK:      %[[v45:.+]] = firrtl.mux(%[[v43]], %wData, %[[v26:.+]]) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
; CHECK:      %[[v46:.+]] = firrtl.mux(%wMask, %[[v45]], %[[v26]]) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
; CHECK:      %[[v47:.+]] = firrtl.mux(%wEn, %[[v46]], %[[v26]]) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
; CHECK:      firrtl.strictconnect %[[v44]], %[[v47]] : !firrtl.uint<8>
