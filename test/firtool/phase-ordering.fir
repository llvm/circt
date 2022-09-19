; RUN: firtool %s --format=fir --ir-fir | FileCheck %s

; Temporary wires should not be introduced by type lowering, and if they are,
; they should be cleaned up by canonicalize.

; CHECK-LABEL: firrtl.module @Issue794
; CHECK-SAME: (in %clock: !firrtl.clock,
; CHECK:       %memory = firrtl.reg  %clock  {firrtl.random_init_start = 0 : ui64, preserve_type = true} : !firrtl.vector<uint<8>, 4>
; CHECK:      %[[v0:.+]] = firrtl.subaccess %memory[%rAddr] : !firrtl.vector<uint<8>, 4>, !firrtl.uint<2>
; CHECK:      firrtl.strictconnect %memory_r_data, %[[v0]] : !firrtl.uint<8>
; CHECK:      %[[v1:.+]] = firrtl.orr %wAddr_1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
; CHECK:      %[[v2:.+]] = firrtl.not %[[v1]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
; CHECK:      %[[v3:.+]] = firrtl.subindex %memory[0] : !firrtl.vector<uint<8>, 4>
; CHECK:      %[[v4:.+]] = firrtl.mux(%[[v2]], %wData_1, %wData_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
; CHECK:      %[[v5:.+]] = firrtl.mux(%wMask_1, %[[v4]], %wData_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
; CHECK:      %[[v6:.+]] = firrtl.mux(%wEn_1, %[[v5]], %wData_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
; CHECK:      firrtl.strictconnect %[[v3]], %[[v6]] : !firrtl.uint<8>
circuit Issue794: %[[{
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~Issue794|Issue794"
  }, 
  {
    "class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"
  }]]
  module Issue794:
    input clock: Clock
    input rAddr: UInt<2>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wAddr: UInt<2>[2]
    input wEn: UInt<1>[2]
    input wMask: UInt<1>[2]
    input wData: UInt<8>[2]

    mem memory:
      data-type => UInt<8>
      depth => 4
      reader => r
      writer => w0,w1
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    rData <= memory.r.data

    memory.w0.clk <= clock
    memory.w0.en <= wEn[0]
    memory.w0.addr <= wAddr[0]
    memory.w0.mask <= wMask[0]
    memory.w0.data <= wData[0]

    memory.w1.clk <= clock
    memory.w1.en <= wEn[1]
    memory.w1.addr <= wAddr[1]
    memory.w1.mask <= wMask[1]
    memory.w1.data <= wData[1]
