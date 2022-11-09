; RUN: firtool %s --format=fir | FileCheck %s  --check-prefixes COMMON,NOLOWER
; RUN: firtool %s --format=fir -lower-memories | FileCheck %s --check-prefixes COMMON,LOWER

circuit ByteEnableMemory:
  module ByteEnableMemory :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip readAddr : UInt<16>, dataOut : UInt<8>[2], flip readEnable : UInt<1>, flip dataIn : UInt<8>[2], flip writeAddr : UInt<16>, flip writeMask : UInt<1>[2]}

    smem mem : UInt<8>[2] [2] @[MemoryMask.scala 24:24]
    node _T = bits(io.writeAddr, 0, 0)
    write mport MPORT = mem[_T], clock
    when io.writeMask[0] :
      MPORT[0] <= io.dataIn[0]
    when io.writeMask[1] :
      MPORT[1] <= io.dataIn[1]
    wire _WIRE : UInt @[MemoryMask.scala 27:25]
    _WIRE is invalid @[MemoryMask.scala 27:25]
    when io.readEnable : @[MemoryMask.scala 27:25]
      _WIRE <= io.readAddr @[MemoryMask.scala 27:25]
      node _T_1 = or(_WIRE, UInt<1>("h0")) @[MemoryMask.scala 27:25]
      node _T_2 = bits(_T_1, 0, 0) @[MemoryMask.scala 27:25]
      read mport MPORT_1 = mem[_T_2], clock @[MemoryMask.scala 27:25]
    io.dataOut <= MPORT_1 @[MemoryMask.scala 27:14]

;COMMON: module mem_
;COMMON: W0_clk,
;NOLOWER: input [15:0] W0_data,
;LOWER: input [7:0] W0_data,
;NOLOWER: input [1:0] W0_mask,
;LOWER: output [7:0] R0_data);
;NOLOWER: output [15:0] R0_data);
;LOWER: reg [7:0] Memory[0:1];
;NOLOWER: reg [15:0] Memory[0:1];
;COMMON always @(posedge R0_clk) begin
;NOLOWER if (W0_en & W0_mask[0])
;COMMON module ByteEnableMemory(
;NOLOWER mem_combMem mem_ext (
;LOWER mem_0_combMem mem_0_ext (
;LOWER mem_0_combMem mem_1_ext (
;COMMON endmodule