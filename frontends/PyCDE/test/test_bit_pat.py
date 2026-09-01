# RUN: %PYTHON% %s | FileCheck %s

from pycde import Input, Output, generator, Module
from pycde.types import Bits, SInt, UInt
from pycde.testing import unittestmodule

from pycde.bit_pat import BitPat, dict_lookup

# CHECK-LABEL:  msft.module @TestBitPat {} (%inp1: ui36, %inp2: si25, %inp3: i32) -> (out1: i32, out2: i32, match: i1, no_match: i1)
# CHECK-NEXT:     %c3_i32 = hw.constant 3 : i32
# CHECK-NEXT:     %c28799_i32 = hw.constant 28799 : i32
# CHECK-NEXT:     %c4099_i32 = hw.constant 4099 : i32
# CHECK-NEXT:     %c28799_i32_0 = hw.constant 28799 : i32
# CHECK-NEXT:     %c8195_i32 = hw.constant 8195 : i32
# CHECK-NEXT:     %c28799_i32_1 = hw.constant 28799 : i32
# CHECK-NEXT:     %c16387_i32 = hw.constant 16387 : i32
# CHECK-NEXT:     %c28799_i32_2 = hw.constant 28799 : i32
# CHECK-NEXT:     %c20483_i32 = hw.constant 20483 : i32
# CHECK-NEXT:     %c28799_i32_3 = hw.constant 28799 : i32
# CHECK-NEXT:     %c2989_i32 = hw.constant 2989 : i32
# CHECK-NEXT:     %c64206_i32 = hw.constant 64206 : i32
# CHECK-NEXT:     %c3054_i32 = hw.constant 3054 : i32
# CHECK-NEXT:     %c48879_i32 = hw.constant 48879 : i32
# CHECK-NEXT:     %c3245_i32 = hw.constant 3245 : i32
# CHECK-NEXT:     %0 = hwarith.cast %inp1 : (ui36) -> i32
# CHECK-NEXT:     %1 = comb.and bin %0, %c28799_i32 : i32
# CHECK-NEXT:     %2 = comb.icmp bin eq %c28799_i32, %1 : i32
# CHECK-NEXT:     %3 = hwarith.cast %inp2 : (si25) -> i32
# CHECK-NEXT:     %4 = comb.and bin %3, %c28799_i32 : i32
# CHECK-NEXT:     %5 = comb.icmp bin ne %c3_i32, %4 : i32
# CHECK-NEXT:     %c19_i32 = hw.constant 19 : i32
# CHECK-NEXT:     %c-1_i32 = hw.constant -1 : i32
# CHECK-NEXT:     %c0_i32 = hw.constant 0 : i32
# CHECK-NEXT:     %6 = comb.and bin %inp3, %c28799_i32 : i32
# CHECK-NEXT:     %7 = comb.icmp bin eq %c28799_i32, %6 : i32
# CHECK-NEXT:     %8 = comb.mux bin %7, %c2989_i32, %c0_i32 {sv.namehint = "mux_None_in0_in1"} : i32
# CHECK-NEXT:     %9 = comb.and bin %inp3, %c28799_i32_0 : i32
# CHECK-NEXT:     %10 = comb.icmp bin eq %c28799_i32_0, %9 : i32
# CHECK-NEXT:     %11 = comb.mux bin %10, %c64206_i32, %8 {sv.namehint = "mux_None_mux_None_in0_in1_in1"} : i32
# CHECK-NEXT:     %12 = comb.and bin %inp3, %c28799_i32_1 : i32
# CHECK-NEXT:     %13 = comb.icmp bin eq %c28799_i32_1, %12 : i32
# CHECK-NEXT:     %14 = comb.mux bin %13, %c3054_i32, %11 {sv.namehint = "mux_None_mux_None_mux_None_in0_in1_in1_in1"} : i32
# CHECK-NEXT:     %15 = comb.and bin %inp3, %c28799_i32_2 : i32
# CHECK-NEXT:     %16 = comb.icmp bin eq %c28799_i32_2, %15 : i32
# CHECK-NEXT:     %17 = comb.mux bin %16, %c48879_i32, %14 {sv.namehint = "mux_None_mux_None_mux_None_mux_None_in0_in1_in1_in1_in1"} : i32
# CHECK-NEXT:     %18 = comb.and bin %inp3, %c28799_i32_3 : i32
# CHECK-NEXT:     %19 = comb.icmp bin eq %c28799_i32_3, %18 : i32
# CHECK-NEXT:     %20 = comb.mux bin %19, %c3245_i32, %17 {sv.namehint = "dict_lookup"} : i32
# CHECK-NEXT:     msft.output %c19_i32, %20, %2, %5 : i32, i32, i1, i1


@unittestmodule()
class TestBitPat(Module):
  inp1 = Input(UInt(36))
  inp2 = Input(SInt(25))
  inp3 = Input(Bits(32))
  out1 = Output(Bits(32))
  out2 = Output(Bits(32))
  match = Output(Bits(1))
  no_match = Output(Bits(1))

  @generator
  def build(self):
    LB = BitPat("b?????????????????000?????0000011")
    LH = BitPat("b?????????????????001?????0000011")
    LW = BitPat("b?????????????????010?????0000011")
    LBU = BitPat("b?????????????????100?????0000011")
    LHU = BitPat("b?????????????????101?????0000011")
    inst_map = {
        LB: Bits(32)(0xBAD),
        LH: Bits(32)(0xFACE),
        LW: Bits(32)(0xBEE),
        LBU: Bits(32)(0xBEEF),
        LHU: Bits(32)(0xCAD)
    }
    self.match = LB == self.inp1
    self.no_match = LB != self.inp2
    self.out1 = BitPat("b00000000000000000000000000010011").as_bits()
    self.out2 = dict_lookup(inst_map, self.inp3, Bits(32)(0x0))
