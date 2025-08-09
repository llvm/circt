# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Output, generator, Module
from pycde.types import Bits
from pycde.testing import unittestmodule

from pycde.bit_pat import BitPat


@unittestmodule()
class BitPatError(Module):
  out1 = Output(Bits(32))

  @generator
  def build(self):
    # CHECK: TypeError: Can only convert BitPat with no don't cares to Bits
    self.out1 = BitPat("b000000000????0000000000000010011").as_bits()


# -----


@unittestmodule()
class BitPatError(Module):
  out1 = Output(Bits(1))

  @generator
  def build(self):
    # CHECK: TypeError: BitPat can only be compared to BitVectorSignal
    self.out1 = BitPat("b000000000????0000000000000010011") == 19


# -----


@unittestmodule()
class BitPatError(Module):
  out1 = Output(Bits(1))

  @generator
  def build(self):
    # CHECK: TypeError: BitPat can only be compared to BitVectorSignal
    self.out1 = BitPat("b000000000????0000000000000010011") != 19
