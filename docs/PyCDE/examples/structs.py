from pycde import Input, Output, generator, System, Module
from pycde.types import Bits
from pycde.signals import Struct, BitsSignal


class ExStruct(Struct):
  a: Bits(4)
  b: Bits(32)

  def get_b_xor(self, x: int) -> BitsSignal:
    return self.b ^ Bits(32)(x)


class StructExample(Module):
  inp1 = Input(ExStruct)
  out1 = Output(Bits(32))
  out2 = Output(Bits(4))

  @generator
  def build(self):
    self.out1 = self.inp1.get_b_xor(5432)
    self.out2 = self.inp1.a


system = System(StructExample)
system.compile()
