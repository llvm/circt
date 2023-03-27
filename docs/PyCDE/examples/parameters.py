from pycde import Input, Output, Module, System
from pycde import generator, modparams
from pycde.types import Bits


@modparams
def AddInts(width: int):

  class AddInts(Module):
    a = Input(Bits(width))
    b = Input(Bits(width))
    c = Output(Bits(width))

  return AddInts


class Top(Module):
  a = Input(Bits(32))
  b = Input(Bits(32))
  c = Output(Bits(32))

  @generator
  def construct(self):
    add_ints_m = AddInts(32)
    add_ints = add_ints_m(a=self.a, b=self.b)
    self.c = add_ints.c


system = System(Top, name="ExampleParams")
system.compile()
