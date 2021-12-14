import pycde
from pycde.dialects import comb, hw
from pycde import dim, module, generator, Input, Output


@module
def MyModule(SIZE: int):

  class Mod:
    inp = Input(dim(SIZE))
    out = Output(dim(SIZE))

    @generator
    def construct(mod):
      c1 = hw.ConstantOp(dim(SIZE), 1)
      eq = comb.EqOp(c1, mod.inp)
      a1 = hw.ArrayCreateOp([eq, eq])
      a2 = hw.ArrayCreateOp([eq, eq])
      combined = hw.ArrayConcatOp(a1, a2)
      mod.out = hw.BitcastOp(dim(SIZE), combined)

  return Mod


mymod = MyModule(4)
module = pycde.System([mymod], name="mymod")
module.generate()
module.print()
