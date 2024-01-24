import pycde
from pycde import Input, Module, generator, esi
from pycde.common import Clock
from pycde.bsp import cosim
from pycde.ibis import IbisClass, method
from pycde.types import Array, Bits, UInt

import sys


class Foo(IbisClass):
  # pd -t=sv --circt --write-circt-ir --no-wrapper ~/empty.pd
  # ibistool --lo --ir  emptyFoo.mlir > emptyFoo.lo.mlir
  src_file = "/home/jodemme/ibis_play/empty/emptyFoo.lo.mlir"

  @method
  def add(self, a: UInt(8), b: UInt(8), arr: Array(UInt(8), 16)) -> UInt(8):
    pass


class IbisTestSystem(Module):
  clk = Clock()
  rst = Input(Bits(1))

  @generator
  def build(ports):
    foo = Foo(clk=ports.clk, rst=ports.rst, appid=esi.AppID("empty"))
    esi.FuncService.call(foo.add, esi.AppID("add"))


if __name__ == "__main__":
  s = pycde.System(cosim.CosimBSP(IbisTestSystem),
                   name="IbisTest",
                   output_directory=sys.argv[1])
  s.generate()
  s.print(file=open("ibis_test.mlir", "w"))
  s.compile()
  s.package()
