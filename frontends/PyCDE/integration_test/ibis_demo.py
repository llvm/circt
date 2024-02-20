# REQUIRES: esi-runtime, esi-cosim, rtl-sim, questa, ibis
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% %s %t 2>&1

# Ibis does not currently simulate with Verilator.
# RUN: esi-cosim.py --sim questa -- %PYTHON% %S/test_software/ibis_foo.py cosim env

from symbol import func_type
import pycde
from pycde import Input, Module, generator, esi
from pycde.module import Metadata
from pycde.common import Clock
from pycde.bsp import cosim
from pycde.ibis import IbisClass, method
from pycde.types import Array, Bits, UInt

from pathlib import Path
import sys

__dirname__ = Path(__file__).parent


class DemoTop(IbisClass):
  # ibis -t=sv --circt --write-circt-ir --no-inspection \
  #      --no-control-inspection --no-wrapper --no-debug-view \
  #      --base-library $IBIS_LIB/base.pd --import-dir $IBIS_LIB/
  # ibistool --lo --ir ibis_esi_demoDemoTop.mlir > ibis_esi_demoDemoTop.lo.mlir
  src_file = "ibis_esi_demoDemoTop.lo.mlir"
  support_files = "support_files.f"

  metadata = Metadata(version="0.1",
                      summary="A demonstration of ESI and Ibis",
                      misc={
                          "crcWidth": 64,
                          "style": "stupid"
                      })

  @method
  def add(self, a: UInt(8), b: UInt(8), arr: Array(UInt(8), 16)) -> UInt(8):
    pass

  @method
  def compute_crc(
      self, identifier: UInt(8), input: Array(UInt(8), 64),
      input_bytes: UInt(8), reset: UInt(8)
  ) -> UInt(32):
    pass


class IbisTestSystem(Module):
  clk = Clock()
  rst = Input(Bits(1))

  @generator
  def build(ports):
    add = esi.FuncService.get(esi.AppID("add"), func_type=DemoTop.add.func_type)
    crc = esi.FuncService.get(esi.AppID("crc"),
                              func_type=DemoTop.compute_crc.func_type)
    DemoTop(clk=ports.clk,
            rst=ports.rst,
            appid=esi.AppID("demo"),
            add=add,
            compute_crc=crc)


if __name__ == "__main__":

  s = pycde.System(cosim.CosimBSP(IbisTestSystem),
                   name="IbisTest",
                   output_directory=sys.argv[1])
  s.generate()
  s.print(file=open("ibis_test.mlir", "w"))
  s.compile()
  s.package()
