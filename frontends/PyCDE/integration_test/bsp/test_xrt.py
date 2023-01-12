# REQUIRES: xrt
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t genhw 2>&1

import pycde
from pycde import (Clock, Input, module, generator, types)
from pycde.bsp import XrtBSP

import os
import sys


@module
class Top:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    pass


gendir = sys.argv[1]
command = sys.argv[2] if len(sys.argv) == 3 else "genhw"
if command == "genhw":
  s = pycde.System(XrtBSP(Top),
                   name="ESILoopback",
                   output_directory=gendir,
                   sw_api_langs=["python"])
  s.run_passes(debug=True)
  s.compile()
  s.package()

elif command == "test":
  sys.path.append(os.path.join(gendir, "runtime"))
  import ESILoopback as esi_sys
  from ESILoopback.xrt import Xrt

  acc_conn = Xrt(hw_emu=True, xclbin="ESILoopback.hw_emu.xclbin")
  top = esi_sys.top(acc_conn)

  print("Success: all tests pass!")
