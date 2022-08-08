# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import generator, types
from pycde.common import Clock, Input
from pycde.constructs import Reg, Wire
from pycde.testing import unittestmodule


@unittestmodule()
class WireTypeTest:
  In = Input(types.i8)

  @generator
  def create(ports):
    w = Wire(types.i4)
    # CHECK: TypeError: Cannot assign i8 to i4
    w.assign(ports.In)


# -----


@unittestmodule()
class WireDoubleAssignTest:
  In = Input(types.i8)

  @generator
  def create(ports):
    w = Wire(types.i8)
    w.assign(ports.In)
    # CHECK: ValueError: Cannot assign value to Wire twice.
    w.assign(ports.In)


# -----


@unittestmodule()
class RegTypeTest:
  clk = Clock()
  In = Input(types.i8)

  @generator
  def create(ports):
    r = Reg(types.i4)
    # CHECK: TypeError: Cannot assign i8 to i4
    r.assign(ports.In)


# -----


@unittestmodule()
class RegDoubleAssignTest:
  Clk = Clock()
  In = Input(types.i8)

  @generator
  def create(ports):
    r = Reg(types.i8)
    r.assign(ports.In)
    # CHECK: ValueError: Cannot assign value to Reg twice.
    r.assign(ports.In)
