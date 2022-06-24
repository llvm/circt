# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Input, Output, types
from pycde.module import externmodule, generator
from pycde.testing import unittestmodule


# CHECK: TypeError: Module parameter definitions cannot have *args
@externmodule
def foo(*args):
  pass


# -----


# CHECK: TypeError: Module parameter definitions cannot have **kwargs
@externmodule
def bar(**kwargs):
  pass


# -----


@unittestmodule()
class ClkError:
  a = Input(types.i32)

  @generator
  def build(ports):
    # CHECK: ValueError: If 'clk' not specified, must be in clock block
    ports.a.reg()
