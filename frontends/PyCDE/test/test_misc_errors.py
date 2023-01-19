# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Input, generator, dim, Module
from pycde.constructs import Mux
from pycde.testing import unittestmodule


@unittestmodule()
class Mux1(Module):

  In = Input(dim(3, 4, 5))
  Sel = Input(dim(8))

  @generator
  def create(ports):
    # CHECK: TypeError: 'Sel' bit width must be clog2 of number of inputs
    Mux(ports.Sel, ports.In[3], ports.In[1])


# -----


@unittestmodule()
class Mux2(Module):

  Sel = Input(dim(8))

  @generator
  def create(ports):
    # CHECK: ValueError: 'Mux' must have 1 or more data input
    Mux(ports.Sel)
