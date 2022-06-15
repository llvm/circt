# RUN: py-split-input-file %s | FileCheck %s

from pycde.module import externmodule


# CHECK: TypeError: Module parameter definitions cannot have *args
@externmodule
def foo(*args):
  pass


# -----


# CHECK: TypeError: Module parameter definitions cannot have **kwargs
@externmodule
def bar(**kwargs):
  pass
