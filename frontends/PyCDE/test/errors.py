from pycde.module import externmodule

# RUN: %PYTHON% %s | FileCheck %s

try:

  @externmodule
  def foo(*args):
    pass
except Exception as e:
  print(e)
  # CHECK: Module parameter definitions cannot have *args

try:

  @externmodule
  def bar(**kwargs):
    pass
except Exception as e:
  print(e)
  # CHECK: Module parameter definitions cannot have **kwargs
