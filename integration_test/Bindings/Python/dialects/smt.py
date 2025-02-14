# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import smt
from circt.ir import Context, Location, Module, InsertionPoint

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    true = smt.constant(True)
    false = smt.constant(False)
  # CHECK: smt.constant true
  # CHECK: smt.constant false
  print(m)
