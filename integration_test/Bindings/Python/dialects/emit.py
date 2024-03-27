# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import emit
from circt.ir import Context, Location, Module, InsertionPoint, IntegerAttr, IntegerType

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    fileOp = emit.file("foo.sv")
    # CHECK:      emit.file
    # CHECK-SAME:   file_name = "foo.sv"
    print(fileOp)
