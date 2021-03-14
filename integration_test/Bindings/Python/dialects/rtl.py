# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import rtl

from mlir.ir import *

with Context() as ctx, Location.unknown():
    circt.register_dialects(ctx)
    i32 = IntegerType.get_signless(32)
    a50 = IntegerAttr.get(i32, 50)
    op = rtl.ConstantOp(i32, a50)

    # CHECK: %{{.+}} = rtl.constant 50 : i32
    op.print()
