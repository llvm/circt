# REQUIRES: bindings_python
# RUN: %PYTHON% %s

import circt
from circt import ir
from circt.dialects import arith

# Ensure that the `OpOperand`s returned as the `uses` list of a value have a
# concrete subclass of `OpView`, not just `OpView` itself. This was a regression
# that happened upstream, where you could no longer use
# `isinstance(use.owner, AddIOp)` to check whether a user of a value is a
# specific op.
with ir.Context() as ctx, ir.Location.unknown() as loc:
  circt.register_dialects(ctx)
  module = ir.Module.parse("""
    %0 = arith.constant 0 : i8
    %1 = arith.addi %0, %0 : i8
  """)
  zero, add = list(module.body)
  use = list(zero.result.uses)[0]
  assert use.owner == add
  assert isinstance(add, arith.AddIOp)
  assert isinstance(use.owner, arith.AddIOp)
