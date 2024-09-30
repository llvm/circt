# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.support import walk_with_filter
from circt.dialects import hw
from circt.ir import Context, Module, WalkOrder, WalkResult


def test_walk_with_filter():
  ctx = Context()
  circt.register_dialects(ctx)
  module = Module.parse(
      r"""
    builtin.module {
      hw.module @f() {
        hw.output
      }
    }
  """,
      ctx,
  )

  def callback(op):
    print(op.name)
    return WalkResult.ADVANCE

  # Test post-order walk.
  # CHECK:       Post-order
  # CHECK-NEXT:  hw.output
  # CHECK-NEXT:  hw.module
  # CHECK-NOT:  builtin.module
  print("Post-order")
  walk_with_filter(module.operation, [hw.HWModuleOp, hw.OutputOp], callback,
                   WalkOrder.POST_ORDER)

  # Test pre-order walk.
  # CHECK-NEXT:  Pre-order
  # CHECK-NOT:  builtin.module
  # CHECK-NEXT:  hw.module
  # CHECK-NEXT:  hw.output
  print("Pre-order")
  walk_with_filter(module.operation, [hw.HWModuleOp, hw.OutputOp], callback,
                   WalkOrder.PRE_ORDER)

  # Test interrupt.
  # CHECK-NEXT:  Interrupt post-order
  # CHECK-NEXT:  hw.output
  print("Interrupt post-order")

  def interrupt_callback(op):
    print(op.name)
    return WalkResult.INTERRUPT

  walk_with_filter(module.operation, [hw.OutputOp], interrupt_callback,
                   WalkOrder.POST_ORDER)

  # Test exception.
  # CHECK: Exception
  # CHECK-NEXT: hw.output
  # CHECK-NEXT: Exception raised
  print("Exception")

  def exception_callback(op):
    print(op.name)
    raise ValueError
    return WalkResult.ADVANCE

  try:
    walk_with_filter(module.operation, [hw.OutputOp], exception_callback,
                     WalkOrder.POST_ORDER)
  except RuntimeError:
    print("Exception raised")


test_walk_with_filter()
