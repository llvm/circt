# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

# Build a simple two-module HW design (Top instantiates Inner) and run the
# `hw-flatten-modules` pass on it. After flattening, the instantiation of
# `Inner` should be inlined into `Top`. This also exercises the registration
# of the LLVM dialect's inliner interface, which the FlattenModules pass
# requires when it builds an `mlir::InlinerInterface`.

import circt
from circt.dialects import comb, hw
from circt.ir import (ArrayAttr, Context, InsertionPoint, IntegerType, Location,
                      Module, StringAttr)
from circt.passmanager import PassManager

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i8 = IntegerType.get_signless(8)

  m = Module.create()
  with InsertionPoint(m.body):

    def build_inner(module):
      a, b = module.entry_block.arguments
      hw.OutputOp([comb.add([a, b])])

    inner = hw.HWModuleOp(
        name="Inner",
        input_ports=[("a", i8), ("b", i8)],
        output_ports=[("y", i8)],
        body_builder=build_inner,
        attributes={"sym_visibility": StringAttr.get("private")},
    )
    del inner

    def build_top(module):
      x, = module.entry_block.arguments
      y = hw.instance(
          [i8],
          "inner",
          "Inner",
          [x, x],
          ["a", "b"],
          ["y"],
          parameters=ArrayAttr.get([]),
      )
      hw.OutputOp([y])

    hw.HWModuleOp(name="Top",
                  input_ports=[("x", i8)],
                  output_ports=[("y", i8)],
                  body_builder=build_top)

  # Pre-flatten IR: both modules are still present and `Top` instantiates
  # `Inner`.
  # CHECK-LABEL: === Before hw-flatten-modules ===
  # CHECK: hw.module private @Inner
  # CHECK: hw.module @Top
  # CHECK: hw.instance "inner" @Inner
  print("=== Before hw-flatten-modules ===")
  print(m)

  pm = PassManager.parse("builtin.module(hw-flatten-modules)")
  pm.run(m.operation)

  # After flattening: the `Inner` module is gone and its body is inlined into
  # `Top`, so there should be no `hw.instance` left.
  # CHECK-LABEL: === After hw-flatten-modules ===
  # CHECK:     hw.module @Top
  # CHECK-NOT: hw.instance
  # CHECK-NOT: hw.module private @Inner
  print("=== After hw-flatten-modules ===")
  print(m)
