# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.design_entry import connect
from circt.dialects import comb, hw

from mlir.ir import Context, Location, InsertionPoint, IntegerType, IntegerAttr, Module

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i32 = IntegerType.get_signless(32)

  m = Module.create()
  with InsertionPoint(m.body):

    def build(module):
      # CHECK: %[[CONST:.+]] = hw.constant 1 : i32
      const = hw.ConstantOp(i32, IntegerAttr.get(i32, 1))

      # CHECK: comb.divs %[[CONST]], %[[CONST]]
      comb.DivSOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.divs %[[CONST]], %[[CONST]]
      divs = comb.DivSOp.create(i32)
      connect(divs.lhs, const.result)
      connect(divs.rhs, const.result)

      # CHECK: comb.divu %[[CONST]], %[[CONST]]
      comb.DivUOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.divu %[[CONST]], %[[CONST]]
      divu = comb.DivUOp.create(i32)
      connect(divu.lhs, const.result)
      connect(divu.rhs, const.result)

      # CHECK: comb.mods %[[CONST]], %[[CONST]]
      comb.ModSOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.mods %[[CONST]], %[[CONST]]
      mods = comb.ModSOp.create(i32)
      connect(mods.lhs, const.result)
      connect(mods.rhs, const.result)

      # CHECK: comb.modu %[[CONST]], %[[CONST]]
      comb.ModUOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.modu %[[CONST]], %[[CONST]]
      modu = comb.ModUOp.create(i32)
      connect(modu.lhs, const.result)
      connect(modu.rhs, const.result)

      # CHECK: comb.shl %[[CONST]], %[[CONST]]
      comb.ShlOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.shl %[[CONST]], %[[CONST]]
      shl = comb.ShlOp.create(i32)
      connect(shl.lhs, const.result)
      connect(shl.rhs, const.result)

      # CHECK: comb.shrs %[[CONST]], %[[CONST]]
      comb.ShrSOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.shrs %[[CONST]], %[[CONST]]
      shrs = comb.ShrSOp.create(i32)
      connect(shrs.lhs, const.result)
      connect(shrs.rhs, const.result)

      # CHECK: comb.shru %[[CONST]], %[[CONST]]
      comb.ShrUOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.shru %[[CONST]], %[[CONST]]
      shru = comb.ShrUOp.create(i32)
      connect(shru.lhs, const.result)
      connect(shru.rhs, const.result)

      # CHECK: comb.sub %[[CONST]], %[[CONST]]
      comb.SubOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.sub %[[CONST]], %[[CONST]]
      sub = comb.SubOp.create(i32)
      connect(sub.lhs, const.result)
      connect(sub.rhs, const.result)

      # CHECK: comb.icmp eq %[[CONST]], %[[CONST]]
      comb.EqOp.create(i32, {"lhs": const.result, "rhs": const.result})
      eq = comb.EqOp.create(i32)
      connect(eq.lhs, const.result)
      connect(eq.rhs, const.result)

      # CHECK: comb.icmp ne %[[CONST]], %[[CONST]]
      comb.NeOp.create(i32, {"lhs": const.result, "rhs": const.result})
      ne = comb.NeOp.create(i32)
      connect(ne.lhs, const.result)
      connect(ne.rhs, const.result)

      # CHECK: comb.icmp slt %[[CONST]], %[[CONST]]
      comb.LtSOp.create(i32, {"lhs": const.result, "rhs": const.result})
      lts = comb.LtSOp.create(i32)
      connect(lts.lhs, const.result)
      connect(lts.rhs, const.result)

      # CHECK: comb.icmp sle %[[CONST]], %[[CONST]]
      comb.LeSOp.create(i32, {"lhs": const.result, "rhs": const.result})
      les = comb.LeSOp.create(i32)
      connect(les.lhs, const.result)
      connect(les.rhs, const.result)

      # CHECK: comb.icmp sgt %[[CONST]], %[[CONST]]
      comb.GtSOp.create(i32, {"lhs": const.result, "rhs": const.result})
      gts = comb.GtSOp.create(i32)
      connect(gts.lhs, const.result)
      connect(gts.rhs, const.result)

      # CHECK: comb.icmp sge %[[CONST]], %[[CONST]]
      comb.GeSOp.create(i32, {"lhs": const.result, "rhs": const.result})
      ges = comb.GeSOp.create(i32)
      connect(ges.lhs, const.result)
      connect(ges.rhs, const.result)

      # CHECK: comb.icmp ult %[[CONST]], %[[CONST]]
      comb.LtUOp.create(i32, {"lhs": const.result, "rhs": const.result})
      ltu = comb.LtUOp.create(i32)
      connect(ltu.lhs, const.result)
      connect(ltu.rhs, const.result)

      # CHECK: comb.icmp ule %[[CONST]], %[[CONST]]
      comb.LeUOp.create(i32, {"lhs": const.result, "rhs": const.result})
      leu = comb.LeUOp.create(i32)
      connect(leu.lhs, const.result)
      connect(leu.rhs, const.result)

      # CHECK: comb.icmp ugt %[[CONST]], %[[CONST]]
      comb.GtUOp.create(i32, {"lhs": const.result, "rhs": const.result})
      gtu = comb.GtUOp.create(i32)
      connect(gtu.lhs, const.result)
      connect(gtu.rhs, const.result)

      # CHECK: comb.icmp uge %[[CONST]], %[[CONST]]
      comb.GeUOp.create(i32, {"lhs": const.result, "rhs": const.result})
      geu = comb.GeUOp.create(i32)
      connect(geu.lhs, const.result)
      connect(geu.rhs, const.result)

    hw.HWModuleOp(name="test", body_builder=build)

  print(m)
