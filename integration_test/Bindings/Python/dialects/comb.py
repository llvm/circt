# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.design_entry import connect
from circt.dialects import comb, hw

from mlir.ir import Context, Location, InsertionPoint, IntegerType, IntegerAttr, Module

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i1 = IntegerType.get_signless(1)
  i32 = IntegerType.get_signless(32)

  m = Module.create()
  with InsertionPoint(m.body):

    def build(module):
      # CHECK: %[[CONST:.+]] = hw.constant 1 : i32
      const = hw.ConstantOp(i32, IntegerAttr.get(i32, 1))

      # CHECK: %[[BIT:.+]] = hw.constant true
      bit = hw.ConstantOp(i1, IntegerAttr.get(i1, 1))

      # CHECK: comb.extract %[[CONST]] from 14
      comb.ExtractOp.create(14, i32, input=const.result)
      # CHECK: comb.extract %[[CONST]] from 14
      extract = comb.ExtractOp.create(14, i32)
      connect(extract.input, const.result)

      # CHECK: comb.parity %[[CONST]]
      comb.ParityOp.create(i32, input=const.result)
      # CHECK: comb.parity %[[CONST]]
      parity = comb.ParityOp.create(i32)
      connect(parity.input, const.result)

      # CHECK: comb.sext %[[CONST]]
      comb.SExtOp.create(i32, input=const.result)
      # CHECK: comb.sext %[[CONST]]
      sext = comb.SExtOp.create(i32)
      connect(sext.input, const.result)

      # CHECK: comb.divs %[[CONST]], %[[CONST]]
      comb.DivSOp.create(lhs=const.result, rhs=const.result)
      # CHECK: comb.divs %[[CONST]], %[[CONST]]
      divs = comb.DivSOp.create(i32)
      connect(divs.lhs, const.result)
      connect(divs.rhs, const.result)

      # CHECK: comb.divu %[[CONST]], %[[CONST]]
      comb.DivUOp.create(lhs=const.result, rhs=const.result)
      # CHECK: comb.divu %[[CONST]], %[[CONST]]
      divu = comb.DivUOp.create(i32)
      connect(divu.lhs, const.result)
      connect(divu.rhs, const.result)

      # CHECK: comb.mods %[[CONST]], %[[CONST]]
      comb.ModSOp.create(lhs=const.result, rhs=const.result)
      # CHECK: comb.mods %[[CONST]], %[[CONST]]
      mods = comb.ModSOp.create(i32)
      connect(mods.lhs, const.result)
      connect(mods.rhs, const.result)

      # CHECK: comb.modu %[[CONST]], %[[CONST]]
      comb.ModUOp.create(lhs=const.result, rhs=const.result)
      # CHECK: comb.modu %[[CONST]], %[[CONST]]
      modu = comb.ModUOp.create(i32)
      connect(modu.lhs, const.result)
      connect(modu.rhs, const.result)

      # CHECK: comb.shl %[[CONST]], %[[CONST]]
      comb.ShlOp.create(lhs=const.result, rhs=const.result)
      # CHECK: comb.shl %[[CONST]], %[[CONST]]
      shl = comb.ShlOp.create(i32)
      connect(shl.lhs, const.result)
      connect(shl.rhs, const.result)

      # CHECK: comb.shrs %[[CONST]], %[[CONST]]
      comb.ShrSOp.create(lhs=const.result, rhs=const.result)
      # CHECK: comb.shrs %[[CONST]], %[[CONST]]
      shrs = comb.ShrSOp.create(i32)
      connect(shrs.lhs, const.result)
      connect(shrs.rhs, const.result)

      # CHECK: comb.shru %[[CONST]], %[[CONST]]
      comb.ShrUOp.create(lhs=const.result, rhs=const.result)
      # CHECK: comb.shru %[[CONST]], %[[CONST]]
      shru = comb.ShrUOp.create(i32)
      connect(shru.lhs, const.result)
      connect(shru.rhs, const.result)

      # CHECK: comb.sub %[[CONST]], %[[CONST]]
      comb.SubOp.create(lhs=const.result, rhs=const.result)
      # CHECK: comb.sub %[[CONST]], %[[CONST]]
      sub = comb.SubOp.create(i32)
      connect(sub.lhs, const.result)
      connect(sub.rhs, const.result)

      # CHECK: comb.icmp eq %[[CONST]], %[[CONST]]
      comb.EqOp.create(lhs=const.result, rhs=const.result)
      eq = comb.EqOp.create()
      connect(eq.lhs, const.result)
      connect(eq.rhs, const.result)

      # CHECK: comb.icmp ne %[[CONST]], %[[CONST]]
      comb.NeOp.create(lhs=const.result, rhs=const.result)
      ne = comb.NeOp.create()
      connect(ne.lhs, const.result)
      connect(ne.rhs, const.result)

      # CHECK: comb.icmp slt %[[CONST]], %[[CONST]]
      comb.LtSOp.create(lhs=const.result, rhs=const.result)
      lts = comb.LtSOp.create()
      connect(lts.lhs, const.result)
      connect(lts.rhs, const.result)

      # CHECK: comb.icmp sle %[[CONST]], %[[CONST]]
      comb.LeSOp.create(lhs=const.result, rhs=const.result)
      les = comb.LeSOp.create()
      connect(les.lhs, const.result)
      connect(les.rhs, const.result)

      # CHECK: comb.icmp sgt %[[CONST]], %[[CONST]]
      comb.GtSOp.create(lhs=const.result, rhs=const.result)
      gts = comb.GtSOp.create()
      connect(gts.lhs, const.result)
      connect(gts.rhs, const.result)

      # CHECK: comb.icmp sge %[[CONST]], %[[CONST]]
      comb.GeSOp.create(lhs=const.result, rhs=const.result)
      ges = comb.GeSOp.create()
      connect(ges.lhs, const.result)
      connect(ges.rhs, const.result)

      # CHECK: comb.icmp ult %[[CONST]], %[[CONST]]
      comb.LtUOp.create(lhs=const.result, rhs=const.result)
      ltu = comb.LtUOp.create()
      connect(ltu.lhs, const.result)
      connect(ltu.rhs, const.result)

      # CHECK: comb.icmp ule %[[CONST]], %[[CONST]]
      comb.LeUOp.create(lhs=const.result, rhs=const.result)
      leu = comb.LeUOp.create()
      connect(leu.lhs, const.result)
      connect(leu.rhs, const.result)

      # CHECK: comb.icmp ugt %[[CONST]], %[[CONST]]
      comb.GtUOp.create(lhs=const.result, rhs=const.result)
      gtu = comb.GtUOp.create()
      connect(gtu.lhs, const.result)
      connect(gtu.rhs, const.result)

      # CHECK: comb.icmp uge %[[CONST]], %[[CONST]]
      comb.GeUOp.create(lhs=const.result, rhs=const.result)
      geu = comb.GeUOp.create()
      connect(geu.lhs, const.result)
      connect(geu.rhs, const.result)

      # CHECK: comb.add %[[CONST]], %[[CONST]]
      comb.AddOp.create(const.result, const.result)

      # CHECK: comb.mul %[[CONST]], %[[CONST]]
      comb.MulOp.create(const.result, const.result)

      # CHECK: comb.and %[[CONST]], %[[CONST]]
      comb.AndOp.create(const.result, const.result)

      # CHECK: comb.or %[[CONST]], %[[CONST]]
      comb.OrOp.create(const.result, const.result)

      # CHECK: comb.xor %[[CONST]], %[[CONST]]
      comb.XorOp.create(const.result, const.result)

      # CHECK: comb.merge %[[CONST]], %[[CONST]]
      comb.MergeOp.create(const.result, const.result)

      # CHECK: comb.concat %[[CONST]], %[[CONST]]
      comb.ConcatOp.create(i32, const.result, const.result)

      # CHECK: comb.mux %[[BIT]], %[[CONST]], %[[CONST]]
      comb.MuxOp.create(i32, bit.result, const.result, const.result)

    hw.HWModuleOp(name="test", body_builder=build)

  print(m)
