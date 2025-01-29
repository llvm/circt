# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import rtg, rtgtest
from circt.ir import Context, Location, Module, InsertionPoint, Block, StringAttr, TypeAttr, IndexType
from circt.passmanager import PassManager
from circt import rtgtool_support as rtgtool

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    cpuTy = rtgtest.CPUType.get()
    dictTy = rtg.DictType.get(ctx, [(StringAttr.get('cpu0'), cpuTy),
                                    (StringAttr.get('cpu1'), cpuTy)])

    target = rtg.TargetOp('target_name', TypeAttr.get(dictTy))
    targetBlock = Block.create_at_start(target.bodyRegion, [])
    with InsertionPoint(targetBlock):
      cpuAttr = rtgtest.CPUAttr.get(0)
      cpu0 = rtgtest.CPUDeclOp(cpuAttr)
      cpu1 = rtgtest.CPUDeclOp(rtgtest.CPUAttr.get(cpuAttr.id + 1))
      rtg.YieldOp([cpu0, cpu1])

    test = rtg.TestOp('test_name', TypeAttr.get(dictTy))
    Block.create_at_start(test.bodyRegion, [cpuTy, cpuTy])

  # CHECK: rtg.target @target_name : !rtg.dict<cpu0: !rtgtest.cpu, cpu1: !rtgtest.cpu> {
  # CHECK:   [[V0:%.+]] = rtgtest.cpu_decl <0>
  # CHECK:   [[V1:%.+]] = rtgtest.cpu_decl <1>
  # CHECK:   rtg.yield [[V0]], [[V1]] : !rtgtest.cpu, !rtgtest.cpu
  # CHECK: }
  # CHECK: rtg.test @test_name : !rtg.dict<cpu0: !rtgtest.cpu, cpu1: !rtgtest.cpu> {
  # CHECK: ^bb{{.*}}(%{{.*}}: !rtgtest.cpu, %{{.*}}: !rtgtest.cpu):
  # CHECK: }
  print(m)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    seq = rtg.SequenceOp('seq')
    setTy = rtg.SetType.get(rtg.SequenceType.get())
    seqBlock = Block.create_at_start(seq.bodyRegion, [setTy])

  # CHECK: rtg.sequence @seq {
  # CHECK: ^bb{{.*}}(%{{.*}}: !rtg.set<!rtg.sequence>):
  # CHECK: }
  print(m)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    seq = rtg.SequenceOp('sequence_name')
    Block.create_at_start(seq.bodyRegion, [])

    test = rtg.TestOp('test_name', TypeAttr.get(rtg.DictType.get()))
    block = Block.create_at_start(test.bodyRegion, [])
    with InsertionPoint(block):
      seq_closure = rtg.SequenceClosureOp('sequence_name', [])

  # CHECK: rtg.test @test_name : !rtg.dict<> {
  # CHECK-NEXT:   rtg.sequence_closure
  # CHECK-NEXT: }
  print(m)

  pm = PassManager()
  options = rtgtool.Options(
      seed=0,
      output_format=rtgtool.OutputFormat.ELABORATED_MLIR,
  )
  rtgtool.populate_randomizer_pipeline(pm, options)
  pm.run(m.operation)

  # CHECK: rtg.test @test_name : !rtg.dict<> {
  # CHECK-NEXT: }
  print(m)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    indexTy = IndexType.get()
    sequenceTy = rtg.SequenceType.get()
    labelTy = rtg.LabelType.get()
    setTy = rtg.SetType.get(indexTy)
    bagTy = rtg.BagType.get(indexTy)
    ireg = rtgtest.IntegerRegisterType.get()
    seq = rtg.SequenceOp('seq')
    Block.create_at_start(seq.bodyRegion,
                          [sequenceTy, labelTy, setTy, bagTy, ireg])

  # CHECK: rtg.sequence @seq
  # CHECK: (%{{.*}}: !rtg.sequence, %{{.*}}: !rtg.label, %{{.*}}: !rtg.set<index>, %{{.*}}: !rtg.bag<index>, %{{.*}}: !rtgtest.ireg):
  print(m)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    # CHECK: rtg.fixed_reg #rtgtest.zero
    rtg.FixedRegisterOp(rtgtest.RegZeroAttr.get())
    # CHECK: rtg.fixed_reg #rtgtest.ra
    rtg.FixedRegisterOp(rtgtest.RegRaAttr.get())
    # CHECK: rtg.fixed_reg #rtgtest.sp
    rtg.FixedRegisterOp(rtgtest.RegSpAttr.get())
    # CHECK: rtg.fixed_reg #rtgtest.gp
    rtg.FixedRegisterOp(rtgtest.RegGpAttr.get())
    # CHECK: rtg.fixed_reg #rtgtest.tp
    rtg.FixedRegisterOp(rtgtest.RegTpAttr.get())
    # CHECK: rtg.fixed_reg #rtgtest.t0
    rtg.FixedRegisterOp(rtgtest.RegT0Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.t1
    rtg.FixedRegisterOp(rtgtest.RegT1Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.t2
    rtg.FixedRegisterOp(rtgtest.RegT2Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s0
    rtg.FixedRegisterOp(rtgtest.RegS0Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s1
    rtg.FixedRegisterOp(rtgtest.RegS1Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.a0
    rtg.FixedRegisterOp(rtgtest.RegA0Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.a1
    rtg.FixedRegisterOp(rtgtest.RegA1Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.a2
    rtg.FixedRegisterOp(rtgtest.RegA2Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.a3
    rtg.FixedRegisterOp(rtgtest.RegA3Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.a4
    rtg.FixedRegisterOp(rtgtest.RegA4Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.a5
    rtg.FixedRegisterOp(rtgtest.RegA5Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.a6
    rtg.FixedRegisterOp(rtgtest.RegA6Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.a7
    rtg.FixedRegisterOp(rtgtest.RegA7Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s2
    rtg.FixedRegisterOp(rtgtest.RegS2Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s3
    rtg.FixedRegisterOp(rtgtest.RegS3Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s4
    rtg.FixedRegisterOp(rtgtest.RegS4Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s5
    rtg.FixedRegisterOp(rtgtest.RegS5Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s6
    rtg.FixedRegisterOp(rtgtest.RegS6Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s7
    rtg.FixedRegisterOp(rtgtest.RegS7Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s8
    rtg.FixedRegisterOp(rtgtest.RegS8Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s9
    rtg.FixedRegisterOp(rtgtest.RegS9Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s10
    rtg.FixedRegisterOp(rtgtest.RegS10Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.s11
    rtg.FixedRegisterOp(rtgtest.RegS11Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.t3
    rtg.FixedRegisterOp(rtgtest.RegT3Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.t4
    rtg.FixedRegisterOp(rtgtest.RegT4Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.t5
    rtg.FixedRegisterOp(rtgtest.RegT5Attr.get())
    # CHECK: rtg.fixed_reg #rtgtest.t6
    rtg.FixedRegisterOp(rtgtest.RegT6Attr.get())

  print(m)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    # CHECK: rtgtest.immediate #rtgtest.imm12<3> : !rtgtest.imm12
    rtgtest.ImmediateOp(rtgtest.Imm12Attr.get(3))
    # CHECK: rtgtest.immediate #rtgtest.imm13<3> : !rtgtest.imm13
    rtgtest.ImmediateOp(rtgtest.Imm13Attr.get(3))
    # CHECK: rtgtest.immediate #rtgtest.imm21<3> : !rtgtest.imm21
    rtgtest.ImmediateOp(rtgtest.Imm21Attr.get(3))
    # CHECK: rtgtest.immediate #rtgtest.imm32<3> : !rtgtest.imm32
    rtgtest.ImmediateOp(rtgtest.Imm32Attr.get(3))

  print(m)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    seq = rtg.SequenceOp('seq')
    block = Block.create_at_start(seq.bodyRegion, [])
    with InsertionPoint(block):
      l = rtg.label_decl("label", [])
      visibility = rtg.LabelVisibilityAttr.get(rtg.GLOBAL)
      rtg.label(visibility, l)
      assert visibility.value == rtg.GLOBAL

  # CHECK: rtg.sequence @seq
  # CHECK: rtg.label_decl "label"
  # CHECK: rtg.label global {{%.+}}
  print(m)
