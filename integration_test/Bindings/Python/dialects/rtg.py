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
      cpu0 = rtgtest.CPUDeclOp(0)
      cpu1 = rtgtest.CPUDeclOp(1)
      rtg.YieldOp([cpu0, cpu1])

    test = rtg.TestOp('test_name', TypeAttr.get(dictTy))
    Block.create_at_start(test.bodyRegion, [cpuTy, cpuTy])

  # CHECK: rtg.target @target_name : !rtg.dict<cpu0: !rtgtest.cpu, cpu1: !rtgtest.cpu> {
  # CHECK:   [[V0:%.+]] = rtgtest.cpu_decl 0
  # CHECK:   [[V1:%.+]] = rtgtest.cpu_decl 1
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
    seq = rtg.SequenceOp('seq')
    Block.create_at_start(seq.bodyRegion, [sequenceTy, labelTy, setTy, bagTy])

  # CHECK: rtg.sequence @seq
  # CHECK: (%{{.*}}: !rtg.sequence, %{{.*}}: !rtg.label, %{{.*}}: !rtg.set<index>, %{{.*}}: !rtg.bag<index>):
  print(m)
