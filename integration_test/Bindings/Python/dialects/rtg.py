# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import rtg, rtgtest
from circt.ir import Context, Location, Module, InsertionPoint, Block, StringAttr, TypeAttr, IndexType
from circt.passmanager import PassManager

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    cpuTy = rtgtest.CPUType.get()
    dictTy = rtg.DictType.get([(StringAttr.get('cpu0'), cpuTy),
                               (StringAttr.get('cpu1'), cpuTy)], ctx)

    target = rtg.TargetOp('target_name', TypeAttr.get(dictTy))
    targetBlock = Block.create_at_start(target.bodyRegion, [])
    with InsertionPoint(targetBlock):
      cpuAttr = rtgtest.CPUAttr.get(0)
      cpu0 = rtg.ConstantOp(cpuAttr)
      cpu1 = rtg.ConstantOp(rtgtest.CPUAttr.get(cpuAttr.id + 1))
      rtg.YieldOp([cpu0, cpu1])

    test = rtg.TestOp('test_name', 'test_name', TypeAttr.get(dictTy))
    Block.create_at_start(test.bodyRegion, [cpuTy, cpuTy])

  # CHECK: rtg.target @target_name : !rtg.dict<cpu0: !rtgtest.cpu, cpu1: !rtgtest.cpu> {
  # CHECK:   [[V0:%.+]] = rtg.constant #rtgtest.cpu<0>
  # CHECK:   [[V1:%.+]] = rtg.constant #rtgtest.cpu<1>
  # CHECK:   rtg.yield [[V0]], [[V1]] : !rtgtest.cpu, !rtgtest.cpu
  # CHECK: }
  # CHECK: rtg.test @test_name(cpu0 = %cpu0: !rtgtest.cpu, cpu1 = %cpu1: !rtgtest.cpu) {
  # CHECK: }
  print(m)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    setTy = rtg.SetType.get(rtg.SequenceType.get())
    seq = rtg.SequenceOp('seq', TypeAttr.get(rtg.SequenceType.get([setTy])))
    seqBlock = Block.create_at_start(seq.bodyRegion, [setTy])

    # CHECK: !rtg.sequence{{$}}
    print(setTy.element_type)

  # CHECK: rtg.sequence @seq(%{{.*}}: !rtg.set<!rtg.sequence>) {
  # CHECK: }
  print(m)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    seq = rtg.SequenceOp('sequence_name', TypeAttr.get(rtg.SequenceType.get()))
    Block.create_at_start(seq.bodyRegion, [])

    test = rtg.TestOp('test_name', 'test_name',
                      TypeAttr.get(rtg.DictType.get()))
    block = Block.create_at_start(test.bodyRegion, [])
    with InsertionPoint(block):
      seq_get = rtg.GetSequenceOp(rtg.SequenceType.get(), 'sequence_name')
      rtg.RandomizeSequenceOp(seq_get)

    target = rtg.TargetOp('target', TypeAttr.get(rtg.DictType.get()))
    block = Block.create_at_start(target.bodyRegion, [])
    with InsertionPoint(block):
      rtg.YieldOp([])

  # CHECK: rtg.test @test_name() {
  # CHECK-NEXT:   [[SEQ:%.+]] = rtg.get_sequence @sequence_name
  # CHECK-NEXT:   rtg.randomize_sequence [[SEQ]]
  # CHECK-NEXT: }
  print(m)

  pm = PassManager()
  pm.add('rtg-randomization-pipeline{seed=0}')
  pm.run(m.operation)

  # CHECK: rtg.test @test_name_target() template "test_name" target @target {
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
    randomizedSequenceTy = rtg.RandomizedSequenceType.get()
    seq = rtg.SequenceOp(
        'seq',
        TypeAttr.get(
            rtg.SequenceType.get(
                [sequenceTy, labelTy, setTy, bagTy, ireg,
                 randomizedSequenceTy])))
    Block.create_at_start(
        seq.bodyRegion,
        [sequenceTy, labelTy, setTy, bagTy, ireg, randomizedSequenceTy])

    # CHECK: index{{$}}
    print(bagTy.element_type)

  # CHECK: rtg.sequence @seq(%{{.*}}: !rtg.sequence, %{{.*}}: !rtg.isa.label, %{{.*}}: !rtg.set<index>, %{{.*}}: !rtg.bag<index>, %{{.*}}: !rtgtest.ireg, %{{.*}}: !rtg.randomized_sequence)
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
    seq = rtg.SequenceOp('seq', TypeAttr.get(rtg.SequenceType.get([])))
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

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  attr = rtg.DefaultContextAttr.get(rtgtest.CPUType.get())
  # CHECK: !rtgtest.cpu
  print(attr.type)
  # CHECK: #rtg.default : !rtgtest.cpu
  print(attr)

  attr = rtg.AnyContextAttr.get(rtgtest.CPUType.get())
  # CHECK: !rtgtest.cpu
  print(attr.type)
  # CHECK: #rtg.any_context : !rtgtest.cpu
  print(attr)

  immediate_type = rtg.ImmediateType.get(32)
  # CHECK: width=32
  print(f"width={immediate_type.width}")
  # CHECK: !rtg.isa.immediate<32>
  print(immediate_type)

  immediate_attr = rtg.ImmediateAttr.get(32, 42)
  # CHECK: width=32
  print(f"width={immediate_attr.width}")
  # CHECK: value=42
  print(f"value={immediate_attr.value}")
  # CHECK: #rtg.isa.immediate<32, 42>
  print(immediate_attr)

  memory_block_type = rtg.MemoryBlockType.get(32)
  # CHECK: width=32
  print(f"width={memory_block_type.address_width}")
  # CHECK: !rtg.isa.memory_block<32>
  print(memory_block_type)

  memoryTy = rtg.MemoryType.get(32)
  # CHECK: address_width=32
  print(f'address_width={memoryTy.address_width}')
  # CHECK: !rtg.isa.memory<32>
  print(memoryTy)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  indexTy = IndexType.get()
  arr = rtg.ArrayType.get(indexTy)
  # CHECK: element_type=index
  print(f"element_type={arr.element_type}")
  # CHECK: !rtg.array<index>
  print(arr)

  tup = rtg.TupleType.get([indexTy, indexTy])
  # CHECK: fields=[IndexType(index), IndexType(index)]
  print(f"fields={tup.fields}")

  # CHECK: !rtg.tuple<index, index>
  print(tup)

  tup = rtg.TupleType.get([])
  # CHECK: fields=[]
  print(f"fields={tup.fields}")
  # CHECK: !rtg.tuple
  print(tup)
