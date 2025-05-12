# REQUIRES: bindings_python
# RUN: %PYTHON% %s %T && FileCheck %s --input-file=%T/test0_target.s --check-prefix=TEST0 && FileCheck %s --input-file=%T/test1_target.s --check-prefix=TEST1

import sys
import circt

from circt.dialects import rtg, rtgtest
from circt.ir import Context, Location, Module, InsertionPoint, Block, TypeAttr
from circt.passmanager import PassManager
from circt import rtgtool_support as rtgtool

# Tests the split_file option and that the strings of unsupported instructions
# are passed properly to the emission pass.
with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    test = rtg.TestOp('test0', 'test0', TypeAttr.get(rtg.DictType.get()))
    block = Block.create_at_start(test.bodyRegion, [])
    with InsertionPoint(block):
      rtgtest.rv32i_ebreak()

    test = rtg.TestOp('test1', 'test1', TypeAttr.get(rtg.DictType.get()))
    block = Block.create_at_start(test.bodyRegion, [])
    with InsertionPoint(block):
      rtgtest.rv32i_ecall()

    target = rtg.TargetOp('target', TypeAttr.get(rtg.DictType.get()))
    block = Block.create_at_start(target.bodyRegion, [])
    with InsertionPoint(block):
      rtg.YieldOp([])

  pm = PassManager()
  options = rtgtool.Options(seed=0,
                            output_format=rtgtool.OutputFormat.ASM,
                            split_output=True,
                            unsupported_instructions=['rtgtest.rv32i.ebreak'],
                            output_path=sys.argv[1])
  rtgtool.populate_randomizer_pipeline(pm, options)
  pm.run(m.operation)

  # TEST0: ebreak
  # TEST0: .word 0x

  # TEST1: ecall
  print(m)
