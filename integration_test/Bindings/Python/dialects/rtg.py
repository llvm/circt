# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import rtg
from circt.ir import Context, Location, Module, InsertionPoint, Block, IntegerType

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    i32Ty = IntegerType.get_signless(32)
    snp = rtg.SequenceOp(rtg.SequenceType.get(ctx, [i32Ty]))
    block = Block.create_at_start(snp.bodyRegion, [i32Ty])
    with InsertionPoint(block):
      lbl = rtg.label_decl(i32Ty, "lbl", [])
      rtg.label(lbl)
  # CHECK: rtg.sequence {
  # CHECK: ^bb{{.*}}(%{{.*}}: i32):
  # CHECK: [[LBL:%.+]] = rtg.label.decl "lbl" -> i32
  # CHECK: rtg.label [[LBL]] : i32
  # CHECK: } -> <i32>
  print(m)
