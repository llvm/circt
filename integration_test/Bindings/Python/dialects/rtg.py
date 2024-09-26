# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import rtg
from circt.ir import Context, Location, Module, InsertionPoint, Block

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    snp = rtg.SnippetOp(rtg.SnippetType.get(ctx))
    block = Block.create_at_start(snp.bodyRegion, [])
    with InsertionPoint(block):
      rtg.label("lbl", [])
  # CHECK: rtg.snippet
  # CHECK: rtg.label "lbl"
  print(m)
