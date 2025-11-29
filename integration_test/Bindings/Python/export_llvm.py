# REQUIRES: bindings_python
# RUN: %PYTHON% %s

import circt
from circt.passmanager import PassManager
import io

with circt.ir.Context() as ctx, circt.ir.Location.unknown():
  circt.register_dialects(ctx)

  m = circt.ir.Module.parse("""
    module {
      hw.module @Adder(in %a : i32, in %b : i32, out out : i32) {
        %0 = comb.add %a, %b : i32
        hw.output %0 : i32
      }
    }
    """)

  PassManager.parse("builtin.module(arc-add-taps)").run(m.operation)
  PassManager.parse("builtin.module(arc-lower-state)").run(m.operation)
  PassManager.parse("builtin.module(arc.model(arc-allocate-state))").run(
      m.operation)
  PassManager.parse("builtin.module(lower-arc-to-llvm)").run(m.operation)
  PassManager.parse("builtin.module(cse)").run(m.operation)
  PassManager.parse("builtin.module(arc-canonicalizer)").run(m.operation)

  buffer = io.StringIO()
  circt.export_llvm_ir(m, buffer)
  assert len(buffer.getvalue()) > 0
