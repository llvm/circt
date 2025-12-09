# REQUIRES: bindings_python
# RUN: %PYTHON% %s | %PYTHON% -m filecheck %s

import circt
import io

with circt.ir.Context() as ctx, circt.ir.Location.unknown():
  circt.register_dialects(ctx)

  m = circt.ir.Module.parse("""
        llvm.func @add(%arg0: i64, %arg1: i64) -> i64 { 
            %0 = llvm.add %arg0, %arg1  : i64 
            llvm.return %0 : i64 
        }
    """)

  # CHECK: define i64 @add(i64 %0, i64 %1) {
  # CHECK:   %3 = add i64 %0, %1
  # CHECK:   ret i64 %3
  # CHECK: }
  buffer = io.StringIO()
  circt.export_llvm_ir(m, buffer)
  print(buffer.getvalue())
