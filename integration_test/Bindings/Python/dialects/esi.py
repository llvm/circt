# REQUIRES: bindings_python
# RUN: %PYTHON% %s

from circt import esi
# from circt.dialects import esi

# from mlir.ir import *
# from mlir.dialects import builtin

esi.attachports()
