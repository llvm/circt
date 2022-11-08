# REQUIRES: bindings_python
# RUN: %PYTHON% %s

import circt
from circt.dialects import comb, esi, hw, seq, sv

from mlir.passmanager import PassManager
from mlir.ir import Context

with Context():
  pm = PassManager.parse("builtin.module(cse)")
