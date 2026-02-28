# REQUIRES: bindings_python
# RUN: %PYTHON% %s

import circt
from circt.dialects import comb, esi, hw, pipeline, seq, sv

from circt.passmanager import PassManager
from circt.ir import Context

with Context():
  pm = PassManager.parse("builtin.module(cse)")
