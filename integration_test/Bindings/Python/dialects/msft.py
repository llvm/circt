# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

from circt import msft

# CHECK: Successfully imported circt.msft
print("Successfully imported circt.msft")
