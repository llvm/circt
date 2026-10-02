# RUN: %PYTHON% %s %t | FileCheck %s

import sys

from pycde import System
from pycde.circt.dialects import sv

module_str = """
sv.verbatim "before"
sv.package @Types {
  hw.typedecl @word : i8
} {hw.verilogName = "ExternalTypes", sym_visibility = "private"}
sv.verbatim "after"
"""


def check_position(system, op):
  if isinstance(op, sv.PackageExternOp):
    # Check the replacement before it is moved into the importing system.
    siblings = list(op.operation.parent.regions[0].blocks[0].operations)
    assert siblings[0] == op
    assert len(siblings) == 2
  return None, None, None


system = System([], name="import", output_directory=sys.argv[1])
system.import_mlir(module_str, external_packages=True, importer=check_position)
assert system.mod.operation.verify()

# CHECK:      sv.verbatim "before"
# CHECK-NEXT: sv.package.extern @Types {
# CHECK-NEXT:   hw.typedecl @word : i8
# CHECK-NEXT: } {sym_visibility = "private", verilogName = "ExternalTypes"}
# CHECK-NEXT: sv.verbatim "after"
system.print()
