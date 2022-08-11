# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pathlib import Path
from tempfile import NamedTemporaryFile

from pycde.system import System

tmp = NamedTemporaryFile()
tmp.write(b"""
hw.module @add(%a: i1, %b: i1) -> (out: i1) {
  %0 = comb.add %a, %b : i1
  hw.output %0 : i1
}

hw.module @and(%a: i1, %b: i1) -> (out: i1) {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}
""")
tmp.flush()

system = System.import_hw_modules(tmp.name)
system.generate()

# CHECK: hw.module @add(%a: i1, %b: i1) -> (out: i1)
# CHECK:   %0 = comb.add %a, %b : i1
# CHECK:   hw.output %0 : i1

# CHECK: hw.module @and(%a: i1, %b: i1) -> (out: i1)
# CHECK:   %0 = comb.and %a, %b : i1
# CHECK:   hw.output %0 : i1

system.print()
tmp.close()
