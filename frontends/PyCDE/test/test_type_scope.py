# RUN: %PYTHON% %s | FileCheck %s

from pycde import System, Module
from pycde.types import Bits, TypeAlias

TypeAlias(Bits(8), "ScopeWord")


class Dummy(Module):
  pass


# By default, the aliases are declared in an `sv.package` named after the
# system.
# CHECK-LABEL: === package
# CHECK:       sv.package @pycde {
# CHECK-NEXT:    hw.typedecl @ScopeWord : i8
# CHECK-NEXT:  } {hw.verilogName = "DummyTypes"}
# CHECK-NOT:   hw.type_scope
print("=== package")
s = System(Dummy, output_directory="out_package")
TypeAlias.declare_aliases(s.mod, "DummyTypes")
s.print()

# The legacy, include-guarded type scope is used when requested. An existing
# type scope is reused rather than creating a package.
# CHECK-LABEL: === legacy
# CHECK:       sv.verbatim "`ifndef __PYCDE_TYPES__"
# CHECK-NEXT:  sv.verbatim "`define __PYCDE_TYPES__"
# CHECK-NEXT:  hw.type_scope @pycde {
# CHECK-NEXT:    hw.typedecl @ScopeWord : i8
# CHECK-NEXT:  }
# CHECK-NEXT:  sv.verbatim "`endif // __PYCDE_TYPES__"
# CHECK-NOT:   sv.package
print("=== legacy")
s = System(Dummy, output_directory="out_legacy")
TypeAlias.declare_aliases(s.mod, legacy_type_scope=True)
TypeAlias.declare_aliases(s.mod, "DummyTypes")
s.print()

# An imported type scope in the layout emitted by previous versions of PyCDE is
# reused.
# CHECK-LABEL: === imported
# CHECK:       sv.verbatim "`ifndef __PYCDE_TYPES__"
# CHECK-NEXT:  sv.verbatim "`define __PYCDE_TYPES__"
# CHECK-NEXT:  hw.type_scope @pycde {
# CHECK-NEXT:    hw.typedecl @Existing : i4
# CHECK-NEXT:    hw.typedecl @ScopeWord : i8
# CHECK-NEXT:  }
# CHECK-NEXT:  sv.verbatim "`endif // __PYCDE_TYPES__"
# CHECK-NOT:   sv.package
print("=== imported")
s = System(Dummy, output_directory="out_imported")
s.import_mlir("""
sv.verbatim "`ifndef __PYCDE_TYPES__"
sv.verbatim "`define __PYCDE_TYPES__"
hw.type_scope @pycde {
  hw.typedecl @Existing : i4
}
sv.verbatim "`endif // __PYCDE_TYPES__"
""")
TypeAlias.declare_aliases(s.mod, "DummyTypes")
s.print()
