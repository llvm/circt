# RUN: %PYTHON% %s %t.default default
# RUN: %PYTHON% %s %t.external external
# RUN: %PYTHON% %s %t.preprocess preprocess
# RUN: %PYTHON% %s %t.marked marked

from pathlib import Path
import sys

from pycde import Module, System, generator
from pycde.circt import ir
from pycde.circt.dialects import hw, sv
from pycde.module import ModuleBuilder, import_hw_module
from pycde.types import Bits, TypeAlias

mode = sys.argv[2]
external = mode != "default"
module_str = """
module {
  sv.package @ExtTypes {
    hw.typedecl @Req, "Req" : !hw.struct<addr: i32, data: i8>
  }
  hw.module.extern @ExtMod(in %req : !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>)
}
"""


def importer(system, op):
  if isinstance(op, hw.HWModuleExternOp):
    cls = import_hw_module(system, op, ModuleBuilder)
    return "ExtMod", cls, cls._builder
  return None, None, None


def preprocess_op(op):
  if isinstance(op, sv.PackageOp):
    op.attributes["extern"] = ir.UnitAttr.get()
  return op


class Top(Module):

  @generator
  def construct(ports):
    ExtMod = imported["ExtMod"]
    ExtMod(req=Bits(40)(0).bitcast(ExtMod.req.type))


TypeAlias(Bits(8), "LocalAlias")
system = System([Top], name="repro", output_directory=sys.argv[1])
if mode == "marked":
  module_str = module_str.replace("  }\n", "  } {extern}\n")

# Exercise both string and file import, as well as output_filename overrides.
options = {"importer": importer}
if mode == "external":
  input_file = Path(sys.argv[1]) / "imported.mlir"
  input_file.write_text(module_str)
  options.update(file=input_file,
                 external_packages=True,
                 output_filename="imported.sv")
else:
  options["module_str"] = module_str
  if mode == "preprocess":
    options["preprocess_op"] = preprocess_op
imported = system.import_mlir(**options)

# The flag applies only to this import, not to other or PyCDE-created packages.
system.import_mlir("""
sv.package @LocalTypes {
  hw.typedecl @word : i8
}
""")

package = next(
    op for op in system.body
    if isinstance(op, sv.PackageOp) and op.sym_name.value == "ExtTypes")
assert ("extern" in package.attributes) == external
assert len(package.body.blocks[0].operations) == 1
system.compile()
assert system.mod.operation.verify()

hw_dir = system.hw_output_dir
top_sv = (hw_dir / "Top.sv").read_text()
assert "ExtTypes::Req" in top_sv
assert "ExtMod" in top_sv
assert not (hw_dir / "ExtMod.sv").exists()
assert (hw_dir / "LocalTypes.sv").exists()
assert (hw_dir / "reproTypes.sv").exists()
assert not (hw_dir / "imported.sv").exists()

filelist = (hw_dir / "filelist.f").read_text()
assert ("ExtTypes.sv" in filelist) == (not external)
assert (hw_dir / "ExtTypes.sv").exists() == (not external)
verilog = "\n".join(path.read_text() for path in hw_dir.glob("*.sv"))
assert ("package ExtTypes;" in verilog) == (not external)
