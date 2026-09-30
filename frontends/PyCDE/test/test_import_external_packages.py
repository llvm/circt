# RUN: %PYTHON% %s %t.default default
# RUN: %PYTHON% %s %t.external external
# RUN: %PYTHON% %s %t.external-string external-string
# RUN: %PYTHON% %s %t.preprocess preprocess
# RUN: %PYTHON% %s %t.existing existing
# RUN: %PYTHON% %s %t.existing-external existing-external

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
    hw.typedecl @ReqAlias : !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>
  } {test.marker = "preserved"} loc("provider.mlir":3:5)
  hw.module.extern @ExtMod(in %req : !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>)
}
"""


def importer(system, op):
  if isinstance(op, (sv.PackageOp, sv.PackageExternOp)):
    assert isinstance(op, sv.PackageExternOp) == external
  if isinstance(op, hw.HWModuleExternOp):
    cls = import_hw_module(system, op, ModuleBuilder)
    return "ExtMod", cls, cls._builder
  return None, None, None


def preprocess_op(op):
  if isinstance(op, sv.PackageOp):
    external_op = sv.PackageExternOp(op.sym_name, loc=op.location, ip=False)
    for attr_name in op.attributes:
      external_op.attributes[attr_name] = op.attributes[attr_name]
    op.body.blocks[0].append_to(external_op.body)
    op.erase()
    return external_op
  return op


class Top(Module):

  @generator
  def construct(ports):
    ExtMod = imported["ExtMod"]
    ExtMod(req=Bits(40)(0).bitcast(ExtMod.req.type))


TypeAlias(Bits(8), "LocalAlias")
system = System([Top], name="repro", output_directory=sys.argv[1])
if mode in ("existing", "existing-external"):
  module_str = module_str.replace("sv.package ", "sv.package.extern ")

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
  if mode in ("external-string", "existing-external"):
    options["external_packages"] = True
imported = system.import_mlir(**options)

package = imported["ExtTypes"].op
assert isinstance(package, sv.PackageExternOp if external else sv.PackageOp)
assert "extern" not in package.attributes
assert ir.StringAttr(package.attributes["test.marker"]).value == "preserved"
assert package.location == ir.Location.file("provider.mlir", 3, 5)
assert len(package.body.blocks[0].operations) == 2
assert package.body.blocks[0].operations[1].sym_name.value == "ReqAlias"

# Conversion preserves non-public visibility, explicit names, empty bodies, and
# declaration metadata. Consecutive packages exercise replacement during import.
metadata_import = system.import_mlir("""
sv.package @PrivateTypes {
  hw.typedecl @word : i8 {test.marker = "word"} loc("provider.mlir":8:3)
} {sym_visibility = "private"}
sv.package @NamedTypes {
  hw.typedecl @word : !hw.typealias<@PrivateTypes::@word, i8>
} {hw.verilogName = "ProviderTypes", sym_visibility = "private"}
sv.package @EmptyTypes {}
""",
                                     external_packages=True)
private_package = metadata_import["PrivateTypes"].op
named_package = metadata_import["NamedTypes"].op
empty_package = metadata_import["EmptyTypes"].op
for op in (private_package, named_package, empty_package):
  assert isinstance(op, sv.PackageExternOp)
for op, name in ((private_package, "PrivateTypes"), (named_package,
                                                     "ProviderTypes")):
  assert ir.SymbolTable.get_visibility(op).value == "private"
  assert ir.StringAttr(op.attributes["hw.verilogName"]).value == name
word = private_package.body.blocks[0].operations[0]
assert ir.StringAttr(word.attributes["test.marker"]).value == "word"
assert word.location == ir.Location.file("provider.mlir", 8, 3)
assert len(empty_package.body.blocks[0].operations) == 0

# The flag applies only to this import, not to other or PyCDE-created packages.
system.import_mlir("""
sv.package @LocalTypes {
  hw.typedecl @word : i8
}
""")

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
for name in ("PrivateTypes", "NamedTypes", "ProviderTypes", "EmptyTypes"):
  assert not (hw_dir / f"{name}.sv").exists()

filelist = (hw_dir / "filelist.f").read_text()
assert ("ExtTypes.sv" in filelist) == (not external)
assert (hw_dir / "ExtTypes.sv").exists() == (not external)
verilog = "\n".join(path.read_text() for path in hw_dir.glob("*.sv"))
assert ("package ExtTypes;" in verilog) == (not external)
