# REQUIRES: bindings_python
# RUN: %PYTHON% %s 2> %t | FileCheck %s
# RUN: cat %t | FileCheck --check-prefix=ERR %s

import circt
from circt import msft
from circt.dialects import hw, msft as msft_ops

import mlir.ir as ir
import sys

with ir.Context() as ctx, ir.Location.unknown():
  circt.register_dialects(ctx)
  i32 = ir.IntegerType.get_signless(32)
  i1 = ir.IntegerType.get_signless(1)

  m = ir.Module.create()
  with ir.InsertionPoint(m.body):
    extmod = hw.HWModuleExternOp(name='MyExternMod',
                                 input_ports=[],
                                 output_ports=[])

    op = hw.HWModuleOp(name='MyWidget',
                       input_ports=[],
                       output_ports=[],
                       body_builder=lambda module: hw.OutputOp([]))

    top = hw.HWModuleOp(name='top',
                        input_ports=[],
                        output_ports=[],
                        body_builder=lambda module: hw.OutputOp([]))

    msft_mod = msft_ops.MSFTModuleOp(name='msft_mod',
                                     input_ports=[],
                                     output_ports=[],
                                     parameters=ir.DictAttr.get(
                                         {"WIDTH": ir.IntegerAttr.get(i32, 8)}))

  with ir.InsertionPoint.at_block_terminator(op.body.blocks[0]):
    ext_inst = extmod.create("ext1")

  with ir.InsertionPoint.at_block_terminator(top.body.blocks[0]):
    path = op.create("inst1")
    minst = msft_mod.create("minst")

  # CHECK: #msft.physloc<M20K, 2, 6, 1>
  physAttr = msft.PhysLocationAttr.get(msft.M20K, x=2, y=6, num=1)
  print(physAttr)

  path = msft.RootedInstancePathAttr.get(
      ir.Attribute.parse("@top"),
      [ir.StringAttr.get("inst1"),
       ir.StringAttr.get("ext1")])
  print(path)
  # CHECK-NEXT: #msft<"@top[\22inst1\22,\22ext1\22]">
  # CHECK-NEXT: #msft.switch.inst<@top["inst1","ext1"]=#msft.physloc<M20K, 2, 6, 1>>
  instSwitch = msft.SwitchInstanceAttr.get([(path, physAttr)])
  print(instSwitch)

  resolved_inst = msft.get_instance(top.operation,
                                    ir.Attribute.parse("@inst1::@ext1"))
  assert (resolved_inst == ext_inst.operation)
  resolved_inst.attributes["loc:foo_subpath"] = instSwitch

  not_found_inst = msft.get_instance(top.operation,
                                     ir.Attribute.parse("@inst_none::@ext1"))
  assert (not_found_inst is None)

  # CHECK: hw.module @MyWidget()
  # CHECK:   hw.output
  # CHECK: msft.module @msft_mod {WIDTH = 8 : i32} ()
  m.operation.print()

  db = msft.PlacementDB(top.operation)

  assert db.get_instance_at(physAttr) is None
  place_rc = db.add_placement(physAttr, path, "foo_subpath", resolved_inst)
  assert place_rc
  located_inst = db.get_instance_at(physAttr)
  assert located_inst is not None
  assert located_inst[0] == path
  assert located_inst[1] == "foo_subpath"
  assert located_inst[2] == resolved_inst

  num_failed = db.add_design_placements()
  assert num_failed == 1
  # ERR: error: 'hw.instance' op Could not apply placement #msft.physloc<M20K, 2, 6, 1>. Position already occupied by hw.instance "ext1" @MyExternMod

  # CHECK-LABEL: === tcl ===
  print("=== tcl ===")

  # CHECK: proc top_config { parent } {
  # CHECK:   set_location_assignment M20K_X2_Y6_N1 -to $parent|inst1|ext1|ext1|foo_subpath
  msft.export_tcl(top.operation, sys.stdout)

  devdb = msft.PrimitiveDB()
  assert not devdb.is_valid_location(physAttr)
  devdb.add_primitive(physAttr)
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=2, y=50, num=1))
  assert devdb.is_valid_location(physAttr)

  seeded_pdb = msft.PlacementDB(top.operation, devdb)

  print(seeded_pdb.get_nearest_free_in_column(msft.M20K, 2, 4))
  # CHECK: #msft.physloc<M20K, 2, 6, 1>

  rc = seeded_pdb.add_placement(physAttr, path, "foo_subpath", resolved_inst)
  assert rc

  print(seeded_pdb.get_nearest_free_in_column(msft.M20K, 2, 4))

  # CHECK: #msft.physloc<M20K, 2, 50, 1>


  def print_placement(loc, placement):
    if placement:
      path = msft.RootedInstancePathAttr(placement[0])
      print(f"{loc}, {path}")
    else:
      print(f"{loc}")

  print("=== Placements:")
  seeded_pdb.walk_placements(print_placement)
  # CHECK-LABEL: === Placements:
  # CHECK: #msft.physloc<M20K, 2, 6, 1>, #msft<"@top[\22inst1\22,\22ext1\22]">
  # CHECK: #msft.physloc<M20K, 2, 50, 1>

  print("=== Placements (col 2):")
  seeded_pdb.walk_placements(print_placement, bounds=(2, 2, None, None))
  # CHECK-LABEL: === Placements (col 2):
  # CHECK: #msft.physloc<M20K, 2, 6, 1>, #msft<"@top[\22inst1\22,\22ext1\22]">
  # CHECK: #msft.physloc<M20K, 2, 50, 1>

  print("=== Placements (col 2, row > 10):")
  seeded_pdb.walk_placements(print_placement, bounds=(2, 2, 10, None))
  # CHECK-LABEL: === Placements (col 2, row > 10):
  # CHECK: #msft.physloc<M20K, 2, 50, 1>

  print("=== Placements (col 6):")
  seeded_pdb.walk_placements(print_placement, bounds=(6, 6, None, None))
  # CHECK-LABEL: === Placements (col 6):

  print("=== Errors:", file=sys.stderr)
  # TODO: Python's sys.stderr doesn't seem to be shared with C++ errors.
  # See https://github.com/llvm/circt/issues/1983 for more info.
  sys.stderr.flush()
  # ERR-LABEL: === Errors:
  bad_loc = msft.PhysLocationAttr.get(msft.M20K, x=7, y=99, num=1)
  rc = seeded_pdb.add_placement(bad_loc, path, "foo_subpath", resolved_inst)
  assert not rc
  # ERR: error: 'hw.instance' op Could not apply placement. Invalid location
