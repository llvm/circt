# REQUIRES: bindings_python
# RUN: %PYTHON% %s
# RUN: %PYTHON% %s 2> %t | FileCheck %s
# RUN: cat %t | FileCheck --check-prefix=ERR %s

import circt
from circt.dialects import esi, hw

import circt.ir as ir
import circt.passmanager
import sys

# CHECK-LABEL: === appid ===
# ERR-LABEL:   === appid errors ===
print("=== appid ===")
print("=== appid errors ===", file=sys.stderr)
sys.stderr.flush()

with ir.Context() as ctx, ir.Location.unknown():
  circt.register_dialects(ctx)
  unknown = ir.Location.unknown()

  appid1 = esi.AppIDAttr.get("foo", 4)
  # CHECK: appid1: #esi.appid<"foo"[4]>, foo, 4
  print(f"appid1: {appid1}, {appid1.name}, {appid1.index}")

  appid_noidx = esi.AppIDAttr.get("noidx")
  # CHECK: appid_noidx: #esi.appid<"noidx">, noidx, None
  print(f"appid_noidx: {appid_noidx}, {appid_noidx.name}, {appid_noidx.index}")

  appid_path1 = esi.AppIDPathAttr.get(ir.FlatSymbolRefAttr.get("Foo"), [appid1])
  # CHECK: appid_path1: #esi.appid_path<@Foo[<"foo"[4]>]>, @Foo, 1, #esi.appid<"foo"[4]>
  print(f"appid_path1: {appid_path1}, {appid_path1.root}, "
        f"{len(appid_path1)}, {appid_path1[0]}")

  # Valid appid tree.
  mainmod = ir.Module.create()
  with ir.InsertionPoint(mainmod.body):
    extmod = hw.HWModuleExternOp(name='ExternModA',
                                 input_ports=[],
                                 output_ports=[])
    mymod3 = hw.HWModuleOp(name='MyMod', input_ports=[], output_ports=[])
    with ir.InsertionPoint(mymod3.add_entry_block()):
      inst = extmod.instantiate("inst1", sym_name="inst1")
      inst.operation.attributes["esi.appid"] = esi.AppIDAttr.get("bar", 2)
      hw.OutputOp([])

    top = hw.HWModuleOp(name='Top', input_ports=[], output_ports=[])
    with ir.InsertionPoint(top.add_entry_block()):
      mymod3.instantiate("myMod", sym_name="myMod")
      ext_inst = extmod.instantiate("ext_inst1", sym_name="ext_inst1")
      ext_inst.operation.attributes["esi.appid"] = esi.AppIDAttr.get("ext", 0)
      hw.OutputOp([])
  appid_idx = esi.AppIDIndex(mainmod.operation)

  # CHECK:      hw.module.extern @ExternModA()
  # CHECK:      hw.module @MyMod()
  # CHECK:        hw.instance "inst1" sym @inst1 @ExternModA() -> () {esi.appid = #esi.appid<"bar"[2]>}
  # CHECK:      hw.module @Top()
  # CHECK:        hw.instance "myMod" sym @myMod @MyMod() -> ()
  # CHECK:        hw.instance "ext_inst1" sym @ext_inst1 @ExternModA() -> () {esi.appid = #esi.appid<"ext"[0]>}
  print(mainmod)

  appids = appid_idx.get_child_appids_of(top)
  # CHECK: [#esi.appid<"bar"[2]>, #esi.appid<"ext"[0]>]
  print(appids)
  path = appid_idx.get_appid_path(top, esi.AppIDAttr.get("bar", 2),
                                  ir.Location.file("msft.py", 12, 0))
  # CHECK: [#hw.innerNameRef<@Top::@myMod>, #hw.innerNameRef<@MyMod::@inst1>]
  print(path)

  # Invalid appid queries
  # ERR:      error: could not find appid '#esi.appid<"NoExist"[2]>'
  path = appid_idx.get_appid_path(top, esi.AppIDAttr.get("NoExist", 2), unknown)
  assert path is None

  # Invalid appid tree.
  mod2 = ir.Module.create()
  with ir.InsertionPoint(mod2.body):
    extmod = hw.HWModuleExternOp(name='ExternModA',
                                 input_ports=[],
                                 output_ports=[])

    mymod = hw.HWModuleOp(name='MyMod', input_ports=[], output_ports=[])
    with ir.InsertionPoint(mymod.add_entry_block()):
      inst = extmod.instantiate("inst1", sym_name="inst1")
      inst.operation.attributes["esi.appid"] = esi.AppIDAttr.get("bar", 2)
      hw.OutputOp([])

    top = hw.HWModuleOp(name='Top', input_ports=[], output_ports=[])
    with ir.InsertionPoint(top.add_entry_block()):
      mymod.instantiate("myMod1", sym_name="myMod1")
      mymod.instantiate("myMod2", sym_name="myMod2")
      hw.OutputOp([])

  # ERR:        error: 'hw.instance' op Found multiple identical AppIDs in same module
  appid_idx = esi.AppIDIndex(mod2.operation)
