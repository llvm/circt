// RUN: circt-translate %s --emit-hgldd | FileCheck %s
// RUN: circt-translate %s --emit-split-hgldd --hgldd-output-dir=%T --hgldd-source-prefix=my_source --hgldd-output-prefix=my_verilog
// RUN: cat %T/Foo.dd | FileCheck %s --check-prefix=CHECK-FOO
// RUN: cat %T/Bar.dd | FileCheck %s --check-prefix=CHECK-BAR

#loc1 = loc("InputFoo.scala":4:10)
#loc2 = loc("InputFoo.scala":5:11)
#loc3 = loc("InputFoo.scala":6:12)
#loc4 = loc("InputBar.scala":8:5)
#loc5 = loc("InputBar.scala":14:5)
#loc6 = loc("InputBar.scala":21:10)
#loc7 = loc("InputBar.scala":22:11)
#loc8 = loc("InputBar.scala":23:12)
#loc9 = loc("InputBar.scala":25:15)
#loc10 = loc("Foo.sv":42:10)
#loc11 = loc("Bar.sv":49:10)

// CHECK-FOO:      "file_info": [
// CHECK-FOO-NEXT:   "my_source{{/|\\\\}}InputFoo.scala"
// CHECK-FOO-NEXT:   "my_verilog{{/|\\\\}}Foo.sv"
// CHECK-FOO-NEXT:   "my_source{{/|\\\\}}InputBar.scala"
// CHECK-FOO-NEXT: ]
// CHECK-FOO-NEXT: "hdl_file_index": 2

// CHECK-FOO: "kind": "module"
// CHECK-FOO: "obj_name": "Foo"

// CHECK-BAR:      "file_info": [
// CHECK-BAR-NEXT:   "my_source{{/|\\\\}}InputBar.scala"
// CHECK-BAR-NEXT:   "my_verilog{{/|\\\\}}Bar.sv"
// CHECK-BAR-NEXT: ]
// CHECK-BAR-NEXT: "hdl_file_index": 2

// CHECK-BAR: "kind": "module"
// CHECK-BAR: "obj_name": "Bar"

// CHECK-LABEL: FILE "Foo.dd"
// CHECK: "HGLDD"
// CHECK:   "version": "1.0"
// CHECK:   "file_info": [
// CHECK:     "InputFoo.scala"
// CHECK:     "Foo.sv"
// CHECK:     "InputBar.scala"
// CHECK:   ]
// CHECK:   "hdl_file_index": 2
// CHECK: "obj_name": "Foo"
// CHECK: "module_name": "Foo"
// CHECK:   "hgl_loc"
// CHECK:     "file": 1,
// CHECK:     "begin_line": 4
// CHECK:     "begin_column": 10
// CHECK:   "hdl_loc"
// CHECK:     "file": 2,
// CHECK:     "begin_line": 42
// CHECK:     "begin_column": 10
// CHECK:   "port_vars"
// CHECK:     "var_name": "a"
// CHECK:     "var_name": "b"
// CHECK:   "children"
// CHECK:     "name": "b0"
// CHECK:     "obj_name": "Bar"
// CHECK:     "module_name": "Bar"
// CHECK:     "hgl_loc"
// CHECK:       "file": 3,
// CHECK:     "name": "b1"
// CHECK:     "obj_name": "Bar"
// CHECK:     "module_name": "Bar"
// CHECK:     "hgl_loc"
// CHECK:       "file": 3,
hw.module @Foo(in %a: i32 loc(#loc2), out b: i32 loc(#loc3)) {
  %b0.y = hw.instance "b0" @Bar(x: %a: i32) -> (y: i32) loc(#loc4)
  %b1.y = hw.instance "b1" @Bar(x: %b0.y: i32) -> (y: i32) loc(#loc5)
  hw.output %b1.y : i32 loc(#loc1)
} loc(fused[#loc1, "emitted"(#loc10)])

// CHECK-LABEL: FILE "Bar.dd"
// CHECK: "module_name": "Bar"
// CHECK:   "port_vars"
// CHECK:     "var_name": "x"
// CHECK:     "var_name": "y"
// CHECK:     "var_name": "z"
hw.module private @Bar(in %x: i32 loc(#loc7), out y: i32 loc(#loc8)) {
  %0 = comb.mul %x, %x : i32 loc(#loc9)
  %z = hw.wire %0 : i32 loc(#loc9)
  hw.output %z : i32 loc(#loc6)
} loc(fused[#loc6, "emitted"(#loc11)])

// CHECK-LABEL: "obj_name": "SingleResult"
// CHECK:       "module_name": "CustomSingleResult123"
// CHECK:       "isExtModule": 1
hw.module.extern @SingleResult(out outPort: i1) attributes {verilogName = "CustomSingleResult123"}

// CHECK-LABEL: "module_name": "LegalizedNames"
// CHECK:       "children"
// CHECK:         "name": "reg"
// CHECK:         "hdl_obj_name": "reg_0"
// CHECK:         "obj_name": "Dummy"
// CHECK:         "module_name": "CustomDummy"
hw.module @LegalizedNames() {
  hw.instance "reg" @Dummy() -> () {hw.verilogName = "reg_0"}
}
hw.module.extern @Dummy() attributes {verilogName = "CustomDummy"}
