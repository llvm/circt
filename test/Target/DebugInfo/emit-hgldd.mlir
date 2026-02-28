// RUN: circt-translate %s --emit-hgldd | FileCheck %s
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-translate %s --emit-split-hgldd --hgldd-output-dir=%t.dir --hgldd-source-prefix=my_source --hgldd-output-prefix=my_verilog
// RUN: cat %t.dir/Foo.dd | FileCheck %s --check-prefix=CHECK-FOO
// RUN: cat %t.dir/Bar.dd | FileCheck %s --check-prefix=CHECK-BAR

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
// CHECK: {
// CHECK-NEXT: "HGLDD"
// CHECK-NEXT:   "version": "1.0"
// CHECK-NEXT:   "file_info": [
// CHECK-NEXT:     "InputFoo.scala"
// CHECK-NEXT:     "Foo.sv"
// CHECK-NEXT:     "InputBar.scala"
// CHECK-NEXT:   ]
// CHECK-NEXT:   "hdl_file_index": 2
// CHECK-NEXT: }
// CHECK-NEXT: "objects"
// CHECK-NEXT: {
// CHECK-NEXT:   "kind": "module"
// CHECK-NEXT:   "obj_name": "Foo"
// CHECK-NEXT:   "module_name": "Foo"
// CHECK-NEXT:   "hgl_loc"
// CHECK-NEXT:     "begin_column": 10
// CHECK-NEXT:     "begin_line": 4
// CHECK-NEXT:     "end_column": 10
// CHECK-NEXT:     "end_line": 4
// CHECK-NEXT:     "file": 1
// CHECK-NEXT:   }
// CHECK-NEXT:   "hdl_loc"
// CHECK-NEXT:     "begin_column": 10
// CHECK-NEXT:     "begin_line": 42
// CHECK-NEXT:     "end_column": 10
// CHECK-NEXT:     "end_line": 42
// CHECK-NEXT:     "file": 2
// CHECK-NEXT:   }
// CHECK-NEXT:   "port_vars"
// CHECK:          "var_name": "inA"
// CHECK:          "var_name": "outB"
// CHECK:        "children"
// CHECK-LABEL:    "name": "b0"
// CHECK:          "obj_name": "Bar"
// CHECK:          "module_name": "Bar"
// CHECK:          "hgl_loc"
// CHECK:            "file": 3
// CHECK-LABEL:    "name": "b1"
// CHECK:          "obj_name": "Bar"
// CHECK:          "module_name": "Bar"
// CHECK:          "hgl_loc"
// CHECK:            "file": 3
hw.module @Foo(in %a: i32 loc(#loc2), out b: i32 loc(#loc3)) {
  dbg.variable "inA", %a : i32 loc(#loc2)
  dbg.variable "outB", %b1.y : i32 loc(#loc3)
  %c42_i8 = hw.constant 42 : i8
  dbg.variable "var1", %c42_i8 : i8 loc(#loc3)
  %b0.y = hw.instance "b0" @Bar(x: %a: i32) -> (y: i32) loc(#loc4)
  %b1.y = hw.instance "b1" @Bar(x: %b0.y: i32) -> (y: i32) loc(#loc5)
  hw.output %b1.y : i32 loc(#loc1)
} loc(fused[#loc1, "emitted"(#loc10)])

// CHECK-LABEL: FILE "Bar.dd"
// CHECK: "module_name": "Bar"
// CHECK:   "port_vars"
// CHECK:     "var_name": "inX"
// CHECK:     "var_name": "outY"
// CHECK:     "var_name": "varZ"
hw.module private @Bar(in %x: i32 loc(#loc7), out y: i32 loc(#loc8)) {
  %0 = comb.mul %x, %x : i32 loc(#loc9)
  dbg.variable "inX", %x : i32 loc(#loc7)
  dbg.variable "outY", %0 : i32 loc(#loc8)
  dbg.variable "varZ", %0 : i32 loc(#loc9)
  %1 = comb.add %0, %x : i32 loc(#loc9)
  dbg.variable "add", %1 : i32 loc(#loc9)
  hw.output %0 : i32 loc(#loc6)
} loc(fused[#loc6, "emitted"(#loc11)])

// CHECK-LABEL: FILE "global.dd"
// CHECK-LABEL: "obj_name": "Aggregates_data"
// CHECK:         "var_name": "a"
// CHECK:         "var_name": "b"
// CHECK:         "var_name": "c"
// CHECK-LABEL: "obj_name": "Aggregates"
// CHECK:       "module_name": "Aggregates"
// CHECK:         "var_name": "data"
hw.module @Aggregates(in %data_a: i32, in %data_b: i42, in %data_c_0: i17, in %data_c_1: i17) {
  %0 = dbg.array [%data_c_0, %data_c_1] : i17
  %1 = dbg.struct {"a": %data_a, "b": %data_b, "c": %0} : i32, i42, !dbg.array
  dbg.variable "data", %1 : !dbg.struct
}

// CHECK-LABEL: "obj_name": "EmptyAggregates"
// CHECK:       "module_name": "EmptyAggregates"
// CHECK:         "var_name": "x"
// CHECK:         "value": {"integer_num":0}
// CHECK:         "type_name": "bit"
// CHECK:         "var_name": "y"
// CHECK:         "value": {"integer_num":0}
// CHECK:         "type_name": "bit"
// CHECK:         "var_name": "z"
// CHECK:         "value": {"opcode":"'{","operands":[{"integer_num":0},{"integer_num":0}]}
// CHECK:         "type_name": "EmptyAggregates_z"
hw.module @EmptyAggregates() {
  %0 = dbg.array []
  %1 = dbg.struct {}
  %2 = dbg.struct {"a": %0, "b": %1} : !dbg.array, !dbg.struct
  dbg.variable "x", %0 : !dbg.array
  dbg.variable "y", %1 : !dbg.struct
  dbg.variable "z", %2 : !dbg.struct
}

// CHECK-LABEL: "obj_name": "SingleElementAggregates"
// CHECK:       "module_name": "SingleElementAggregates"
// CHECK:         "var_name": "varFoo"
// CHECK:         "value": {"opcode":"'{","operands":[{"sig_name":"foo"}]}
// CHECK:         "type_name": "logic"
// CHECK:         "unpacked_range": [
// CHECK-NEXT:      0
// CHECK-NEXT:      0
// CHECK-NEXT:    ]
// CHECK:         "var_name": "varBar"
// CHECK:         "value": {"opcode":"'{","operands":[{"sig_name":"bar"}]}
// CHECK:         "type_name": "SingleElementAggregates_varBar"
hw.module @SingleElementAggregates() {
  %foo = sv.wire : !hw.inout<i1>
  %bar = sv.wire : !hw.inout<i1>
  %0 = sv.read_inout %foo : !hw.inout<i1>
  %1 = sv.read_inout %bar : !hw.inout<i1>
  %2 = dbg.array [%0] : i1
  %3 = dbg.struct {"x": %1} : i1
  dbg.variable "varFoo", %2 : !dbg.array
  dbg.variable "varBar", %3 : !dbg.struct
}

// CHECK-LABEL: "obj_name": "MultiDimensionalArrays"
// CHECK:       "module_name": "MultiDimensionalArrays"
// CHECK:         "var_name": "array"
// CHECK:         "value": {"opcode":"'{","operands":[{"opcode":"'{","operands":[{"sig_name":"a"},{"sig_name":"b"},{"sig_name":"c"},{"sig_name":"d"}]}]}
// CHECK:         "type_name": "logic"
// CHECK:         "unpacked_range": [
// CHECK-NEXT:      0
// CHECK-NEXT:      0
// CHECK-NEXT:      3
// CHECK-NEXT:      0
// CHECK-NEXT:    ]
hw.module @MultiDimensionalArrays(in %a: i42, in %b: i42, in %c: i42, in %d: i42) {
  %0 = dbg.array [%a, %b, %c, %d] : i42
  %1 = dbg.array [%0] : !dbg.array
  dbg.variable "array", %1 : !dbg.array
}

// CHECK-LABEL: "module_name": "Expressions"
hw.module @Expressions(in %a: i1, in %b: i1) {
  // CHECK-LABEL: "var_name": "blockArg"
  // CHECK: "value": {"sig_name":"a"}
  // CHECK: "type_name": "logic"
  dbg.variable "blockArg", %a : i1

  %0 = hw.wire %a {hw.verilogName = "explicitName"} : i1

  // CHECK-LABEL: "var_name": "constA"
  // CHECK: "value": {"bit_vector":"00000010100111001"}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 16
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]

  // CHECK-LABEL: "var_name": "constB"
  // CHECK: "value": {"bit_vector":"000000000000000000000000000010001100101001"}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 41
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]

  // CHECK-LABEL: "var_name": "constC"
  // CHECK: "value": {"bit_vector":"0000"}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 3
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]

  // CHECK-LABEL: "var_name": "constD"
  // CHECK: "value": {"bit_vector":"0"}
  // CHECK: "type_name": "logic"

  %k0 = hw.constant 1337 : i17
  %k1 = hw.constant 9001 : i42
  %k2 = hw.constant 0 : i4
  %k3 = hw.constant 0 : i0
  dbg.variable "constA", %k0 : i17
  dbg.variable "constB", %k1 : i42
  dbg.variable "constC", %k2 : i4
  dbg.variable "constD", %k3 : i0

  // CHECK-LABEL: "var_name": "readWire"
  // CHECK: "value": {"sig_name":"svWire"}
  // CHECK: "type_name": "logic"
  %svWire = sv.wire : !hw.inout<i1>
  %3 = sv.read_inout %svWire : !hw.inout<i1>
  dbg.variable "readWire", %3 : i1

  // CHECK-LABEL: "var_name": "readReg"
  // CHECK: "value": {"sig_name":"svReg"}
  // CHECK: "type_name": "logic"
  %svReg = sv.reg : !hw.inout<i1>
  %4 = sv.read_inout %svReg : !hw.inout<i1>
  dbg.variable "readReg", %4 : i1

  // CHECK-LABEL: "var_name": "readLogic"
  // CHECK: "value": {"sig_name":"svLogic"}
  // CHECK: "type_name": "logic"
  %svLogic = sv.logic : !hw.inout<i1>
  %5 = sv.read_inout %svLogic : !hw.inout<i1>
  dbg.variable "readLogic", %5 : i1

  // CHECK-LABEL: "var_name": "myWire"
  // CHECK: "value": {"sig_name":"hwWire"}
  // CHECK: "type_name": "logic"
  %hwWire = hw.wire %a : i1
  dbg.variable "myWire", %hwWire : i1

  // CHECK-LABEL: "var_name": "unaryParity"
  // CHECK: "value": {"opcode":"^","operands":[{"sig_name":"a"}]}
  // CHECK: "type_name": "logic"
  %6 = comb.parity %a : i1
  dbg.variable "unaryParity", %6 : i1

  // CHECK-LABEL: "var_name": "binaryAdd"
  // CHECK: "value": {"opcode":"+","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %7 = comb.add %a, %b : i1
  dbg.variable "binaryAdd", %7 : i1

  // CHECK-LABEL: "var_name": "binarySub"
  // CHECK: "value": {"opcode":"-","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %8 = comb.sub %a, %b : i1
  dbg.variable "binarySub", %8 : i1

  // CHECK-LABEL: "var_name": "binaryMul"
  // CHECK: "value": {"opcode":"*","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %9 = comb.mul %a, %b : i1
  dbg.variable "binaryMul", %9 : i1

  // CHECK-LABEL: "var_name": "binaryDiv1"
  // CHECK: "value": {"opcode":"/","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "binaryDiv2"
  // CHECK: "value": {"opcode":"/","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %10 = comb.divu %a, %b : i1
  %11 = comb.divs %a, %b : i1
  dbg.variable "binaryDiv1", %10 : i1
  dbg.variable "binaryDiv2", %11 : i1

  // CHECK-LABEL: "var_name": "binaryMod1"
  // CHECK: "value": {"opcode":"%","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "binaryMod2"
  // CHECK: "value": {"opcode":"%","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %12 = comb.modu %a, %b : i1
  %13 = comb.mods %a, %b : i1
  dbg.variable "binaryMod1", %12 : i1
  dbg.variable "binaryMod2", %13 : i1

  // CHECK-LABEL: "var_name": "binaryShl"
  // CHECK: "value": {"opcode":"<<","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "binaryShr1"
  // CHECK: "value": {"opcode":">>","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "binaryShr2"
  // CHECK: "value": {"opcode":">>>","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %14 = comb.shl %a, %b : i1
  %15 = comb.shru %a, %b : i1
  %16 = comb.shrs %a, %b : i1
  dbg.variable "binaryShl", %14 : i1
  dbg.variable "binaryShr1", %15 : i1
  dbg.variable "binaryShr2", %16 : i1

  // CHECK-LABEL: "var_name": "cmpEq"
  // CHECK: "value": {"opcode":"==","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpNe"
  // CHECK: "value": {"opcode":"!=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpCeq"
  // CHECK: "value": {"opcode":"===","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpCne"
  // CHECK: "value": {"opcode":"!==","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpWeq"
  // CHECK: "value": {"opcode":"==?","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpWne"
  // CHECK: "value": {"opcode":"!=?","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpUlt"
  // CHECK: "value": {"opcode":"<","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpSlt"
  // CHECK: "value": {"opcode":"<","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpUgt"
  // CHECK: "value": {"opcode":">","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpSgt"
  // CHECK: "value": {"opcode":">","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpUle"
  // CHECK: "value": {"opcode":"<=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpSle"
  // CHECK: "value": {"opcode":"<=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpUge"
  // CHECK: "value": {"opcode":">=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  // CHECK-LABEL: "var_name": "cmpSge"
  // CHECK: "value": {"opcode":">=","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %17 = comb.icmp eq %a, %b : i1
  %18 = comb.icmp ne %a, %b : i1
  %19 = comb.icmp ceq %a, %b : i1
  %20 = comb.icmp cne %a, %b : i1
  %21 = comb.icmp weq %a, %b : i1
  %22 = comb.icmp wne %a, %b : i1
  %23 = comb.icmp ult %a, %b : i1
  %24 = comb.icmp slt %a, %b : i1
  %25 = comb.icmp ugt %a, %b : i1
  %26 = comb.icmp sgt %a, %b : i1
  %27 = comb.icmp ule %a, %b : i1
  %28 = comb.icmp sle %a, %b : i1
  %29 = comb.icmp uge %a, %b : i1
  %30 = comb.icmp sge %a, %b : i1
  dbg.variable "cmpEq", %17 : i1
  dbg.variable "cmpNe", %18 : i1
  dbg.variable "cmpCeq", %19 : i1
  dbg.variable "cmpCne", %20 : i1
  dbg.variable "cmpWeq", %21 : i1
  dbg.variable "cmpWne", %22 : i1
  dbg.variable "cmpUlt", %23 : i1
  dbg.variable "cmpSlt", %24 : i1
  dbg.variable "cmpUgt", %25 : i1
  dbg.variable "cmpSgt", %26 : i1
  dbg.variable "cmpUle", %27 : i1
  dbg.variable "cmpSle", %28 : i1
  dbg.variable "cmpUge", %29 : i1
  dbg.variable "cmpSge", %30 : i1

  // CHECK-LABEL: "var_name": "opAnd"
  // CHECK: "value": {"opcode":"&","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %31 = comb.and %a, %b : i1
  dbg.variable "opAnd", %31 : i1

  // CHECK-LABEL: "var_name": "opOr"
  // CHECK: "value": {"opcode":"|","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %32 = comb.or %a, %b : i1
  dbg.variable "opOr", %32 : i1

  // CHECK-LABEL: "var_name": "opXor"
  // CHECK: "value": {"opcode":"^","operands":[{"sig_name":"a"},{"sig_name":"b"}]}
  // CHECK: "type_name": "logic"
  %33 = comb.xor %a, %b : i1
  dbg.variable "opXor", %33 : i1

  // CHECK-LABEL: "var_name": "concat"
  // CHECK: "value": {"opcode":"{}","operands":[{"sig_name":"a"},{"sig_name":"b"},{"sig_name":"explicitName"}]}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 2
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]
  %34 = comb.concat %a, %b, %0 : i1, i1, i1
  dbg.variable "concat", %34 : i3

  // CHECK-LABEL: "var_name": "replicate"
  // CHECK: "value": {"opcode":"R{}","operands":[{"integer_num":3},{"sig_name":"a"}]}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 2
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]
  %35 = comb.replicate %a : (i1) -> i3
  dbg.variable "replicate", %35 : i3

  // CHECK-LABEL: "var_name": "extract"
  // CHECK: "value": {"opcode":"[]","operands":[{"sig_name":"wideWire"},{"integer_num":19},{"integer_num":12}]}
  // CHECK: "type_name": "logic"
  // CHECK: "packed_range": [
  // CHECK-NEXT: 7
  // CHECK-NEXT: 0
  // CHECK-NEXT: ]
  %wideWire = hw.wire %k1 : i42
  %36 = comb.extract %wideWire from 12 : (i42) -> i8
  dbg.variable "extract", %36 : i8

  // CHECK-LABEL: "var_name": "mux"
  // CHECK: "value": {"opcode":"?:","operands":[{"sig_name":"a"},{"sig_name":"b"},{"sig_name":"explicitName"}]}
  // CHECK: "type_name": "logic"
  %37 = comb.mux %a, %b, %0 : i1
  dbg.variable "mux", %37 : i1
}

// CHECK-LABEL: "obj_name": "SingleResult"
// CHECK:       "module_name": "CustomSingleResult123"
// CHECK:       "isExtModule": 1
hw.module.extern @SingleResult(out outPort: i1) attributes {verilogName = "CustomSingleResult123"}

// CHECK-LABEL: "module_name": "LegalizedNames"
// CHECK:       "port_vars"
// CHECK:          "var_name": "myWire"
// CHECK:          "value": {"sig_name":"wire_1"}
// CHECK:       "children"
// CHECK:         "name": "myInst"
// CHECK:         "hdl_obj_name": "reg_0"
// CHECK:         "obj_name": "Dummy"
// CHECK:         "module_name": "CustomDummy"
hw.module @LegalizedNames() {
  hw.instance "myInst" @Dummy() -> () {hw.verilogName = "reg_0"}
  %false = hw.constant false
  %myWire = hw.wire %false {hw.verilogName = "wire_1"} : i1
}
hw.module.extern @Dummy() attributes {verilogName = "CustomDummy"}

// CHECK-LABEL: "obj_name": "InlineScopes"
// CHECK:       "port_vars"
// CHECK:         "var_name": "x"
// CHECK:         "value": {"sig_name":"a"}
// CHECK:       "children"
// CHECK:         "name": "child"
// CHECK:         "port_vars"
// CHECK:           "var_name": "y"
// CHECK:           "value": {"sig_name":"a"}
// CHECK:         "children"
// CHECK:           "name": "more"
// CHECK:           "port_vars"
// CHECK:             "var_name": "z"
// CHECK:             "value": {"sig_name":"a"}
hw.module @InlineScopes(in %a: i42) {
  %1 = dbg.scope "child", "InlinedChild"
  %2 = dbg.scope "more", "InlinedMore" scope %1
  dbg.variable "x", %a : i42
  dbg.variable "y", %a scope %1 : i42
  dbg.variable "z", %a scope %2 : i42
}

// See https://github.com/llvm/circt/issues/6735
// CHECK-LABEL: "obj_name": "Issue6735_Case1"
hw.module @Issue6735_Case1(out someOutput: i1) {
  // Don't use instance name as signal name directly.
  // CHECK: "var_name": "varA"
  // CHECK-NOT: "sig_name":"instA"
  // CHECK-NOT: "sig_name":"instAVerilog"
  // CHECK: "sig_name":"wireA"
  %0 = hw.instance "instA" @SingleResult() -> (outPort: i1) {hw.verilogName = "instAVerilog"}
  dbg.variable "varA", %0 : i1
  %wireA = hw.wire %0 : i1

  // Use SV declarations to refer to instance output.
  // CHECK: "var_name": "varB"
  // CHECK-NOT: "sig_name":"instB"
  // CHECK-NOT: "field":"outPort"
  // CHECK: "sig_name":"wireB"
  %b = hw.instance "instB" @SingleResult() -> (outPort: i1)
  dbg.variable "varB", %b : i1
  %wireB = sv.wire : !hw.inout<i1>
  sv.assign %wireB, %b : i1
  // CHECK: "var_name": "varC"
  // CHECK-NOT: "sig_name":"instC"
  // CHECK-NOT: "field":"outPort"
  // CHECK: "sig_name":"wireC"
  %c = hw.instance "instC" @SingleResult() -> (outPort: i1)
  dbg.variable "varC", %c : i1
  %wireC = sv.logic : !hw.inout<i1>
  sv.assign %wireC, %c : i1
  // CHECK: "var_name": "varD"
  // CHECK-NOT: "sig_name":"instD"
  // CHECK-NOT: "field":"outPort"
  // CHECK: "sig_name":"wireD"
  %d = hw.instance "instD" @SingleResult() -> (outPort: i1)
  dbg.variable "varD", %d : i1
  %wireD = sv.logic : !hw.inout<i1>
  sv.assign %wireD, %d : i1

  // Use module's output port name to refer to instance output.
  // CHECK: "var_name": "varZ"
  // CHECK-NOT: "sig_name":"instZ"
  // CHECK-NOT: "field":"outPort"
  // CHECK: "sig_name":"someOutput"
  %z = hw.instance "instZ" @SingleResult() -> (outPort: i1)
  dbg.variable "varZ", %z : i1
  hw.output %z : i1
}

// CHECK-LABEL: "obj_name": "Issue6735_Case2"
hw.module @Issue6735_Case2(out x : i36, out y : i36 {hw.verilogName = "verilogY"}) {
  %a, %b = hw.instance "bar" @MultipleResults() -> (a: i36, b: i36)
  // CHECK: "var_name": "portA"
  // CHECK-NOT: "field":"a"
  // CHECK-NOT: "var_ref":
  // CHECK-NOT: "sig_name":"bar"
  // CHECK: "sig_name":"x"
  dbg.variable "portA", %a : i36
  // CHECK: "var_name": "portB"
  // CHECK-NOT: "field":"b"
  // CHECK-NOT: "var_ref":
  // CHECK-NOT: "sig_name":"bar"
  // CHECK: "sig_name":"verilogY"
  dbg.variable "portB", %b : i36
  hw.output %a, %b : i36, i36
}
hw.module.extern @MultipleResults(out a : i36, out b : i36)

// CHECK-LABEL: "obj_name": "Issue6749"
hw.module @Issue6749(in %a: i42) {
  // Variables with empty names must have a non-empty name in the output.
  // CHECK-NOT: "var_name": ""
  dbg.variable "", %a : i42

  // Uniquify duplicate variable names.
  // CHECK: "var_name": "myVar"
  // CHECK-NOT: "var_name": "myVar"
  // CHECK: "var_name": "myVar_0"
  dbg.variable "myVar", %a : i42
  dbg.variable "myVar", %a : i42

  // Uniquify Verilog keyword collisions.
  // CHECK-NOT: "var_name": "signed"
  // CHECK: "var_name": "signed_0"
  dbg.variable "signed", %a : i42

  // Scopes with empty names must have a non-empty name in the output.
  // CHECK: "children": [
  // CHECK-NOT: "name": ""
  %scope = dbg.scope "", "SomeScope"
}
