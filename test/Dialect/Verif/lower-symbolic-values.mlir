// RUN: circt-opt --verif-lower-symbolic-values=mode=extmodule %s | FileCheck --check-prefixes=CHECK,CHECK-EXTMODULE %s
// RUN: circt-opt --verif-lower-symbolic-values=mode=yosys %s | FileCheck --check-prefixes=CHECK,CHECK-YOSYS %s

// CHECK-LABEL: hw.module @Foo
hw.module @Foo() {
  // CHECK-EXTMODULE-NOT: verif.symbolic_value
  // CHECK-YOSYS-NOT: verif.symbolic_value

  // CHECK-EXTMODULE: [[SYM:%.+]] = hw.instance {{.*}} @circt.symbolic_value.42<WIDTH: i32 = 42>
  // CHECK-YOSYS: [[TMP:%.+]] = sv.wire {sv.attributes = [#sv.attribute<"anyseq">]} : !hw.inout<i42>
  // CHECK-YOSYS: [[SYM:%.+]] = sv.read_inout [[TMP]]
  // CHECK: dbg.variable "x0", [[SYM]] : i42
  %0 = verif.symbolic_value : i42
  dbg.variable "x0", %0 : i42

  // CHECK-EXTMODULE: [[TMP:%.+]] = hw.instance {{.*}} @circt.symbolic_value.12<WIDTH: i32 = 12>
  // CHECK-EXTMODULE: [[SYM:%.+]] = hw.bitcast [[TMP]] : (i12) -> !hw.array<4xi3>
  // CHECK-YOSYS: [[TMP:%.+]] = sv.wire {sv.attributes = [#sv.attribute<"anyseq">]} : !hw.inout<array<4xi3>>
  // CHECK-YOSYS: [[SYM:%.+]] = sv.read_inout [[TMP]]
  // CHECK: dbg.variable "x1", [[SYM]] : !hw.array<4xi3>
  %1 = verif.symbolic_value : !hw.array<4xi3>
  dbg.variable "x1", %1 : !hw.array<4xi3>

  // Reuse existing extmodule for same i42 type.
  // CHECK-EXTMODULE: [[SYM:%.+]] = hw.instance {{.*}} @circt.symbolic_value.42<WIDTH: i32 = 42>
  // CHECK-YOSYS: [[TMP:%.+]] = sv.wire {sv.attributes = [#sv.attribute<"anyseq">]} : !hw.inout<i42>
  // CHECK-YOSYS: [[SYM:%.+]] = sv.read_inout [[TMP]]
  // CHECK: dbg.variable "x2", [[SYM]] : i42
  %2 = verif.symbolic_value : i42
  dbg.variable "x2", %2 : i42

  // Reuse existing extmodule for same 42 bit types, cast to array<6 x i7>.
  // CHECK-EXTMODULE: [[TMP:%.+]] = hw.instance {{.*}} @circt.symbolic_value.42<WIDTH: i32 = 42>
  // CHECK-EXTMODULE: [[SYM:%.+]] = hw.bitcast [[TMP]] : (i42) -> !hw.array<6xi7>
  // CHECK-YOSYS: [[TMP:%.+]] = sv.wire {sv.attributes = [#sv.attribute<"anyseq">]} : !hw.inout<array<6xi7>>
  // CHECK-YOSYS: [[SYM:%.+]] = sv.read_inout [[TMP]]
  // CHECK: dbg.variable "x3", [[SYM]] : !hw.array<6xi7>
  %3 = verif.symbolic_value : !hw.array<6xi7>
  dbg.variable "x3", %3 : !hw.array<6xi7>
}

// CHECK-EXTMODULE: hw.module.extern @circt.symbolic_value.42<WIDTH: i32>
// CHECK-EXTMODULE-SAME: verilogName = "circt_symbolic_value"

// CHECK-EXTMODULE: hw.module.extern @circt.symbolic_value.12<WIDTH: i32>
// CHECK-EXTMODULE-SAME: verilogName = "circt_symbolic_value"
