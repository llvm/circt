//RUN: circt-translate --export-llhd-verilog %s | FileCheck %s

// CHECK-LABEL: _check_relations
llhd.entity @check_relations() -> () {
  // CHECK-NEXT: wire [63:0] _[[A:.*]] = 64'd42;
  %a = llhd.const 42 : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] == _[[A]];
  %1 = cmpi eq, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] != _[[A]];
  %2 = cmpi ne, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] >= _[[A]];
  %3 = cmpi uge, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] > _[[A]];
  %4 = cmpi ugt, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] <= _[[A]];
  %5 = cmpi ule, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] < _[[A]];
  %6 = cmpi ult, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = $signed(_[[A]]) >= $signed(_[[A]]);
  %7 = cmpi sge, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = $signed(_[[A]]) > $signed(_[[A]]);
  %8 = cmpi sgt, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = $signed(_[[A]]) <= $signed(_[[A]]);
  %9 = cmpi sle, %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = $signed(_[[A]]) < $signed(_[[A]]);
  %10 = cmpi slt, %a, %a : i64
}
