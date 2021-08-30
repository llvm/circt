//RUN: circt-translate --export-llhd-verilog %s | FileCheck %s

// CHECK-LABEL: _check_relations
llhd.entity @check_relations() -> () {
  // CHECK-NEXT: wire [63:0] _[[A:.*]] = 64'd42;
  %a = llhd.const 42 : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] == _[[A]];
  %1 = comb.icmp eq %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] != _[[A]];
  %2 = comb.icmp ne %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] >= _[[A]];
  %3 = comb.icmp uge %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] > _[[A]];
  %4 = comb.icmp ugt %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] <= _[[A]];
  %5 = comb.icmp ule %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = _[[A]] < _[[A]];
  %6 = comb.icmp ult %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = $signed(_[[A]]) >= $signed(_[[A]]);
  %7 = comb.icmp sge %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = $signed(_[[A]]) > $signed(_[[A]]);
  %8 = comb.icmp sgt %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = $signed(_[[A]]) <= $signed(_[[A]]);
  %9 = comb.icmp sle %a, %a : i64
  // CHECK-NEXT: wire  _{{.*}} = $signed(_[[A]]) < $signed(_[[A]]);
  %10 = comb.icmp slt %a, %a : i64
}
