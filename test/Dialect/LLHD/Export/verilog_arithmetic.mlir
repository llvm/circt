//RUN: circt-translate --export-llhd-verilog %s | FileCheck %s

// CHECK-LABEL: _check_arithmetic
llhd.entity @check_arithmetic() -> () {
  // CHECK-NEXT: wire [63:0] _[[A:.*]] = 64'd42;
  %a = llhd.const 42 : i64

  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]];
  %1 = comb.add %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]];
  %2 = comb.mul %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] + _[[A]] + _[[A]];
  %3 = comb.add %a, %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] * _[[A]] * _[[A]];
  %4 = comb.mul %a, %a, %a : i64

  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] - _[[A]];
  %5 = comb.sub %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] / _[[A]];
  %6 = comb.divu %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = $signed(_[[A]]) / $signed(_[[A]]);
  %7 = comb.divs %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] % _[[A]];
  %8 = comb.modu %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = $signed(_[[A]]) % $signed(_[[A]]);
  %9 = comb.mods %a, %a : i64
}
