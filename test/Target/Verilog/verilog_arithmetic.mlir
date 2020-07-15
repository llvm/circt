//RUN: circt-translate --llhd-to-verilog %s | FileCheck %s

// CHECK-LABEL: _check_arithmetic
llhd.entity @check_arithmetic() -> () {
  // CHECK-NEXT: wire [63:0] _[[A:.*]] = 64'd42;
  %a = llhd.const 42 : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = -_[[A]];
  %0 = llhd.neg %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] + _[[A]];
  %1 = addi %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] - _[[A]];
  %2 = subi %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] * _[[A]];
  %3 = muli %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] / _[[A]];
  %4 = divi_unsigned %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = $signed(_[[A]]) / $signed(_[[A]]);
  %5 = divi_signed %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] % _[[A]];
  %6 = remi_unsigned %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = $signed(_[[A]]) % $signed(_[[A]]);
  %7 = remi_signed %a, %a : i64
}
