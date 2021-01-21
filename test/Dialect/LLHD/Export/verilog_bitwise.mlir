//RUN: circt-translate --export-llhd-verilog %s | FileCheck %s

// CHECK-LABEL: _check_bitwise
llhd.entity @check_bitwise() -> () {
  // CHECK-NEXT: wire [63:0] _[[A:.*]] = 64'd42;
  %a = llhd.const 42 : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = ~_[[A]];
  %0 = llhd.not %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] & _[[A]];
  %1 = llhd.and %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] | _[[A]];
  %2 = llhd.or %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] ^ _[[A]];
  %3 = llhd.xor %a, %a : i64

  // CHECK-NEXT: wire [4:0] _[[HIDDEN:.*]] = 5'd0;
  %hidden = llhd.const 0 : i5
  // CHECK-NEXT: wire [1:0] _[[AMT:.*]] = 2'd3;
  %amt = llhd.const 3 : i2

  // CHECK-NEXT: wire [68:0] _[[TMP0:.*]] = {_[[A]], _[[HIDDEN]]};
  // CHECK-NEXT: wire [68:0] _[[TMP1:.*]] = _[[TMP0]] << _[[AMT]];
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[TMP1]][68:5];
  %4 = llhd.shl %a, %hidden, %amt : (i64, i5, i2) -> i64
  // CHECK-NEXT: wire [68:0] _[[TMP0:.*]] = {_[[HIDDEN]], _[[A]]};
  // CHECK-NEXT: wire [68:0] _[[TMP1:.*]] = _[[TMP0]] << _[[AMT]];
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[TMP1]][63:0];
  %5 = llhd.shr %a, %hidden, %amt : (i64, i5, i2) -> i64
}
