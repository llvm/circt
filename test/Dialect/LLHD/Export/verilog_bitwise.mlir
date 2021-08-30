//RUN: circt-translate --export-llhd-verilog %s | FileCheck %s

// CHECK-LABEL: _check_bitwise
llhd.entity @check_bitwise() -> () {
  // CHECK-NEXT: wire [63:0] _[[A:.*]] = 64'd42;
  %a = llhd.const 42 : i64

  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]];
  %1 = comb.and %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]];
  %2 = comb.or %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]];
  %3 = comb.xor %a : i64

  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] & _[[A]] & _[[A]];
  %4 = comb.and %a, %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] | _[[A]] | _[[A]];
  %5 = comb.or %a, %a, %a : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] ^ _[[A]] ^ _[[A]];
  %6 = comb.xor %a, %a, %a : i64

  // CHECK-NEXT: wire [4:0] _[[HIDDEN:.*]] = 5'd0;
  %hidden = llhd.const 0 : i5
  // CHECK-NEXT: wire [1:0] _[[AMT:.*]] = 2'd3;
  %amt = llhd.const 3 : i2
  // CHECK-NEXT: wire [63:0] _[[AMT64:.*]] = 64'd3;
  %amt64 = llhd.const 3 : i64

  // CHECK-NEXT: wire [68:0] _[[TMP0:.*]] = {_[[A]], _[[HIDDEN]]};
  // CHECK-NEXT: wire [68:0] _[[TMP1:.*]] = _[[TMP0]] << _[[AMT]];
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[TMP1]][68:5];
  %7 = llhd.shl %a, %hidden, %amt : (i64, i5, i2) -> i64
  // CHECK-NEXT: wire [68:0] _[[TMP0:.*]] = {_[[HIDDEN]], _[[A]]};
  // CHECK-NEXT: wire [68:0] _[[TMP1:.*]] = _[[TMP0]] << _[[AMT]];
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[TMP1]][63:0];
  %8 = llhd.shr %a, %hidden, %amt : (i64, i5, i2) -> i64

  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] << _[[AMT64]];
  %9 = comb.shl %a, %amt64 : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[A]] >> _[[AMT64]];
  %10 = comb.shru %a, %amt64 : i64
  // CHECK-NEXT: wire [63:0] _{{.*}} = $signed(_[[A]]) >>> $signed(_[[AMT64]]);
  %11 = comb.shrs %a, %amt64 : i64

  // CHECK-NEXT: wire _{{.*}} = ^_[[A]];
  %12 = comb.parity %a : i64

  // CHECK-NEXT: wire [31:0] _[[EXT:.*]] = _[[A]][36:5];
  %13 = comb.extract %a from 5 : (i64) -> i32

  // CHECK-NEXT: wire [63:0] _[[SEXT:.*]] = {{[{][{]}}32{_[[EXT]][31]{{[}][}]}}, _[[EXT]]};
  %14 = comb.sext %13 : (i32) -> i64

  // CHECK-NEXT: wire [191:0] _{{.*}} = {_[[A]], _[[SEXT]], _[[A]]};
  %15 = comb.concat %a, %14, %a : (i64, i64, i64) -> i192

  // CHECK-NEXT: wire _[[COND:.*]] = 1'd1;
  // CHECK-NEXT: wire [63:0] _{{.*}} = _[[COND]] ? _[[A]] : _[[SEXT]];
  %cond = llhd.const 1 : i1
  %16 = comb.mux %cond, %a, %14 : i64
}
