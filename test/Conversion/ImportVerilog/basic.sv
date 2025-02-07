// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Ignore time unit and precision.
timeunit 100ps;
timeprecision 10fs;
timeunit 100ps/10fs;
`timescale 100ps/10fs;

// Ignore type aliases and enum variant names imported into the parent scope.
typedef int MyInt;
typedef enum { VariantA, VariantB } MyEnum;

// Ignore imports.
import Package::*;

// CHECK-LABEL: moore.module @Empty() {
// CHECK:       }
module Empty;
  ; // empty member
endmodule


// CHECK: moore.module private @DedupA(in %a : !moore.i32)
// CHECK-NOT: @DedupA
module DedupA(input int a);
endmodule

// CHECK: moore.module private @DedupB(in %a : !moore.i32)
// CHECK: moore.module private @DedupB_0(in %a : !moore.i16)
// CHECK-NOT: @DedupB
module DedupB #(parameter int W, parameter type T = bit [W-1:0])(input T a);
endmodule

// CHECK-LABEL: moore.module @Dedup()
module Dedup;
  int a;
  shortint b;
  // CHECK-LABEL: moore.instance "a0" @DedupA(
  DedupA a0(a);
  // CHECK-LABEL: moore.instance "a1" @DedupA(
  DedupA a1(a);
  // CHECK-LABEL: moore.instance "b0" @DedupB(
  DedupB #(32) b0(a);
  // CHECK-LABEL: moore.instance "b1" @DedupB(
  DedupB #(32) b1(a);
  // CHECK-LABEL: moore.instance "b2" @DedupB_0(
  DedupB #(16) b2(b);
endmodule

// CHECK-LABEL: moore.module @NestedA() {
// CHECK:         moore.instance "NestedB" @NestedB
// CHECK:       }
// CHECK-LABEL: moore.module private @NestedB() {
// CHECK:         moore.instance "NestedC" @NestedC
// CHECK:       }
// CHECK-LABEL: moore.module private @NestedC() {
// CHECK:       }
module NestedA;
  module NestedB;
    module NestedC;
    endmodule
  endmodule
endmodule

// CHECK-LABEL: moore.module private @Child() {
// CHECK:       }
module Child;
endmodule

// CHECK-LABEL: moore.module @Parent() {
// CHECK:         moore.instance "child" @Child
// CHECK:       }
module Parent;
  Child child();
endmodule

// CHECK-LABEL: moore.module @Basic
module Basic;
  // CHECK: %v0 = moore.variable : <l1>
  // CHECK: %v1 = moore.variable : <i32>
  // CHECK: [[TMP1:%.+]] = moore.read %v1 :
  // CHECK: %v2 = moore.variable [[TMP1]] : <i32>
  var v0;
  int v1;
  int v2 = v1;

  // CHECK: %w0 = moore.net wire : <l1>
  wire w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w1 = moore.net wire [[TMP1]] : <l1>
  wire w1 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w2 = moore.net uwire [[TMP1]] : <l1>
  uwire w2 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w3 = moore.net tri [[TMP1]] : <l1>
  tri w3 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w4 = moore.net triand [[TMP1]] : <l1>
  triand w4 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w5 = moore.net trior [[TMP1]] : <l1>
  trior w5 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w6 = moore.net wand [[TMP1]] : <l1>
  wand w6 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w7 = moore.net wor [[TMP1]] : <l1>
  wor w7 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w8 = moore.net trireg [[TMP1]] : <l1>
  trireg w8 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w9 = moore.net tri0 [[TMP1]] : <l1>
  tri0 w9 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0
  // CHECK: %w10 = moore.net tri1 [[TMP1]] : <l1>
  tri1 w10 = w0;
  // CHECK: %w11 = moore.net supply0 : <l1>
  supply0 w11;
  // CHECK: %w12 = moore.net supply1 : <l1>
  supply1 w12;

  // CHECK: %b1 = moore.variable : <i1>
  // CHECK: [[TMP1:%.+]] = moore.read %b1
  // CHECK: %b2 = moore.variable [[TMP1]] : <i1>
  bit [0:0] b1;
  bit b2 = b1;

  // CHECK: moore.procedure initial {
  // CHECK: }
  initial;

  // CHECK: moore.procedure final {
  // CHECK: }
  final begin end

  // CHECK: moore.procedure always {
  // CHECK:   %x = moore.variable
  // CHECK:   %y = moore.variable
  // CHECK: }
  always begin
    int x;
    begin
      int y;
    end
  end

  // CHECK: moore.procedure always_comb {
  // CHECK: }
  always_comb begin end

  // CHECK: moore.procedure always_latch {
  // CHECK: }
  always_latch begin end

  // CHECK: moore.procedure always_ff {
  // CHECK: }
  always_ff @* begin end

  // CHECK: [[TMP1:%.+]] = moore.read %v2
  // CHECK: moore.assign %v1, [[TMP1]] : i32
  assign v1 = v2;

  // CHECK: %pkgType0 = moore.variable : <l42>
  PackageType pkgType0;
  // CHECK: %pkgType1 = moore.variable : <l42>
  Package::PackageType pkgType1;

  // CHECK: [[VARIANT_A:%.+]] = moore.constant 0 :
  // CHECK: %ev1 = moore.variable [[VARIANT_A]]
  // CHECK: [[VARIANT_B:%.+]] = moore.constant 1 :
  // CHECK: %ev2 = moore.variable [[VARIANT_B]]
  MyEnum ev1 = VariantA;
  MyEnum ev2 = VariantB;

  // CHECK: [[STR_WELCOME:%.+]] = moore.string_constant "Welcome to Moore" : i128
  // CHECK: [[CONV_WELCOME:%.+]] = moore.conversion [[STR_WELCOME]] : !moore.i128 -> !moore.string
  // CHECK: [[VAR_S:%.+]] = moore.variable [[CONV_WELCOME]] : <string>
  string s = "Welcome to Moore";

  // CHECK: [[VAR_S1:%.+]] = moore.variable : <string>
  // CHECK: [[STR_HELLO:%.+]] = moore.string_constant "Hello World" : i88
  // CHECK: [[CONV_HELLO:%.+]] = moore.conversion [[STR_HELLO]] : !moore.i88 -> !moore.string
  // CHECK: moore.assign [[VAR_S1]], [[CONV_HELLO]] : string
  string s1; 
  assign s1 = "Hello World";

  typedef struct packed { bit x; bit y; } MyStruct;
  // CHECK: [[VAR_S2:%.+]] = moore.variable : <struct<{x: i1, y: i1}>>
  MyStruct s2;
  // CHECK: [[TMP1:%.+]] = moore.read [[VAR_S2]]
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.struct<{x: i1, y: i1}> -> !moore.i2
  // CHECK: [[TMP3:%.+]] = moore.not [[TMP2]] : i2
  // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.i2 -> !moore.struct<{x: i1, y: i1}>
  // CHECK: moore.assign [[VAR_S2]], [[TMP4]]
  assign s2 = ~s2;
  // CHECK: [[TMP1:%.+]] = moore.read [[VAR_S2]]
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.struct<{x: i1, y: i1}> -> !moore.i2
  // CHECK: [[TMP3:%.+]] = moore.not [[TMP2]] : i2
  // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.i2 -> !moore.struct<{x: i1, y: i1}>
  // CHECK: [[VAR_S3:%.+]] = moore.variable [[TMP4]] : <struct<{x: i1, y: i1}>>
  MyStruct s3 = ~s2;
endmodule

// CHECK-LABEL: func.func private @dummyA(
// CHECK-LABEL: func.func private @dummyB(
// CHECK-LABEL: func.func private @dummyC(
// CHECK-LABEL: func.func private @dummyD(
// CHECK-LABEL: func.func private @dummyE(
// CHECK-LABEL: func.func private @dummyF(
function void dummyA(); endfunction
function void dummyB(); endfunction
function void dummyC(); endfunction
function void dummyD(int a); endfunction
function bit dummyE(bit a); return a; endfunction
function void dummyF(integer a); endfunction

// CHECK-LABEL: moore.module @Parameters(
module Parameters;
  // CHECK: [[TMP:%.+]] = moore.constant 1 : l32
  // CHECK: dbg.variable "p1", [[TMP]] : !moore.l32
  parameter p1 = 1;

  // CHECK: [[TMP:%.+]] = moore.constant 1 : l32
  // CHECK: dbg.variable "p2", [[TMP]] : !moore.l32
  parameter p2 = p1;

  // CHECK: [[TMP:%.+]] = moore.constant 2 : l32
  // CHECK: dbg.variable "lp1", [[TMP]] : !moore.l32
  localparam lp1 = 2;

  // CHECK: [[TMP:%.+]] = moore.constant 2 : l32
  // CHECK: dbg.variable "lp2", [[TMP]] : !moore.l32
  localparam lp2 = lp1;

  // CHECK: [[TMP:%.+]] = moore.constant 3 : l32
  // CHECK: dbg.variable "sp1", [[TMP]] : !moore.l32
  specparam sp1 = 3;

  // CHECK: [[TMP:%.+]] = moore.constant 3 : l32
  // CHECK: dbg.variable "sp2", [[TMP]] : !moore.l32
  specparam sp2 = sp1;

  initial begin
    // CHECK: [[TMP:%.+]] = moore.constant 1 : l32
    // CHECK: func.call @dummyF([[TMP]])
    // CHECK: [[TMP:%.+]] = moore.constant 1 : l32
    // CHECK: func.call @dummyF([[TMP]])
    // CHECK: [[TMP:%.+]] = moore.constant 2 : l32
    // CHECK: func.call @dummyF([[TMP]])
    // CHECK: [[TMP:%.+]] = moore.constant 2 : l32
    // CHECK: func.call @dummyF([[TMP]])
    // CHECK: [[TMP:%.+]] = moore.constant 3 : l32
    // CHECK: func.call @dummyF([[TMP]])
    // CHECK: [[TMP:%.+]] = moore.constant 3 : l32
    // CHECK: func.call @dummyF([[TMP]])
    dummyF(p1);
    dummyF(p2);
    dummyF(lp1);
    dummyF(lp2);
    dummyF(sp1);
    dummyF(sp2);
  end
endmodule

// CHECK-LABEL: func.func private @ConditionalStatements(
// CHECK-SAME: %arg0: !moore.i1
// CHECK-SAME: %arg1: !moore.i1
function void ConditionalStatements(bit x, bit y);
  // CHECK: [[COND:%.+]] = moore.conversion %arg0 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND]], ^[[BB1:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB2]]
  // CHECK: ^[[BB2]]:
  if (x) dummyA();

  // CHECK: [[COND1:%.+]] = moore.and %arg0, %arg1
  // CHECK: [[COND2:%.+]] = moore.conversion [[COND1]] : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND2]], ^[[BB1:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB2]]
  // CHECK: ^[[BB2]]:
  if (x &&& y) dummyA();

  // CHECK: [[COND:%.+]] = moore.conversion %arg0 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND]], ^[[BB1:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB3:.+]]
  // CHECK: ^[[BB2]]:
  // CHECK: call @dummyB()
  // CHECK: cf.br ^[[BB3]]
  // CHECK: ^[[BB3]]:
  if (x)
    dummyA();
  else
    dummyB();

  // CHECK: [[COND:%.+]] = moore.conversion %arg0 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND]], ^[[BB1:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB6:.+]]
  // CHECK: ^[[BB2]]:
  // CHECK: [[COND:%.+]] = moore.conversion %arg1 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND]], ^[[BB3:.+]], ^[[BB4:.+]]
  // CHECK: ^[[BB3]]:
  // CHECK: call @dummyB()
  // CHECK: cf.br ^[[BB5:.+]]
  // CHECK: ^[[BB4]]:
  // CHECK: call @dummyC()
  // CHECK: cf.br ^[[BB5]]
  // CHECK: ^[[BB5]]:
  // CHECK: cf.br ^[[BB6]]
  // CHECK: ^[[BB6]]:
  if (x)
    dummyA();
  else if (y)
    dummyB();
  else
    dummyC();

  // CHECK: [[COND:%.+]] = moore.conversion %arg0 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND]], ^[[BB1:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK: return
  // CHECK: ^[[BB2]]:
  if (x) return;
endfunction

// CHECK-LABEL: func.func private @CaseStatements(
// CHECK-SAME: %arg0: !moore.i32
// CHECK-SAME: %arg1: !moore.i32
// CHECK-SAME: %arg2: !moore.i32
// CHECK-SAME: %arg3: !moore.i32
function void CaseStatements(int x, int a, int b, int c);
  // CHECK: [[FLAG:%.+]] = moore.add %arg0, %arg0
  case (x + x)
    // CHECK: [[COND1:%.+]] = moore.case_eq [[FLAG]], %arg1
    // CHECK: [[COND2:%.+]] = moore.conversion [[COND1]] : !moore.i1 -> i1
    // CHECK: cf.cond_br [[COND2]], ^[[BB1:.+]], ^[[BB2:.+]]
    // CHECK: ^[[BB1]]:
    // CHECK: call @dummyA()
    // CHECK: cf.br ^[[BB3:.+]]
    a: dummyA();
    // CHECK: ^[[BB2]]:
    // CHECK: call @dummyB()
    // CHECK: cf.br ^[[BB3]]
    default: dummyB();
    // CHECK: ^[[BB3]]:
  endcase

  // CHECK: [[COND1:%.+]] = moore.case_eq %arg0, %arg1
  // CHECK: [[COND2:%.+]] = moore.conversion [[COND1]] : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND2]], ^[[BB_MATCH:.+]], ^[[BB1:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK: [[TMP:%.+]] = moore.add %arg2, %arg3
  // CHECK: [[COND1:%.+]] = moore.case_eq %arg0, [[TMP]]
  // CHECK: [[COND2:%.+]] = moore.conversion [[COND1]] : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND2]], ^[[BB_MATCH:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB_MATCH]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB_EXIT:.+]]
  // CHECK: ^[[BB2]]:
  // CHECK: call @dummyB()
  // CHECK: cf.br ^[[BB_EXIT]]
  // CHECK: ^[[BB_EXIT]]:
  case (x)
    a, (b+c): dummyA();
    default: dummyB();
  endcase

  // CHECK: [[COND1:%.+]] = moore.casez_eq %arg0, %arg1
  // CHECK: [[COND2:%.+]] = moore.conversion [[COND1]] : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND2]], ^[[BB1:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB3:.+]]
  // CHECK: ^[[BB2]]:
  // CHECK: cf.br ^[[BB3]]
  // CHECK: ^[[BB3]]:
  casez (x)
    a: dummyA();
  endcase

  // CHECK: [[COND1:%.+]] = moore.casexz_eq %arg0, %arg1
  // CHECK: [[COND2:%.+]] = moore.conversion [[COND1]] : !moore.i1 -> i1
  // CHECK: cf.cond_br [[COND2]], ^[[BB1:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB3:.+]]
  // CHECK: ^[[BB2]]:
  // CHECK: cf.br ^[[BB3]]
  // CHECK: ^[[BB3]]:
  casex (x)
    a: dummyA();
  endcase
endfunction

// CHECK-LABEL: func.func private @ForLoopStatements(
// CHECK-SAME: %arg0: !moore.i32
// CHECK-SAME: %arg1: !moore.i32
// CHECK-SAME: %arg2: !moore.i1
function void ForLoopStatements(int a, int b, bit c);
  int x;

  // CHECK: moore.blocking_assign %x, %arg0
  // CHECK: cf.br ^[[BB_CHECK:.+]]
  // CHECK: ^[[BB_CHECK]]:
  // CHECK: [[TMP1:%.+]] = moore.read %x
  // CHECK: [[TMP2:%.+]] = moore.slt [[TMP1]], %arg1
  // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i1 -> i1
  // CHECK: cf.cond_br [[TMP3]], ^[[BB_BODY:.+]], ^[[BB_EXIT:.+]]
  // CHECK: ^[[BB_BODY]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB_STEP:.+]]
  // CHECK: ^[[BB_STEP]]:
  // CHECK: call @dummyB()
  // CHECK: cf.br ^[[BB_CHECK]]
  // CHECK: ^[[BB_EXIT]]:
  for (x = a; x < b; dummyB()) dummyA();

  // CHECK: %y = moore.variable %arg0 : <i32>
  // CHECK: cf.br ^[[BB_CHECK:.+]]
  // CHECK: ^[[BB_CHECK]]:
  // CHECK: [[TMP1:%.+]] = moore.read %y
  // CHECK: [[TMP2:%.+]] = moore.slt [[TMP1]], %arg1
  // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i1 -> i1
  // CHECK: cf.cond_br [[TMP3]], ^[[BB_BODY:.+]], ^[[BB_EXIT:.+]]
  // CHECK: ^[[BB_BODY]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB_STEP:.+]]
  // CHECK: ^[[BB_STEP]]:
  // CHECK: call @dummyB()
  // CHECK: cf.br ^[[BB_CHECK]]
  // CHECK: ^[[BB_EXIT]]:
  for (int y = a; y < b; dummyB()) dummyA();

  // CHECK: cf.br ^[[BB_CHECK:.+]]
  // CHECK: ^[[BB_CHECK]]:
  // CHECK: [[TMP:%.+]] = moore.conversion %arg2 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[TMP]], ^[[BB_BODY:.+]], ^[[BB_EXIT:.+]]
  // CHECK: ^[[BB_BODY]]:
  // CHECK: [[TMP:%.+]] = moore.conversion %arg2 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[TMP]], ^[[BB_TRUE:.+]], ^[[BB_FALSE:.+]]
  // CHECK: ^[[BB_TRUE]]:
  // CHECK: cf.br ^[[BB_STEP:.+]]
  // CHECK: ^[[BB_FALSE]]:
  // CHECK: cf.br ^[[BB_EXIT]]
  // CHECK: ^[[BB_STEP]]:
  // CHECK: call @dummyB()
  // CHECK: cf.br ^[[BB_CHECK]]
  // CHECK: ^[[BB_EXIT]]:
  for (; c; dummyB())
    if (c)
      continue;
    else
      break;
endfunction

// CHECK-LABEL: func.func private @ForeverLoopStatements(
// CHECK-SAME: %arg0: !moore.i1
// CHECK-SAME: %arg1: !moore.i1
function void ForeverLoopStatements(bit x, bit y);
  // CHECK: cf.br ^[[BB_BODY:.+]]
  // CHECK: ^[[BB_BODY]]:
  forever begin
    if (x) begin
      // CHECK: call @dummyA()
      // CHECK: cf.br ^[[BB_EXIT:.+]]
      dummyA();
      break;
    end else begin
      // CHECK: call @dummyB()
      // CHECK: cf.br ^[[BB_BODY]]
      dummyB();
      continue;
    end
  end
  // CHECK: ^[[BB_EXIT]]:

  // CHECK: cf.br ^[[BB_BODY:.+]]
  // CHECK: ^[[BB_BODY]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB_BODY]]
  forever dummyA();
endfunction

// CHECK: func.func private @ForeachStatements(%[[ARG0:.*]]: !moore.i32, %[[ARG1:.*]]: !moore.i1) {
function void ForeachStatements(int x, bit y);
// CHECK: %[[ARRAY:.*]] = moore.variable : <uarray<3 x uarray<3 x uarray<3 x uarray<8 x l8>>>>>
  logic [7:0] array [3:1][4:2][5:3][6:-1];
// CHECK: %[[C2:.*]] = moore.constant 2 : i32
// CHECK: %[[I:.*]] = moore.variable %[[C2]] : <i32>
// CHECK: cf.br ^[[BB1:.*]]
// CHECK: ^[[BB1]]:
// CHECK: %[[C4:.*]] = moore.constant 4 : i32
// CHECK: %[[I_VAL:.*]] = moore.read %[[I]] : <i32>
// CHECK: %[[CMP1:.*]] = moore.sle %[[I_VAL]], %[[C4]] : i32 -> i1
// CHECK: %[[CONV1:.*]] = moore.conversion %[[CMP1]] : !moore.i1 -> i1
// CHECK: cf.cond_br %[[CONV1]], ^[[BB2:.*]], ^[[BB10:.*]]
// CHECK: ^[[BB2]]:
// CHECK: %[[CM1:.*]] = moore.constant -1 : i32
// CHECK: %[[J:.*]] = moore.variable %[[CM1]] : <i32>
// CHECK: cf.br ^[[BB3:.*]]
// CHECK: ^[[BB3]]:
// CHECK: %[[C6:.*]] = moore.constant 6 : i32
// CHECK: %[[J_VAL:.*]] = moore.read %[[J]] : <i32>
// CHECK: %[[CMP2:.*]] = moore.sle %[[J_VAL]], %[[C6]] : i32 -> i1
// CHECK: %[[CONV2:.*]] = moore.conversion %[[CMP2]] : !moore.i1 -> i1
// CHECK: cf.cond_br %[[CONV2]], ^[[BB4:.*]], ^[[BB8:.*]]
  foreach (array[, i, ,j]) begin
// CHECK: ^[[BB4]]:
// CHECK: %[[CONV3:.*]] = moore.conversion %[[ARG1]] : !moore.i1 -> i1
// CHECK: cf.cond_br %[[CONV3]], ^[[BB5:.*]], ^[[BB6:.*]]
    if (y) begin
// CHECK: ^[[BB5]]:
// CHECK: call @dummyA() : () -> ()
// CHECK: cf.br ^[[BB8]]
      dummyA();
      break;
    end else begin
// CHECK: ^[[BB6]]:
// CHECK: call @dummyB() : () -> ()
// CHECK: cf.br ^[[BB7:.*]]
      dummyB();
      continue;
    end
// CHECK: ^[[BB7]]:
// CHECK: %[[J_VAL2:.*]] = moore.read %[[J]] : <i32>
// CHECK: %[[C1_1:.*]] = moore.constant 1 : i32
// CHECK: %[[ADD1:.*]] = moore.add %[[J_VAL2]], %[[C1_1]] : i32
// CHECK: moore.blocking_assign %[[J]], %[[ADD1]] : i32
// CHECK: cf.br ^[[BB3]]
// CHECK: ^[[BB8]]:
// CHECK: cf.br ^[[BB9:.*]]
// CHECK: ^[[BB9]]:
// CHECK: %[[I_VAL2:.*]] = moore.read %[[I]] : <i32>
// CHECK: %[[C1_2:.*]] = moore.constant 1 : i32
// CHECK: %[[ADD2:.*]] = moore.add %[[I_VAL2]], %[[C1_2]] : i32
// CHECK: moore.blocking_assign %[[I]], %[[ADD2]] : i32
// CHECK: cf.br ^[[BB1]]
// CHECK: ^[[BB10]]:
// CHECK: return
  end
endfunction


// CHECK-LABEL: func.func private @WhileLoopStatements(
// CHECK-SAME: %arg0: !moore.i1
// CHECK-SAME: %arg1: !moore.i1
function void WhileLoopStatements(bit x, bit y);
  // CHECK: cf.br ^[[BB_CHECK:.+]]
  // CHECK: ^[[BB_CHECK]]:
  // CHECK: [[TMP:%.+]] = moore.conversion %arg0 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[TMP]], ^[[BB_BODY:.+]], ^[[BB_EXIT:.+]]
  // CHECK: ^[[BB_BODY]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB_CHECK]]
  // CHECK: ^[[BB_EXIT]]:
  while (x) dummyA();

  // CHECK: cf.br ^[[BB_BODY:.+]]
  // CHECK: ^[[BB_BODY]]:
  // CHECK: call @dummyA()
  // CHECK: cf.br ^[[BB_CHECK:.+]]
  // CHECK: ^[[BB_CHECK]]:
  // CHECK: [[TMP:%.+]] = moore.conversion %arg0 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[TMP]], ^[[BB_BODY]], ^[[BB_EXIT:.+]]
  // CHECK: ^[[BB_EXIT]]:
  do dummyA(); while (x);

  // CHECK: cf.br ^[[BB_CHECK:.+]]
  // CHECK: ^[[BB_CHECK]]:
  // CHECK: [[TMP:%.+]] = moore.conversion %arg0 : !moore.i1 -> i1
  // CHECK: cf.cond_br [[TMP]], ^[[BB_BODY:.+]], ^[[BB_EXIT:.+]]
  // CHECK: ^[[BB_BODY]]:
  while (x) begin
    if (y) begin
      // CHECK: call @dummyA()
      // CHECK: cf.br ^[[BB_EXIT]]
      dummyA();
      break;
    end else begin
      // CHECK: call @dummyB()
      // CHECK: cf.br ^[[BB_CHECK]]
      dummyB();
      continue;
    end
  end
  // CHECK: ^[[BB_EXIT]]:
endfunction

// CHECK-LABEL: func.func private @RepeatLoopStatements(
// CHECK-SAME: %arg0: !moore.i32
// CHECK-SAME: %arg1: !moore.i1
function void RepeatLoopStatements(int x, bit y);
  // CHECK: cf.br ^[[BB_CHECK:.+]](%arg0 : !moore.i32)
  repeat (x) begin
    // CHECK: ^[[BB_CHECK]]([[COUNT:%.+]]: !moore.i32):
    // CHECK: [[TMP1:%.+]] = moore.bool_cast [[COUNT]] : i32 -> i1
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i1 -> i1
    // CHECK: cf.cond_br [[TMP2]], ^[[BB_BODY:.+]], ^[[BB_EXIT:.+]]
    // CHECK: ^[[BB_BODY]]:
    if (y) begin
      // CHECK: call @dummyA()
      // CHECK: cf.br ^[[BB_EXIT]]
      dummyA();
      break;
    end else begin
      // CHECK: call @dummyB()
      // CHECK: cf.br ^[[BB_STEP:.+]]
      dummyB();
      continue;
    end
    // CHECK: ^[[BB_STEP]]:
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : i32
    // CHECK: [[TMP2:%.+]] = moore.sub [[COUNT]], [[TMP1]] : i32
    // CHECK: cf.br ^[[BB_CHECK]]([[TMP2]] : !moore.i32)
  end
  // CHECK: ^[[BB_EXIT]]:
endfunction

// CHECK-LABEL: moore.module @Statements
module Statements;
  bit x, y, z;
  int i;
  initial begin
    // CHECK: %a = moore.variable : <i32>
    automatic int a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK moore.blocking_assign %i, [[TMP1]] : i32
    i = a;

    //===------------------------------------------------------------------===//
    // Assignments

    // CHECK: [[TMP1:%.+]] = moore.read %y
    // CHECK: moore.blocking_assign %x, [[TMP1]] : i1
    x = y;

    // CHECK: [[TMP1:%.+]] = moore.read %z
    // CHECK: moore.blocking_assign %y, [[TMP1]] : i1
    // CHECK: moore.blocking_assign %x, [[TMP1]] : i1
    x = (y = z);

    // CHECK: [[TMP1:%.+]] = moore.read %y
    // CHECK: moore.nonblocking_assign %x, [[TMP1]] : i1
    x <= y;
  end
endmodule

// CHECK-LABEL: moore.module @Expressions
module Expressions;
  // CHECK: %a = moore.variable : <i32>
  // CHECK: %b = moore.variable : <i32>
  // CHECK: %c = moore.variable : <i32>
  int a, b, c;
  // CHECK: %j = moore.variable : <i32>
  int j;
  // CHECK: %up = moore.variable : <uarray<4 x l11>>
  logic [10:0] up [3:0];
  // CHECK: %p1 = moore.variable : <l11>
  // CHECK: %p2 = moore.variable : <l11>
  // CHECK: %p3 = moore.variable : <l11>
  // CHECK: %p4 = moore.variable : <l11>
  logic [11:1] p1, p2, p3, p4;
  // CHECK: %yy = moore.variable : <i96>
  bit [96:1] yy;
  // CHECK: %dd = moore.variable : <i100>
  bit [99:0] dd;
  // CHECK: %u = moore.variable : <i32>
  // CHECK: %w = moore.variable : <i32>
  int unsigned u, w;
  // CHECK: %v = moore.variable : <array<2 x i4>>
  bit [1:0][3:0] v;
  // CHECK: %d = moore.variable : <l32>
  // CHECK: %e = moore.variable : <l32>
  // CHECK: %f = moore.variable : <l32>
  integer d, e, f;
  integer unsigned g, h, k;
  // CHECK: %x = moore.variable : <i1>
  bit x;
  // CHECK: %y = moore.variable : <l1>
  logic y;
  // CHECK: %vec_1 = moore.variable : <l32>
  logic [31:0] vec_1;
  // CHECK: %vec_1a = moore.variable : <l17>
  logic [31:15] vec_1a;
  // CHECK: %vec_1b = moore.variable : <l32>
  logic [-31:0] vec_1b;
  // CHECK: %vec_2 = moore.variable : <l32>
  logic [0:31] vec_2;
  // CHECK: %vec_3 = moore.variable : <l16>
  logic [15:0] vec_3;
  // CHECK: %vec_4 = moore.variable : <l32>
  logic [31:0] vec_4;
  // CHECK: %vec_5 = moore.variable : <l48>
  logic [47:0] vec_5;
  // CHECK: %arr = moore.variable : <uarray<3 x uarray<6 x i4>>>
  bit [4:1] arr [1:3][2:7];

  logic arr_1 [63:0];
  // CHECK: %struct0 = moore.variable : <struct<{a: i32, b: i32}>>
  struct packed {
    int a, b;
  } struct0;
  // CHECK: %ustruct0 = moore.variable : <ustruct<{a: i32, b: i32}>>
  struct {
    int a, b;
  } ustruct0;
  // CHECK: %struct1 = moore.variable : <struct<{c: struct<{a: i32, b: i32}>, d: struct<{a: i32, b: i32}>}>>
  struct packed {
    struct packed {
      int a, b;
    } c, d;
  } struct1;
  // CHECK: %union0 = moore.variable : <union<{a: i32, b: i32}>>
  union packed {
    int a, b;
  } union0;
  // CHECK: %union1 = moore.variable : <union<{c: union<{a: i32, b: i32}>, d: union<{a: i32, b: i32}>}>>
  union packed {
    union packed {
      int a, b;
    } c, d;
  } union1;
  // CHECK: %r1 = moore.variable : <real>
  // CHECK: %r2 = moore.variable : <real>
  real r1,r2;
  // CHECK: %arrayInt = moore.variable : <array<2 x i32>>
  bit [1:0][31:0] arrayInt;
  // CHECK: %uarrayInt = moore.variable : <uarray<2 x i32>>
  bit [31:0] uarrayInt [2];
  // CHECK: %m = moore.variable : <l4>
  logic [3:0] m;

  initial begin
    // CHECK: moore.constant 0 : i32
    c = '0;
    // CHECK: moore.constant -1 : i32
    c = '1;
    // CHECK: moore.constant hXXXXXXXX : l32
    f = 'X;
    // CHECK: moore.constant hZZZZZZZZ : l32
    f = 'Z;
    // CHECK: moore.constant 42 : i32
    c = 42;
    // CHECK: moore.constant 42 : i19
    c = 19'd42;
    // CHECK: moore.constant 42 : i19
    c = 19'sd42;
    // CHECK: moore.constant 123456789123456789123456789123456789 : i128
    c = 128'd123456789123456789123456789123456789;
    // CHECK: moore.constant h123XZ : l19
    f = 19'h123XZ;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.read %c
    // CHECK: moore.concat [[TMP1]], [[TMP2]], [[TMP3]] : (!moore.i32, !moore.i32, !moore.i32) -> i96
    a = {a, b, c};
    // CHECK: [[TMP1:%.+]] = moore.read %d
    // CHECK: [[TMP2:%.+]] = moore.read %e
    // CHECK: moore.concat [[TMP1]], [[TMP2]] : (!moore.l32, !moore.l32) -> l64
    d = {d, e};
    // CHECK: moore.concat_ref %a, %b, %c : (!moore.ref<i32>, !moore.ref<i32>, !moore.ref<i32>) -> <i96>
    {a, b, c} = a;
    // CHECK: moore.concat_ref %d, %e : (!moore.ref<l32>, !moore.ref<l32>) -> <l64>
    {d, e} = d;
    // CHECK: [[TMP1:%.+]] = moore.read %j : <i32>
    // CHECK: moore.blocking_assign %a, [[TMP1]] : i32
    a = { >> {j}};
    // CHECK: [[TMP1:%.+]] = moore.read %j : <i32>
    // CHECK: [[TMP2:%.+]] = moore.extract [[TMP1]] from 0 : i32 -> i8
    // CHECK: [[TMP3:%.+]] = moore.extract [[TMP1]] from 8 : i32 -> i8
    // CHECK: [[TMP4:%.+]] = moore.extract [[TMP1]] from 16 : i32 -> i8
    // CHECK: [[TMP5:%.+]] = moore.extract [[TMP1]] from 24 : i32 -> i8
    // CHECK: [[TMP6:%.+]] = moore.concat [[TMP2]], [[TMP3]], [[TMP4]], [[TMP5]] : (!moore.i8, !moore.i8, !moore.i8, !moore.i8) -> i32
    // CHECK: moore.blocking_assign %a, [[TMP6]] : i32
    a = { << byte {j}};
    // CHECK: [[TMP1:%.+]] = moore.read %j : <i32>
    // CHECK: [[TMP2:%.+]] = moore.extract [[TMP1]] from 0 : i32 -> i16
    // CHECK: [[TMP3:%.+]] = moore.extract [[TMP1]] from 16 : i32 -> i16
    // CHECK: [[TMP4:%.+]] = moore.concat [[TMP2]], [[TMP3]] : (!moore.i16, !moore.i16) -> i32
    // CHECK: moore.blocking_assign %a, [[TMP4]] : i32
    a = { << 16 {j}};
    // CHECK: [[TMP1:%.+]] = moore.constant 53 : i8
    // CHECK: [[TMP2:%.+]] = moore.extract [[TMP1]] from 0 : i8 -> i1
    // CHECK: [[TMP3:%.+]] = moore.extract [[TMP1]] from 1 : i8 -> i1
    // CHECK: [[TMP4:%.+]] = moore.extract [[TMP1]] from 2 : i8 -> i1
    // CHECK: [[TMP5:%.+]] = moore.extract [[TMP1]] from 3 : i8 -> i1
    // CHECK: [[TMP6:%.+]] = moore.extract [[TMP1]] from 4 : i8 -> i1
    // CHECK: [[TMP7:%.+]] = moore.extract [[TMP1]] from 5 : i8 -> i1
    // CHECK: [[TMP8:%.+]] = moore.extract [[TMP1]] from 6 : i8 -> i1
    // CHECK: [[TMP9:%.+]] = moore.extract [[TMP1]] from 7 : i8 -> i1
    // CHECK: [[TMP10:%.+]] = moore.concat [[TMP2]], [[TMP3]], [[TMP4]], [[TMP5]], [[TMP6]], [[TMP7]], [[TMP8]], [[TMP9]] : (!moore.i1, !moore.i1, !moore.i1, !moore.i1, !moore.i1, !moore.i1, !moore.i1, !moore.i1) -> i8
    // CHECK: [[TMP11:%.+]] = moore.zext [[TMP10]] : i8 -> i32
    // CHECK: moore.blocking_assign %a, [[TMP11]] : i32
    a = { << { 8'b0011_0101 }};
    // CHECK: [[TMP1:%.+]] = moore.constant -11 : i6
    // CHECK: [[TMP2:%.+]] = moore.extract [[TMP1]] from 0 : i6 -> i4
    // CHECK: [[TMP3:%.+]] = moore.extract [[TMP1]] from 4 : i6 -> i2
    // CHECK: [[TMP4:%.+]] = moore.concat [[TMP2]], [[TMP3]] : (!moore.i4, !moore.i2) -> i6
    // CHECK: [[TMP5:%.+]] = moore.zext [[TMP4]] : i6 -> i32
    // CHECK: moore.blocking_assign %a, [[TMP5]] : i32
    a = { << 4 { 6'b11_0101 }};
    // CHECK: [[TMP1:%.+]] = moore.constant -11 : i6
    // CHECK: [[TMP2:%.+]] = moore.zext [[TMP1]] : i6 -> i32
    // CHECK: moore.blocking_assign %a, [[TMP2]] : i32
    a = { >> 4 { 6'b11_0101 }};
    // CHECK: [[TMP1:%.+]] = moore.constant -3 : i4
    // CHECK: [[TMP2:%.+]] = moore.extract [[TMP1]] from 0 : i4 -> i1
    // CHECK: [[TMP3:%.+]] = moore.extract [[TMP1]] from 1 : i4 -> i1
    // CHECK: [[TMP4:%.+]] = moore.extract [[TMP1]] from 2 : i4 -> i1
    // CHECK: [[TMP5:%.+]] = moore.extract [[TMP1]] from 3 : i4 -> i1
    // CHECK: [[TMP6:%.+]] = moore.concat [[TMP2]], [[TMP3]], [[TMP4]], [[TMP5]] : (!moore.i1, !moore.i1, !moore.i1, !moore.i1) -> i4
    // CHECK: [[TMP7:%.+]] = moore.extract [[TMP6]] from 0 : i4 -> i2
    // CHECK: [[TMP8:%.+]] = moore.extract [[TMP6]] from 2 : i4 -> i2
    // CHECK: [[TMP9:%.+]] = moore.concat [[TMP7]], [[TMP8]] : (!moore.i2, !moore.i2) -> i4
    // CHECK: [[TMP10:%.+]] = moore.zext [[TMP9]] : i4 -> i32
    // CHECK: moore.blocking_assign %a, [[TMP10]] : i32
    a = { << 2 { { << { 4'b1101 }} }};
    // CHECK: [[TMP1:%.+]] = moore.read %a : <i32>
    // CHECK: [[TMP2:%.+]] = moore.read %b : <i32>
    // CHECK: [[TMP3:%.+]] = moore.read %c : <i32>
    // CHECK: [[TMP4:%.+]] = moore.concat [[TMP1]], [[TMP2]], [[TMP3]] : (!moore.i32, !moore.i32, !moore.i32) -> i96
    // CHECK: moore.blocking_assign %yy, [[TMP4]] : i96
    yy = {>>{ a, b, c }};
    // CHECK: [[TMP1:%.+]] = moore.read %a : <i32>
    // CHECK: [[TMP2:%.+]] = moore.read %b : <i32>
    // CHECK: [[TMP3:%.+]] = moore.read %c : <i32>
    // CHECK: [[TMP4:%.+]] = moore.concat [[TMP1]], [[TMP2]], [[TMP3]] : (!moore.i32, !moore.i32, !moore.i32) -> i96
    // CHECK: [[TMP5:%.+]] = moore.zext [[TMP4]] : i96 -> i100
    // CHECK: moore.blocking_assign %dd, [[TMP5]] : i100
    dd = {>>{ a, b, c }};
    // CHECK: [[TMP1:%.+]] = moore.concat_ref %a, %b, %c : (!moore.ref<i32>, !moore.ref<i32>, !moore.ref<i32>) -> <i96>
    // CHECK: [[TMP2:%.+]] = moore.constant 1 : i96
    // CHECK: moore.blocking_assign [[TMP1]], [[TMP2]] : i96
    {>>{ a, b, c }} = 96'b1;
    // CHECK: [[TMP1:%.+]] = moore.concat_ref %a, %b, %c : (!moore.ref<i32>, !moore.ref<i32>, !moore.ref<i32>) -> <i96>
    // CHECK: [[TMP2:%.+]] = moore.constant 31 : i100
    // CHECK: [[TMP3:%.+]] = moore.trunc [[TMP2]] : i100 -> i96
    // CHECK: moore.blocking_assign [[TMP1]], [[TMP3]] : i96
    {>>{ a, b, c }} = 100'b11111;
    // CHECK: [[TMP1:%.+]] = moore.concat_ref %p1, %p2, %p3, %p4 : (!moore.ref<l11>, !moore.ref<l11>, !moore.ref<l11>, !moore.ref<l11>) -> <l44>
    // CHECK: [[TMP2:%.+]] = moore.read %up : <uarray<4 x l11>>
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.uarray<4 x l11> -> !moore.l44
    // CHECK: moore.blocking_assign [[TMP1]], [[TMP3]] : l44
    { >> {p1, p2, p3, p4}} = up;
    // CHECK: [[TMP1:%.+]] = moore.extract_ref %a from 0 : <i32> -> <i8>
    // CHECK: [[TMP2:%.+]] = moore.extract_ref %a from 8 : <i32> -> <i8>
    // CHECK: [[TMP3:%.+]] = moore.extract_ref %a from 16 : <i32> -> <i8>
    // CHECK: [[TMP4:%.+]] = moore.extract_ref %a from 24 : <i32> -> <i8>
    // CHECK: [[TMP5:%.+]] = moore.concat_ref [[TMP1]], [[TMP2]], [[TMP3]], [[TMP4]] : (!moore.ref<i8>, !moore.ref<i8>, !moore.ref<i8>, !moore.ref<i8>) -> <i32>
    // CHECK: [[TMP6:%.+]] = moore.constant 1 : i32
    // CHECK: moore.blocking_assign [[TMP5]], [[TMP6]] : i32
    {<< byte {a}} = 32'b1;
    // CHECK: %[[TMP1:.*]] = moore.read %vec_3 : <l16>
    // CHECK: %[[TMP2:.*]] = moore.read %arr_1 : <uarray<64 x l1>>
    // CHECK: %[[TMP3:.*]] = moore.extract %[[TMP2]] from 0 : uarray<64 x l1> -> uarray<16 x l1>
    // CHECK: %[[TMP4:.*]] = moore.conversion %[[TMP3]] : !moore.uarray<16 x l1> -> !moore.l16
    // CHECK: %[[TMP5:.*]] = moore.concat %[[TMP1]], %[[TMP4]] : (!moore.l16, !moore.l16) -> l32
    // CHECK: %[[TMP6:.*]] = moore.extract %[[TMP5]] from 0 : l32 -> l8
    // CHECK: %[[TMP7:.*]] = moore.extract %[[TMP5]] from 8 : l32 -> l8
    // CHECK: %[[TMP8:.*]] = moore.extract %[[TMP5]] from 16 : l32 -> l8
    // CHECK: %[[TMP9:.*]] = moore.extract %[[TMP5]] from 24 : l32 -> l8
    // CHECK: %[[TMP10:.*]] = moore.concat %[[TMP6]], %[[TMP7]], %[[TMP8]], %[[TMP9]] : (!moore.l8, !moore.l8, !moore.l8, !moore.l8) -> l32
    // CHECK: moore.blocking_assign %vec_1, %[[TMP10]] : l32
    vec_1 = {<<byte{vec_3, arr_1 with [15:0]}};
    // CHECK: %[[TMP1:.*]] = moore.extract_ref %arr_1 from 0 : <uarray<64 x l1>> -> <uarray<16 x l1>>
    // CHECK: %[[TMP2:.*]] = moore.conversion %[[TMP1]] : !moore.ref<uarray<16 x l1>> -> !moore.ref<l16>
    // CHECK: %[[TMP3:.*]] = moore.concat_ref %vec_3, %[[TMP2]] : (!moore.ref<l16>, !moore.ref<l16>) -> <l32>
    // CHECK: %[[TMP4:.*]] = moore.extract_ref %[[TMP3]] from 0 : <l32> -> <l8>
    // CHECK: %[[TMP5:.*]] = moore.extract_ref %[[TMP3]] from 8 : <l32> -> <l8>
    // CHECK: %[[TMP6:.*]] = moore.extract_ref %[[TMP3]] from 16 : <l32> -> <l8>
    // CHECK: %[[TMP7:.*]] = moore.extract_ref %[[TMP3]] from 24 : <l32> -> <l8>
    // CHECK: %[[TMP8:.*]] = moore.concat_ref %[[TMP4]], %[[TMP5]], %[[TMP6]], %[[TMP7]] : (!moore.ref<l8>, !moore.ref<l8>, !moore.ref<l8>, !moore.ref<l8>) -> <l32>
    // CHECK: %[[TMP9:.*]] = moore.read %vec_1 : <l32>
    // CHECK: moore.blocking_assign %[[TMP8]], %[[TMP9]] : l32
    {<<byte{vec_3, arr_1 with [15:0]}} = vec_1;
    // CHECK: [[TMP1:%.+]] = moore.constant 0 : i1
    // CHECK: [[TMP2:%.+]] = moore.concat [[TMP1]] : (!moore.i1) -> i1
    // CHECK: moore.replicate [[TMP2]] : i1 -> i32
    a = {32{1'b0}};
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : <l32>
    // CHECK: moore.extract [[TMP1]] from 1 : l32 -> l3
    y = vec_1[3:1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_2 : <l32>
    // CHECK: moore.extract [[TMP1]] from 29 : l32 -> l2
    y = vec_2[2:3];
    // CHECK: [[TMP1:%.+]] = moore.read %d : <l32>
    // CHECK: [[TMP2:%.+]] = moore.read %x : <i1>
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP2]] : l32, i1 -> l1
    y = d[x];
    // CHECK: [[TMP1:%.+]] = moore.read %a : <i32>
    // CHECK: [[TMP2:%.+]] = moore.read %x : <i1>
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP2]] : i32, i1 -> i1
    x = a[x];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : <l32>
    // CHECK: moore.extract [[TMP1]] from 15 : l32 -> l1
    y = vec_1[15];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : <l32>
    // CHECK: moore.extract [[TMP1]] from 15 : l32 -> l1
    y = vec_1[15+:1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_2 : <l32>
    // CHECK: moore.extract [[TMP1]] from 31 : l32 -> l1
    y = vec_2[0+:1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_2 : <l32> 
    // CHECK: [[TMP2:%.+]] = moore.read %vec_1 : <l32> 
    // CHECK: [[TMP3:%.+]] = moore.constant 31 : l32 
    // CHECK: [[TMP4:%.+]] = moore.sub [[TMP3]], [[TMP2]] : l32 
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP4]] : l32, l32 -> l2
    y = vec_2[vec_1+:2];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_2 : <l32> 
    // CHECK: [[TMP2:%.+]] = moore.read %vec_1 : <l32> 
    // CHECK: [[TMP3:%.+]] = moore.constant 31 : l32 
    // CHECK: [[TMP4:%.+]] = moore.sub [[TMP3]], [[TMP2]] : l32 
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP4]] : l32, l32 -> l1
    y = vec_2[vec_1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1a : <l17> 
    // CHECK: [[TMP2:%.+]] = moore.read %vec_1 : <l32> 
    // CHECK: [[TMP3:%.+]] = moore.constant 15 : l32 
    // CHECK: [[TMP4:%.+]] = moore.sub [[TMP2]], [[TMP3]] : l32 
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP4]] : l17, l32 -> l2
    y = vec_1a[vec_1+:2];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1a : <l17> 
    // CHECK: [[TMP2:%.+]] = moore.read %vec_1 : <l32> 
    // CHECK: [[TMP3:%.+]] = moore.constant 15 : l32 
    // CHECK: [[TMP4:%.+]] = moore.sub [[TMP2]], [[TMP3]] : l32 
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP4]] : l17, l32 -> l1
    y = vec_1a[vec_1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1b : <l32> 
    // CHECK: [[TMP2:%.+]] = moore.read %vec_1 : <l32> 
    // CHECK: [[TMP3:%.+]] = moore.neg [[TMP2]] : l32 
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP3]] : l32, l32 -> l2
    y = vec_1b[vec_1+:2];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1b : <l32> 
    // CHECK: [[TMP2:%.+]] = moore.read %vec_1 : <l32> 
    // CHECK: [[TMP3:%.+]] = moore.neg [[TMP2]] : l32 
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP3]] : l32, l32 -> l1
    y = vec_1b[vec_1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1a : <l17> 
    // CHECK: [[TMP2:%.+]] = moore.read %x : <i1> 
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i1 -> !moore.i5
    // CHECK: [[TMP4:%.+]] = moore.constant 15 : i5
    // CHECK: [[TMP5:%.+]] = moore.sub [[TMP3]], [[TMP4]] : i5
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP5]] : l17, i5 -> l2
    y = vec_1a[x+:2];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1a : <l17> 
    // CHECK: [[TMP2:%.+]] = moore.read %x : <i1> 
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i1 -> !moore.i5
    // CHECK: [[TMP4:%.+]] = moore.constant 15 : i5
    // CHECK: [[TMP5:%.+]] = moore.sub [[TMP3]], [[TMP4]] : i5
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP5]] : l17, i5 -> l1
    y = vec_1a[x];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1
    // CHECK: [[TMP2:%.+]] = moore.constant 1 : i32
    // CHECK: [[TMP3:%.+]] = moore.read %a
    // CHECK: [[TMP4:%.+]] = moore.mul [[TMP2]], [[TMP3]] : i32
    // CHECK: [[TMP5:%.+]] = moore.constant 0 : i32
    // CHECK: [[TMP6:%.+]] = moore.sub [[TMP4]], [[TMP5]] : i32
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP6]] : l32, i32 -> l1
    c = vec_1[1*a-:1];
    // CHECK: [[TMP1:%.+]] = moore.read %arr : <uarray<3 x uarray<6 x i4>>>
    // CHECK: [[TMP3:%.+]] = moore.extract [[TMP1]] from 0 : uarray<3 x uarray<6 x i4>> -> uarray<6 x i4>
    // CHECK: [[TMP5:%.+]] = moore.extract [[TMP3]] from 0 : uarray<6 x i4> -> i4
    // CHECK: moore.extract [[TMP5]] from 2 : i4 -> i2
    c = arr[3][7][4:3];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : <l32>
    // CHECK: [[TMP2:%.+]] = moore.read %c : <i32>
    // CHECK: moore.dyn_extract [[TMP1]] from [[TMP2]] : l32, i32 -> l1
    y = vec_1[c];

    // CHECK: [[TMP2:%.+]] = moore.extract_ref %v from 1 : <array<2 x i4>> -> <i4>
    // CHECK: moore.extract_ref [[TMP2]] from 3 : <i4> -> <i1>
    v[1][3] = x;

    // CHECK: moore.extract_ref %vec_1 from 1 : <l32> -> <l2>
    vec_1[2:1] = y;

    // CHECK: [[X_READ:%.+]] = moore.read %x : <i1>
    // CHECK: moore.dyn_extract_ref %vec_1 from [[X_READ]] : <l32>, i1 -> <l1>
    vec_1[x] = y;
    
    // CHECK: moore.extract_ref %vec_1 from 11 : <l32> -> <l3>
    vec_1[15-:3] = y;

    //===------------------------------------------------------------------===//
    // Unary operators

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: moore.blocking_assign %c, [[TMP1]] : i32
    c = +a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: moore.neg [[TMP1]] : i32
    c = -a;
    // CHECK: [[TMP1:%.+]] = moore.read %v
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.array<2 x i4> -> !moore.i8
    // CHECK: [[TMP3:%.+]] = moore.zext [[TMP2]] : i8 -> i32
    // CHECK: moore.neg [[TMP3]] : i32
    c = -v;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: moore.not [[TMP1]] : i32
    c = ~a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: moore.reduce_and [[TMP1]] : i32 -> i1
    x = &a;
    // CHECK: [[TMP1:%.+]] = moore.read %d
    // CHECK: moore.reduce_and [[TMP1]] : l32 -> l1
    y = &d;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: moore.reduce_or [[TMP1]] : i32 -> i1
    x = |a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: moore.reduce_xor [[TMP1]] : i32 -> i1
    x = ^a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.reduce_and [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = ~&a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.reduce_or [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = ~|a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.reduce_xor [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = ~^a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.reduce_xor [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = ^~a;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = !a;
    // CHECK: [[PRE:%.+]] = moore.read %a
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.add [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[PRE]]
    c = a++;
    // CHECK: [[PRE:%.+]] = moore.read %a
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.sub [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[PRE]]
    c = a--;
    // CHECK: [[PRE:%.+]] = moore.read %a
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.add [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[POST]]
    c = ++a;
    // CHECK: [[PRE:%.+]] = moore.read %a
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.sub [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[POST]]
    c = --a;

    //===------------------------------------------------------------------===//
    // Binary operators

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.add [[TMP1]], [[TMP2]] : i32
    c = a + b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %v
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.array<2 x i4> -> !moore.i8
    // CHECK: [[TMP4:%.+]] = moore.zext [[TMP3]] : i8 -> i32
    // CHECK: moore.add [[TMP1]], [[TMP4]] : i32
    c = a + v;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.sub [[TMP1]], [[TMP2]] : i32
    c = a - b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.mul [[TMP1]], [[TMP2]] : i32
    c = a * b;
    // CHECK: [[TMP1:%.+]] = moore.read %h
    // CHECK: [[TMP2:%.+]] = moore.read %k
    // CHECK: moore.divu [[TMP1]], [[TMP2]] : l32
    g = h / k;
    // CHECK: [[TMP1:%.+]] = moore.read %d
    // CHECK: [[TMP2:%.+]] = moore.read %e
    // CHECK: moore.divs [[TMP1]], [[TMP2]] : l32
    f = d / e;
    // CHECK: [[TMP1:%.+]] = moore.read %h
    // CHECK: [[TMP2:%.+]] = moore.read %k
    // CHECK: moore.modu [[TMP1]], [[TMP2]] : l32
    g = h % k;
    // CHECK: [[TMP1:%.+]] = moore.read %d
    // CHECK: [[TMP2:%.+]] = moore.read %e
    // CHECK: moore.mods [[TMP1]], [[TMP2]] : l32
    f = d % e;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.l32
    // CHECK: [[TMP1:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.l32
    // CHECK: [[TMP1:%.+]] = moore.pows [[TMP2]], [[TMP3]] : l32
    // CHECK: moore.conversion [[TMP1]] : !moore.l32 -> !moore.i32
    c = a ** b;
    // CHECK: [[TMP1:%.+]] = moore.read %u
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.l32
    // CHECK: [[TMP1:%.+]] = moore.read %w
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.l32
    // CHECK: [[TMP1:%.+]] = moore.powu [[TMP2]], [[TMP3]] : l32
    // CHECK: moore.conversion [[TMP1]] : !moore.l32 -> !moore.i32
    u = u ** w;

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.and [[TMP1]], [[TMP2]] : i32
    c = a & b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.or [[TMP1]], [[TMP2]] : i32
    c = a | b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.xor [[TMP1]], [[TMP2]] : i32
    c = a ^ b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.xor [[TMP1]], [[TMP2]] : i32
    // CHECK: moore.not [[TMP3]] : i32
    c = a ~^ b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.xor [[TMP1]], [[TMP2]] : i32
    // CHECK: moore.not [[TMP3]] : i32
    c = a ^~ b;

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.eq [[TMP1]], [[TMP2]] : i32 -> i1
    x = a == b;
    // CHECK: [[TMP1:%.+]] = moore.read %d
    // CHECK: [[TMP2:%.+]] = moore.read %e
    // CHECK: moore.eq [[TMP1]], [[TMP2]] : l32 -> l1
    y = d == e;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.ne [[TMP1]], [[TMP2]] : i32 -> i1
    x = a != b ;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.case_eq [[TMP1]], [[TMP2]] : i32
    x = a === b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.case_ne [[TMP1]], [[TMP2]] : i32
    x = a !== b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    x = a ==? b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.l32
    // CHECK: [[TMP3:%.+]] = moore.read %d
    // CHECK: moore.wildcard_eq [[TMP2]], [[TMP3]] : l32 -> l1
    y = a ==? d;
    // CHECK: [[TMP1:%.+]] = moore.read %d
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i32 -> !moore.l32
    // CHECK: moore.wildcard_eq [[TMP1]], [[TMP3]] : l32 -> l1
    y = d ==? b;
    // CHECK: [[TMP1:%.+]] = moore.read %d
    // CHECK: [[TMP2:%.+]] = moore.read %e
    // CHECK: moore.wildcard_eq [[TMP1]], [[TMP2]] : l32 -> l1
    y = d ==? e;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.wildcard_ne [[TMP1]], [[TMP2]] : i32 -> i1
    x = a !=? b;

    // CHECK: [[TMP1:%.+]] = moore.read %u
    // CHECK: [[TMP2:%.+]] = moore.read %w
    // CHECK: moore.uge [[TMP1]], [[TMP2]] : i32 -> i1
    c = u >= w;
    // CHECK: [[TMP1:%.+]] = moore.read %u
    // CHECK: [[TMP2:%.+]] = moore.read %w
    // CHECK: moore.ugt [[TMP1]], [[TMP2]] : i32 -> i1
    c = u > w;
    // CHECK: [[TMP1:%.+]] = moore.read %u
    // CHECK: [[TMP2:%.+]] = moore.read %w
    // CHECK: moore.ule [[TMP1]], [[TMP2]] : i32 -> i1
    c = u <= w;
    // CHECK: [[TMP1:%.+]] = moore.read %u
    // CHECK: [[TMP2:%.+]] = moore.read %w
    // CHECK: moore.ult [[TMP1]], [[TMP2]] : i32 -> i1
    c = u < w;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.sge [[TMP1]], [[TMP2]] : i32 -> i1
    c = a >= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.sgt [[TMP1]], [[TMP2]] : i32 -> i1
    c = a > b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.sle [[TMP1]], [[TMP2]] : i32 -> i1
    c = a <= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.slt [[TMP1]], [[TMP2]] : i32 -> i1
    c = a < b;

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[A:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
    // CHECK: moore.and [[A]], [[B]] : i1
    c = a && b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[A:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
    // CHECK: moore.or [[A]], [[B]] : i1
    c = a || b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[A:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : i1
    // CHECK: moore.or [[NOT_A]], [[B]] : i1
    c = a -> b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[A:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : i1
    // CHECK: [[NOT_B:%.+]] = moore.not [[B]] : i1
    // CHECK: [[BOTH:%.+]] = moore.and [[A]], [[B]] : i1
    // CHECK: [[NOT_BOTH:%.+]] = moore.and [[NOT_A]], [[NOT_B]] : i1
    // CHECK: moore.or [[BOTH]], [[NOT_BOTH]] : i1
    c = a <-> b;
    // CHECK: [[TMP:%.+]] = moore.read %x : <i1>
    // CHECK: [[Y:%.+]] = moore.read %y : <l1>
    // CHECK: [[X:%.+]] = moore.conversion [[TMP]] : !moore.i1 -> !moore.l1
    // CHECK: moore.and [[X]], [[Y]] : l1
    y = x && y;
    // CHECK: [[Y:%.+]] = moore.read %y : <l1>
    // CHECK: [[TMP:%.+]] = moore.read %x : <i1>
    // CHECK: [[X:%.+]] = moore.conversion [[TMP]] : !moore.i1 -> !moore.l1
    // CHECK: moore.and [[Y]], [[X]] : l1
    y = y && x;

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.shl [[TMP1]], [[TMP2]] : i32, i32
    c = a << b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.shr [[TMP1]], [[TMP2]] : i32, i32
    c = a >> b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.shl [[TMP1]], [[TMP2]] : i32, i32
    c = a <<< b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.ashr [[TMP1]], [[TMP2]] : i32, i32
    c = a >>> b;
    // CHECK: [[TMP1:%.+]] = moore.read %u
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: moore.shr [[TMP1]], [[TMP2]] : i32, i32
    c = u >>> b;

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %a
    // CHECK: moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    c = a inside { a };

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %a
    // CHECK: [[TMP3:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    // CHECK: [[TMP4:%.+]] = moore.read %b
    // CHECK: [[TMP5:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP4]] : i32 -> i1
    // CHECK: moore.or [[TMP3]], [[TMP5]] : i1
    c = a inside { a, b };

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %a
    // CHECK: [[TMP3:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    // CHECK: [[TMP4:%.+]] = moore.read %b
    // CHECK: [[TMP5:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP4]] : i32 -> i1
    // CHECK: [[TMP6:%.+]] = moore.read %a
    // CHECK: [[TMP7:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP6]] : i32 -> i1
    // CHECK: [[TMP8:%.+]] = moore.read %b
    // CHECK: [[TMP9:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP8]] : i32 -> i1
    // CHECK: [[TMP10:%.+]] = moore.or [[TMP7]], [[TMP9]] : i1
    // CHECK: [[TMP11:%.+]] = moore.or [[TMP5]], [[TMP10]] : i1
    // CHECK: moore.or [[TMP3]], [[TMP11]] : i1
    c = a inside { a, b, a, b };

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %a
    // CHECK: [[TMP3:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    // CHECK: [[TMP4:%.+]] = moore.read %b
    // CHECK: [[TMP5:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP4]] : i32 -> i1
    // CHECK: [[TMP6:%.+]] = moore.read %a
    // CHECK: [[TMP7:%.+]] = moore.read %b
    // CHECK: [[TMP8:%.+]] = moore.sge [[TMP1]], [[TMP6]] : i32 -> i1
    // CHECK: [[TMP9:%.+]] = moore.sle [[TMP1]], [[TMP7]] : i32 -> i1
    // CHECK: [[TMP10:%.+]] = moore.and [[TMP8]], [[TMP9]] : i1
    // CHECK: [[TMP11:%.+]] = moore.or [[TMP5]], [[TMP10]] : i1
    // CHECK: moore.or [[TMP3]], [[TMP11]] : i1
    c = a inside { a, b, [a:b] };

    //===------------------------------------------------------------------===//
    // Conditional operator

    // CHECK: [[X_COND:%.+]] = moore.read %x
    // CHECK: moore.conditional [[X_COND]] : i1 -> i32 {
    // CHECK:   [[A_READ:%.+]] = moore.read %a
    // CHECK:   moore.yield [[A_READ]] : i32
    // CHECK: } {
    // CHECK:   [[B_READ:%.+]] = moore.read %b
    // CHECK:   moore.yield [[B_READ]] : i32
    // CHECK: }
    c = x ? a : b;

    // CHECK: [[X_COND:%.+]] = moore.read %x
    // CHECK: moore.conditional [[X_COND]] : i1 -> real {
    // CHECK:   [[R1_READ:%.+]] = moore.read %r1
    // CHECK:   moore.yield [[R1_READ]] : real
    // CHECK: } {
    // CHECK:   [[R2_READ:%.+]] = moore.read %r2
    // CHECK:   moore.yield [[R2_READ]] : real
    // CHECK: }
    r1 = x ? r1 : r2;

    // CHECK: [[A_COND:%.+]] = moore.read %a
    // CHECK: [[TMP1:%.+]] = moore.bool_cast [[A_COND]] : i32 -> i1
    // CHECK: moore.conditional [[TMP1]] : i1 -> i32 {
    // CHECK:   [[A_READ:%.+]] = moore.read %a
    // CHECK:   moore.yield [[A_READ]] : i32
    // CHECK: } {
    // CHECK:   [[B_READ:%.+]] = moore.read %b
    // CHECK:   moore.yield [[B_READ]] : i32
    // CHECK: }
    c = a ? a : b;

    // CHECK: [[A_SGT:%.+]] = moore.read %a
    // CHECK: [[B_SGT:%.+]] = moore.read %b
    // CHECK: [[TMP1:%.+]] = moore.sgt [[A_SGT]], [[B_SGT]] : i32 -> i1
    // CHECK: moore.conditional [[TMP1]] : i1 -> i32 {
    // CHECK:   [[A_ADD:%.+]] = moore.read %a
    // CHECK:   [[B_ADD:%.+]] = moore.read %b
    // CHECK:   [[TMP2:%.+]] = moore.add [[A_ADD]], [[B_ADD]] : i32
    // CHECK:   moore.yield [[TMP2]] : i32
    // CHECK: } {
    // CHECK:   [[A_SUB:%.+]] = moore.read %a
    // CHECK:   [[B_SUB:%.+]] = moore.read %b
    // CHECK:   [[TMP2:%.+]] = moore.sub [[A_SUB]], [[B_SUB]] : i32
    // CHECK:   moore.yield [[TMP2]] : i32
    // CHECK: }
    c = (a > b) ? (a + b) : (a - b);

    //===------------------------------------------------------------------===//
    // Assign operators

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.add [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a += b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.sub [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a -= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.mul [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a *= b;
    // CHECK: [[TMP1:%.+]] = moore.read %f
    // CHECK: [[TMP2:%.+]] = moore.read %d
    // CHECK: [[TMP3:%.+]] = moore.divs [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %f, [[TMP3]]
    f /= d;
    // CHECK: [[TMP1:%.+]] = moore.read %g
    // CHECK: [[TMP2:%.+]] = moore.read %h
    // CHECK: [[TMP3:%.+]] = moore.divu [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %g, [[TMP3]]
    g /= h;
    // CHECK: [[TMP1:%.+]] = moore.read %f
    // CHECK: [[TMP2:%.+]] = moore.read %d
    // CHECK: [[TMP3:%.+]] = moore.mods [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %f, [[TMP3]]
    f %= d;
    // CHECK: [[TMP1:%.+]] = moore.read %g
    // CHECK: [[TMP2:%.+]] = moore.read %h
    // CHECK: [[TMP3:%.+]] = moore.modu [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %g, [[TMP3]]
    g %= h;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.and [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a &= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.or [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a |= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.xor [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a ^= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.shl [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a <<= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.shl [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a <<<= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.shr [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a >>= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.ashr [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a >>>= b;

    // CHECK: [[A_ADD:%.+]] = moore.read %a
    // CHECK: [[A_MUL:%.+]] = moore.read %a
    // CHECK: [[A_DEC:%.+]] = moore.read %a
    // CHECK: [[TMP1:%.+]] = moore.constant 1
    // CHECK: [[TMP2:%.+]] = moore.sub [[A_DEC]], [[TMP1]]
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    // CHECK: [[TMP1:%.+]] = moore.mul [[A_MUL]], [[A_DEC]]
    // CHECK: moore.blocking_assign %a, [[TMP1]]
    // CHECK: [[TMP2:%.+]] = moore.add [[A_ADD]], [[TMP1]]
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a += (a *= a--);

    // CHECK: [[TMP1:%.+]] = moore.struct_extract_ref %struct0, "a" : <struct<{a: i32, b: i32}>> -> <i32>
    // CHECK: [[TMP2:%.+]] = moore.read %a
    // CHECK: moore.blocking_assign [[TMP1]], [[TMP2]]
    struct0.a = a;

    // CHECK: [[TMP1:%.+]] = moore.read %struct0
    // CHECK: [[TMP2:%.+]] = moore.struct_extract [[TMP1]], "b" : struct<{a: i32, b: i32}> -> i32
    // CHECK: moore.blocking_assign %b, [[TMP2]]
    b = struct0.b;

    //===------------------------------------------------------------------===//
    // Assignment Patterns

    // CHECK: [[TMP0:%.+]] = moore.constant 17
    // CHECK: [[TMP1:%.+]] = moore.constant 17
    // CHECK: moore.struct_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
    struct0 = '{default: 17};

    // CHECK: [[TMP0:%.+]] = moore.constant 1337
    // CHECK: [[TMP1:%.+]] = moore.constant 1337
    // CHECK: moore.struct_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
    struct0 = '{int: 1337};

    // CHECK: [[TMP0:%.+]] = moore.constant 420
    // CHECK: moore.struct_create [[TMP0]], [[TMP0]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
    struct0 = '{2{420}};

    // CHECK: [[TMP0:%.+]] = moore.constant 42
    // CHECK: [[TMP1:%.+]] = moore.constant 9001
    // CHECK: moore.struct_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
    struct0 = '{a: 42, b: 9001};

    // CHECK: [[TMP0:%.+]] = moore.constant 43
    // CHECK: [[TMP1:%.+]] = moore.constant 9002
    // CHECK: moore.struct_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
    struct0 = '{43, 9002};

    // CHECK: [[TMP0:%.+]] = moore.constant 44
    // CHECK: [[TMP1:%.+]] = moore.constant 9003
    // CHECK: moore.struct_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> ustruct<{a: i32, b: i32}>
    ustruct0 = '{44, 9003};

    // CHECK: [[TMP0:%.+]] = moore.constant 1
    // CHECK: [[TMP1:%.+]] = moore.constant 2
    // CHECK: [[TMP2:%.+]] = moore.struct_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
    // CHECK: [[TMP0:%.+]] = moore.constant 3
    // CHECK: [[TMP1:%.+]] = moore.constant 4
    // CHECK: [[TMP3:%.+]] = moore.struct_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
    // CHECK: moore.struct_create [[TMP2]], [[TMP3]] : !moore.struct<{a: i32, b: i32}>, !moore.struct<{a: i32, b: i32}> -> struct<{c: struct<{a: i32, b: i32}>, d: struct<{a: i32, b: i32}>}>
    struct1 = '{c: '{a: 1, b: 2}, d: '{a: 3, b: 4}};

    // CHECK: [[TMP0:%.+]] = moore.constant 42
    // CHECK: [[TMP1:%.+]] = moore.constant 9001
    // CHECK: moore.array_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> array<2 x i32>
    arrayInt = '{42, 9001};

    // CHECK: [[TMP0:%.+]] = moore.constant 43
    // CHECK: [[TMP1:%.+]] = moore.constant 9002
    // CHECK: moore.array_create [[TMP0]], [[TMP1]] : !moore.i32, !moore.i32 -> uarray<2 x i32>
    uarrayInt = '{43, 9002};

    // CHECK: [[TMP0:%.+]] = moore.constant 1
    // CHECK: [[TMP1:%.+]] = moore.constant 2
    // CHECK: [[TMP2:%.+]] = moore.constant 3
    // CHECK: [[TMP3:%.+]] = moore.array_create [[TMP0]], [[TMP1]], [[TMP2]], [[TMP0]], [[TMP1]], [[TMP2]] : !moore.i4, !moore.i4, !moore.i4, !moore.i4, !moore.i4, !moore.i4 -> uarray<6 x i4>
    // CHECK: moore.array_create [[TMP3]], [[TMP3]], [[TMP3]] : !moore.uarray<6 x i4>, !moore.uarray<6 x i4>, !moore.uarray<6 x i4> -> uarray<3 x uarray<6 x i4>>
    arr = '{3{'{2{4'd1, 4'd2, 4'd3}}}};

    // CHECK: [[TMP0:%.+]] = moore.constant 0 :
    // CHECK: [[TMP1:%.+]] = moore.constant 0 :
    // CHECK: [[TMP2:%.+]] = moore.constant 1 :
    // CHECK: [[TMP3:%.+]] = moore.constant 0 :
    // CHECK: moore.concat [[TMP3]], [[TMP2]], [[TMP1]], [[TMP0]] : (!moore.l1, !moore.l1, !moore.l1, !moore.l1) -> l4
    m = '{default: '0, 2: '1};
 
    //===------------------------------------------------------------------===//
    // Builtin Functions

    // The following functions are handled by Slang's type checking and don't
    // convert into any IR operations.

    // CHECK: [[TMP:%.+]] = moore.read %u
    // CHECK: moore.blocking_assign %a, [[TMP]]
    a = $signed(u);
    // CHECK: [[TMP:%.+]] = moore.read %a
    // CHECK: moore.blocking_assign %u, [[TMP]]
    u = $unsigned(a);
  end
endmodule

// CHECK-LABEL: moore.module @Conversion
module Conversion;
  // Implicit conversion.
  // CHECK: %a = moore.variable
  // CHECK: [[TMP1:%.+]] = moore.read %a
  // CHECK: [[TMP2:%.+]] = moore.sext [[TMP1]] : i16 -> i32
  // CHECK: %b = moore.variable [[TMP2]]
  shortint a;
  int b = a;

  // Explicit conversion.
  // CHECK: [[TMP1:%.+]] = moore.read %a
  // CHECK: [[TMP2:%.+]] = moore.trunc [[TMP1]] : i16 -> i8
  // CHECK: [[TMP3:%.+]] = moore.sext [[TMP2]] : i8 -> i32
  // CHECK: %c = moore.variable [[TMP3]]
  int c = byte'(a);

  // Sign conversion.
  // CHECK: [[TMP1:%.+]] = moore.read %b
  // CHECK: %d1 = moore.variable [[TMP1]]
  // CHECK: [[TMP2:%.+]] = moore.read %b
  // CHECK: %d2 = moore.variable [[TMP2]]
  bit signed [31:0] d1 = signed'(b);
  bit [31:0] d2 = unsigned'(b);

  // Width conversion.
  // CHECK: [[TMP1:%.+]] = moore.read %b
  // CHECK: [[TMP2:%.+]] = moore.trunc [[TMP1]] : i32 -> i19
  // CHECK: %e = moore.variable [[TMP2]]
  bit signed [18:0] e = 19'(b);

  // Implicit conversion for literals.
  // CHECK: [[TMP1:%.+]] = moore.constant 0 : i64
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i64 -> !moore.struct<{a: i32, b: i32}>
  // CHECK: %f = moore.variable [[TMP2]]
  struct packed { int a; int b; } f = '0;
endmodule

// CHECK-LABEL: moore.module @PortsTop
module PortsTop;
  wire x0, y0, z0;
  logic w0;
  // CHECK: [[x0:%.+]] = moore.read %x0
  // CHECK: [[B:%.+]] = moore.instance "p0" @PortsAnsi(
  // CHECK-SAME:   a: [[x0]]: !moore.l1
  // CHECK-SAME:   c: %z0: !moore.ref<l1>
  // CHECK-SAME:   d: %w0: !moore.ref<l1>
  // CHECK-SAME: ) -> (b: !moore.l1)
  // CHECK-NEXT: moore.assign %y0, [[B]]
  PortsAnsi p0(x0, y0, z0, w0);

  wire x1, y1, z1;
  logic w1;
  // CHECK: [[x1:%.+]] = moore.read %x1
  // CHECK: [[B:%.+]] = moore.instance "p1" @PortsNonAnsi(
  // CHECK-SAME:   a: [[x1]]: !moore.l1
  // CHECK-SAME:   c: %z1: !moore.ref<l1>
  // CHECK-SAME:   d: %w1: !moore.ref<l1>
  // CHECK-SAME: ) -> (b: !moore.l1)
  // CHECK-NEXT: moore.assign %y1, [[B]]
  PortsNonAnsi p1(x1, y1, z1, w1);

  wire x2;
  wire [1:0] y2;
  int z2;
  wire w2, v2;
  // CHECK: [[X2:%.+]] = moore.read %x2
  // CHECK: [[Y2:%.+]] = moore.read %y2
  // CHECK: [[B0:%.+]], [[B1:%.+]], [[B2:%.+]] = moore.instance "p2" @PortsExplicit(
  // CHECK-SAME:   a0: [[X2]]: !moore.l1
  // CHECK-SAME:   a1: [[Y2]]: !moore.l2
  // CHECK-SAME: ) -> (
  // CHECK-SAME:   b0: !moore.i32
  // CHECK-SAME:   b1: !moore.l1
  // CHECK-SAME:   b2: !moore.l1
  // CHECK-SAME: )
  // CHECK-NEXT: moore.assign %z2, [[B0]]
  // CHECK-NEXT: moore.assign %w2, [[B1]]
  // CHECK-NEXT: moore.assign %v2, [[B2]]
  PortsExplicit p2(x2, y2, z2, w2, v2);

  wire x3, y3;
  wire [2:0] z3;
  wire [1:0] w3;
  // CHECK: [[X3:%.+]] = moore.read %x3
  // CHECK: [[Y3:%.+]] = moore.read %y3
  // CHECK: [[V2:%.+]] = moore.extract_ref %z3 from 0
  // CHECK: [[V1:%.+]] = moore.extract_ref %z3 from 1
  // CHECK: [[V0:%.+]] = moore.extract_ref %z3 from 2
  // CHECK: [[V0_READ:%.+]] = moore.read [[V0]]
  // CHECK: [[C1:%.+]] = moore.extract_ref %w3 from 0
  // CHECK: [[C0:%.+]] = moore.extract_ref %w3 from 1
  // CHECK: [[C0_READ:%.+]] = moore.read [[C0]]
  // CHECK: [[V1_VALUE:%.+]], [[C1_VALUE:%.+]] = moore.instance "p3" @MultiPorts(
  // CHECK-SAME:   a0: [[X3]]: !moore.l1
  // CHECK-SAME:   a1: [[Y3]]: !moore.l1
  // CHECK-SAME:   v0: [[V0_READ]]: !moore.l1
  // CHECK-SAME:   v2: [[V2]]: !moore.ref<l1>
  // CHECK-SAME:   c0: [[C0_READ]]: !moore.l1
  // CHECK-SAME: ) -> (
  // CHECK-SAME:   v1: !moore.l1
  // CHECK-SAME:   c1: !moore.l1
  // CHECK-SAME: )
  // CHECK-NEXT: moore.assign [[V1]], [[V1_VALUE]]
  // CHECK-NEXT: moore.assign [[C1]], [[C1_VALUE]]
  MultiPorts p3(x3, y3, z3, w3);

  wire x4, y4;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: [[A_VALUE:%.+]] = moore.read %a
  // CHECK: [[X4:%.+]] = moore.read %x4
  // CHECK: %c = moore.variable : <l1>
  // CHECK: [[C_VALUE:%.+]] = moore.read %c
  // CHECK: [[D_VALUE:%.+]], [[E_VALUE:%.+]] = moore.instance "p4" @PortsUnconnected(
  // CHECK-SAME: a: [[A_VALUE]]: !moore.l1
  // CHECK-SAME: b: [[X4]]: !moore.l1
  // CHECK-SAME: c: [[C_VALUE]]: !moore.l1
  // CHECK-SAME: ) -> (
  // CHECK-SAME: d: !moore.l1
  // CHECK-SAME: e: !moore.l1
  // CHECK-SAME: )
  // CHECK: moore.assign %y4, [[D_VALUE]] : l1
  PortsUnconnected p4(.a(), .b(x4), .c(), .d(y4), .e());
endmodule

// CHECK-LABEL: moore.module private @PortsAnsi
module PortsAnsi(
  // CHECK-SAME: in %a : !moore.l1
  input a,
  // CHECK-SAME: out b : !moore.l1
  output b,
  // CHECK-SAME: in %c : !moore.ref<l1>
  inout c,
  // CHECK-SAME: in %d : !moore.ref<l1>
  ref d
);
  // Internal nets and variables created by Slang for each port.
  // CHECK: [[A_INT:%.+]] = moore.net name "a" wire : <l1>
  // CHECK: [[B_INT:%.+]] = moore.net wire : <l1>
  // CHECK: [[C_INT:%.+]] = moore.net name "c" wire : <l1>
  // CHECK: [[D_INT:%.+]] = moore.variable name "d" : <l1>

  // Mapping ports to local declarations.
  // CHECK: moore.assign [[A_INT]], %a : l1
  // CHECK: [[B_READ:%.+]] = moore.read %b
  // CHECK: [[C_READ:%.+]] = moore.read %c
  // CHECK: moore.assign [[C_INT]], [[C_READ]] : l1
  // CHECK: [[D_READ:%.+]] = moore.read %d
  // CHECK: moore.assign [[D_INT]], [[D_READ]] : l1
  // CHECK: moore.output [[B_READ]] : !moore.l1
endmodule

// CHECK-LABEL: moore.module private @PortsNonAnsi
module PortsNonAnsi(a, b, c, d);
  // CHECK-SAME: in %a : !moore.l1
  input a;
  // CHECK-SAME: out b : !moore.l1
  output b;
  // CHECK-SAME: in %c : !moore.ref<l1>
  inout c;
  // CHECK-SAME: in %d : !moore.ref<l1>
  ref logic d;
endmodule

// CHECK-LABEL: moore.module private @PortsExplicit
module PortsExplicit(
  // CHECK-SAME: in %a0 : !moore.l1
  input .a0(x),
  // CHECK-SAME: in %a1 : !moore.l2
  input .a1({y, z}),
  // CHECK-SAME: out b0 : !moore.i32
  output .b0(42),
  // CHECK-SAME: out b1 : !moore.l1
  output .b1(x),
  // CHECK-SAME: out b2 : !moore.l1
  output .b2(y ^ z)
);
  logic x, y, z;

  // Input mappings
  // CHECK: moore.assign %x, %a0
  // CHECK: [[TMP:%.+]] = moore.concat_ref %y, %z
  // CHECK: moore.assign [[TMP]], %a1

  // Output mappings
  // CHECK: [[B0:%.+]] = moore.constant 42
  // CHECK: [[X_READ:%.+]] = moore.read %x
  // CHECK: [[Y_READ:%.+]] = moore.read %y
  // CHECK: [[Z_READ:%.+]] = moore.read %z
  // CHECK: [[B2:%.+]] = moore.xor [[Y_READ]], [[Z_READ]]
  // CHECK: moore.output [[B0]], [[X_READ]], [[B2]]
endmodule

// CHECK-LABEL: moore.module private @MultiPorts
module MultiPorts(
  // CHECK-SAME: in %a0 : !moore.l1
  .a0(u[0]),
  // CHECK-SAME: in %a1 : !moore.l1
  .a1(u[1]),
  // CHECK-SAME: in %v0 : !moore.l1
  // CHECK-SAME: out v1 : !moore.l1
  // CHECK-SAME: in %v2 : !moore.ref<l1>
  .b({v0, v1, v2}),
  // CHECK-SAME: in %c0 : !moore.l1
  // CHECK-SAME: out c1 : !moore.l1
  {c0, c1}
);
  // CHECK: [[V0:%.+]] = moore.net name "v0" wire
  // CHECK: [[V1:%.+]] = moore.net wire
  // CHECK: [[V2:%.+]] = moore.net name "v2" wire
  // CHECK: [[C0:%.+]] = moore.net name "c0" wire
  // CHECK: [[C1:%.+]] = moore.net wire
  input [1:0] u;
  input v0;
  output v1;
  inout v2;
  input c0;
  output c1;

  // CHECK: [[TMP2:%.+]] = moore.extract_ref %u from 0
  // CHECK: moore.assign [[TMP2]], %a0

  // CHECK: [[TMP2:%.+]] = moore.extract_ref %u from 1
  // CHECK: moore.assign [[TMP2]], %a1

  // CHECK: moore.assign [[V0]], %v0
  // CHECK: [[V1_READ:%.+]] = moore.read %v1
  // CHECK: [[V2_READ:%.+]] = moore.read %v2
  // CHECK: moore.assign [[V2]], [[V2_READ]]
  // CHECK: moore.assign [[C0]], %c0
  // CHECK: [[C1_READ:%.+]] = moore.read %c1
  // CHECK: moore.output [[V1_READ]], [[C1_READ]]
endmodule

// CHECK-LABEL: moore.module private @PortsUnconnected
module PortsUnconnected(
  // CHECK-SAME: in %a : !moore.l1
  input a,
  // CHECK-SAME: in %b : !moore.l1
  input b,
  // CHECK-SAME: in %c : !moore.l1
  input logic c,
  // CHECK-SAME: out d : !moore.l1
  output d,
  // CHECK-SAME: out e : !moore.l1
  output e
);
  // Internal nets and variables created by Slang for each port.
  // CHECK: [[A_INT:%.+]] = moore.net name "a" wire : <l1>
  // CHECK: [[B_INT:%.+]] = moore.net name "b" wire : <l1>
  // CHECK: [[C_INT:%.+]] = moore.variable name "c" : <l1>
  // CHECK: [[D_INT:%.+]] = moore.net wire : <l1>
  // CHECK: [[E_INT:%.+]] = moore.net wire : <l1>
  
  // Mapping ports to local declarations.
  // CHECK: moore.assign [[A_INT]], %a : l1
  // CHECK: moore.assign [[B_INT]], %b : l1
  // CHECK: [[D_READ:%.+]] = moore.read [[D_INT]]
  // CHECK: [[E_READ:%.+]] = moore.read [[E_INT]]
  // CHECK: moore.output [[D_READ]], [[E_READ]] : !moore.l1, !moore.l1
endmodule

// CHECK-LABEL: moore.module @GenerateConstructs()
module GenerateConstructs;
  genvar i;
  // CHECK: [[TMP:%.+]] = moore.constant 2
  // CHECK: dbg.variable "p", [[TMP]]
  parameter p = 2;
  
  generate
    // CHECK: [[TMP:%.+]] = moore.constant 0
    // CHECK: dbg.variable "i", [[TMP]]
    // CHECK: [[TMP:%.+]] = moore.constant 0
    // CHECK: %g1 = moore.variable [[TMP]]
    // CHECK: [[TMP:%.+]] = moore.constant 1
    // CHECK: dbg.variable "i", [[TMP]]
    // CHECK: [[TMP:%.+]] = moore.constant 1
    // CHECK: moore.variable name "g1" [[TMP]]
    for (i = 0; i < 2; i = i + 1) begin
      integer g1 = i;
    end

    // CHECK: [[TMP:%.+]] = moore.constant 2 : i32
    // CHECK: %g2 = moore.variable [[TMP]] : <i32>
    if (p == 2) begin
      int g2 = 2;
    end else begin
      int g2 = 3;
    end
    
    // CHECK: [[TMP:%.+]] = moore.constant 2 : i32
    // CHECK: %g3 = moore.variable [[TMP]] : <i32>
    case (p)
      2: begin
        int g3 = 2;
      end
      default: begin
        int g3 = 3;
      end
    endcase
  endgenerate
endmodule

// Should accept and ignore empty packages.
package Package;
  typedef logic [41:0] PackageType;
endpackage

// CHECK-LABEL: func.func private @simpleFunc1(
// CHECK-SAME:    %arg0: !moore.i32
// CHECK-SAME:    %arg1: !moore.i32
// CHECK-SAME:  ) -> !moore.i32
function int simpleFunc1(int a, b);
  // CHECK: [[RETVAR:%.+]] = moore.variable : <i32>
  // CHECK: [[TMP:%.+]] = moore.add %arg0, %arg1
  // CHECK: moore.blocking_assign [[RETVAR]], [[TMP]]
  simpleFunc1 = a + b;
  // CHECK: [[TMP:%.+]] = moore.read [[RETVAR]]
  // CHECK: return [[TMP]]
endfunction

// CHECK-LABEL: func.func private @simpleFunc2(
// CHECK-SAME:    %arg0: !moore.i32
// CHECK-SAME:    %arg1: !moore.i32
// CHECK-SAME:  ) -> !moore.i32
function int simpleFunc2(int a, b);
  // CHECK: [[TMP:%.+]] = moore.add %arg0, %arg1
  // CHECK: return [[TMP]]
  return a + b;
endfunction

package FuncPackage;
  // CHECK-LABEL: func.func private @"FuncPackage::simpleFunc3"(
  // CHECK-SAME:    %arg0: !moore.i32
  // CHECK-SAME:    %arg1: !moore.i32
  // CHECK-SAME:  ) -> !moore.i32
  function int simpleFunc3(int a, b);
    // CHECK: [[TMP:%.+]] = moore.mul %arg0, %arg1
    // CHECK: return [[TMP]]
    return a * b;
  endfunction
endpackage

// CHECK-LABEL: func.func private @simpleFunc4(
// CHECK-SAME:    %arg0: !moore.i32
// CHECK-SAME:    %arg1: !moore.i32
// CHECK-SAME:  )
function void simpleFunc4(int a, b);
  // CHECK: [[TMP1:%.+]] = call @simpleFunc1(%arg0, %arg1)
  // CHECK: [[TMP2:%.+]] = call @simpleFunc2(%arg0, %arg1)
  // CHECK: {{%.+}} = call @"FuncPackage::simpleFunc3"([[TMP1]], [[TMP2]])
  FuncPackage::simpleFunc3(
    simpleFunc1(a, b),
    simpleFunc2(a, b)
  );
  // CHECK: return
endfunction

// CHECK-LABEL: func.func private @simpleFunc5()
function void simpleFunc5();
  // CHECK: [[TMP1:%.+]] = moore.constant 42 : i32
  // CHECK: [[TMP2:%.+]] = moore.constant 9001 : i32
  // CHECK: call @simpleFunc4([[TMP1]], [[TMP2]])
  simpleFunc4(42, 9001);
  // CHECK: return
endfunction

// CHECK-LABEL: func.func private @funcArgs1(
// CHECK-SAME:    %arg0: !moore.i32
// CHECK-SAME:    %arg1: !moore.ref<i32>
// CHECK-SAME:    %arg2: !moore.ref<i32>
// CHECK-SAME:    %arg3: !moore.ref<i32>
// CHECK-SAME:    %arg4: !moore.ref<i32>
// CHECK-SAME:  )
function automatic void funcArgs1(
  input int a,
  output int b,
  inout int c,
  ref int d,
  const ref int e
);
  // CHECK: moore.blocking_assign %arg1, %arg0
  b = a;
  // CHECK: [[TMP1:%.+]] = moore.read %arg2
  // CHECK: [[TMP2:%.+]] = moore.add [[TMP1]], %arg0
  // CHECK: moore.blocking_assign %arg2, [[TMP2]]
  c += a;
  // CHECK: [[TMP:%.+]] = moore.read %arg4
  // CHECK: moore.blocking_assign %arg3, [[TMP]]
  d = e;
  // CHECK: return
endfunction

// CHECK-LABEL: func.func private @funcArgs2()
function void funcArgs2();
  // CHECK: %x = moore.variable
  // CHECK: %y = moore.variable
  // CHECK: %z = moore.variable
  // CHECK: %w = moore.variable
  int x, y, z, w;
  // CHECK: [[TMP:%.+]] = moore.constant 42
  // CHECK: call @funcArgs1([[TMP]], %x, %y, %z, %w)
  funcArgs1(42, x, y, z, w);
  // CHECK: return
endfunction

// CHECK-LABEL: func.func private @ConvertConditionalExprsToResultType(
function void ConvertConditionalExprsToResultType(bit [15:0] x, struct packed { bit [15:0] a; } y, bit z);
  bit [15:0] r;
  // CHECK: moore.conditional %arg2 : i1 -> i16 {
  // CHECK:   moore.yield %arg0
  // CHECK: } {
  // CHECK:   [[TMP:%.+]] = moore.conversion %arg1
  // CHECK:   moore.yield [[TMP]]
  // CHECK: }
  r = z ? x : y;
  // CHECK: moore.conditional %arg2 : i1 -> i16 {
  // CHECK:   [[TMP:%.+]] = moore.conversion %arg1
  // CHECK:   moore.yield [[TMP]]
  // CHECK: } {
  // CHECK:   moore.yield %arg0
  // CHECK: }
  r = z ? y : x;
endfunction

// CHECK-LABEL: func.func private @ImplicitEventControl(
// CHECK-SAME: [[X:%[^:]+]]: !moore.ref<i32>
// CHECK-SAME: [[Y:%[^:]+]]: !moore.ref<i32>
task automatic ImplicitEventControl(ref int x, ref int y);
  // CHECK: moore.wait_event {
  // CHECK: }
  // CHECK: call @dummyA()
  @* dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  // CHECK: [[TMP:%.+]] = moore.read [[X]]
  // CHECK: call @dummyD([[TMP]])
  @* dummyD(x);

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read [[Y]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  // CHECK: [[TMP1:%.+]] = moore.read [[X]]
  // CHECK: [[TMP2:%.+]] = moore.read [[Y]]
  // CHECK: [[TMP3:%.+]] = moore.add [[TMP1]], [[TMP2]]
  // CHECK: call @dummyD([[TMP3]])
  @* dummyD(x + y);
endtask

// CHECK-LABEL: func.func private @SignalEventControl(
// CHECK-SAME: [[X:%[^:]+]]: !moore.ref<i32>
// CHECK-SAME: [[Y:%[^:]+]]: !moore.ref<i32>
// CHECK-SAME: [[U:%[^:]+]]: !moore.ref<i1>
// CHECK-SAME: [[V:%[^:]+]]: !moore.ref<l1>
task automatic SignalEventControl(ref int x, ref int y, ref bit u, ref logic v);
  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  // CHECK: call @dummyA()
  @x dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(x) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event posedge [[TMP]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(posedge x) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event negedge [[TMP]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(negedge x) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event edge [[TMP]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(edge x) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP1:%.+]] = moore.read [[X]]
  // CHECK:   [[TMP2:%.+]] = moore.read [[U]]
  // CHECK:   moore.detect_event posedge [[TMP1]] if [[TMP2]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(posedge x iff u) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP1:%.+]] = moore.read [[X]]
  // CHECK:   [[TMP2:%.+]] = moore.read [[V]] : <l1>
  // CHECK:   [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.l1 -> !moore.i1
  // CHECK:   moore.detect_event posedge [[TMP1]] if [[TMP3]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(posedge x iff v) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP1:%.+]] = moore.read [[X]]
  // CHECK:   [[TMP2:%.+]] = moore.read [[Y]]
  // CHECK:   [[TMP3:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
  // CHECK:   moore.detect_event posedge [[TMP1]] if [[TMP3]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(posedge x iff y) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read [[Y]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(x or y) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read [[X]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read [[Y]]
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(x, y) dummyA();

  // CHECK: moore.wait_event {
  // CHECK:   [[TMP1:%.+]] = moore.read [[X]]
  // CHECK:   [[TMP2:%.+]] = moore.read [[U]]
  // CHECK:   moore.detect_event posedge [[TMP1]] if [[TMP2]]
  // CHECK:   [[TMP1:%.+]] = moore.read [[Y]]
  // CHECK:   [[TMP2:%.+]] = moore.read [[V]]
  // CHECK:   [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.l1 -> !moore.i1
  // CHECK:   moore.detect_event negedge [[TMP1]] if [[TMP3]]
  // CHECK: }
  // CHECK: call @dummyA()
  @(posedge x iff u, negedge y iff v) dummyA();
endtask

// CHECK-LABEL: func.func private @ImplicitEventControlExamples(
task automatic ImplicitEventControlExamples();
  // Taken from IEEE 1800-2017 section 9.4.2.2 "Implicit event_expression list".
  bit a, b, c, d, f, y, tmp1, tmp2;
  int x;

  // Example 1
  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read %a
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %b
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %c
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %d
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %f
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  @(*) y = (a & b) | (c & d) | dummyE(f);  // equivalent to @(a, b, c, d, f)

  // Example 2
  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read %a
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %b
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %c
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %d
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %tmp1
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %tmp2
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  @* begin  // equivalent to @(a, b, c, d, tmp1, tmp2)
    tmp1 = a & b;
    tmp2 = c & d;
    y = tmp1 | tmp2;
  end

  // Example 3
  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read %b
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read %a
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  @* begin // equivalent to @(b)
    @(a) y = b; // a inside @(a) is not added to outer @*
  end

  // Example 4
  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read %a
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %b
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %c
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %d
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read %c
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %d
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  @* begin // equivalent to @(a, b, c, d)
    y = a ^ b;
    @* y = c ^ d; // equivalent to @(c, d)
  end

  // Example 5
  // CHECK: moore.wait_event {
  // CHECK:   [[TMP:%.+]] = moore.read %a
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK:   [[TMP:%.+]] = moore.read %b
  // CHECK:   moore.detect_event any [[TMP]]
  // CHECK: }
  @* begin // equivalent to @(a, b)
    x[a] = !b;
  end
endtask

// CHECK-LABEL: moore.module @ImmediateAssert(in %clk : !moore.l1) 
module ImmediateAssert(input clk);
  // CHECK: [[CLK:%.+]] = moore.net name "clk" wire : <l1>
  // CHECK: [[A:%.+]] = moore.variable : <i1>
  bit a;

  // CHECK: moore.procedure always
    // CHECK: [[READ_CLK:%.+]] = moore.read [[CLK]] : <l1>
    // CHECK: [[OneBX:%.+]] = moore.constant bX : l1
    // CHECK: [[NE:%.+]] = moore.ne [[READ_CLK]], [[OneBX]] : l1 -> l1
    // CHECK: moore.assert immediate [[NE]] : l1
  assert (clk != 1'bx);

  // CHECK: moore.procedure always
    // CHECK: [[C100:%.+]] = moore.constant 100 : i32
    // CHECK: [[BC:%.+]] = moore.bool_cast [[C100]] : i32 -> i1
    // CHECK: moore.assume observed [[BC]] : i1
  assume #0 (100);

  // CHECK: moore.procedure always
    // CHECK: [[READ_A:%.+]] = moore.read [[A]] : <i1>
    // CHECK: moore.cover final [[READ_A]] : i1
  cover final (a);
endmodule

// CHECK-LABEL: moore.module @ImmediateAssertiWithActionBlock() 
module ImmediateAssertiWithActionBlock;
  logic x;
  int a;
// CHECK: moore.procedure always {
  // CHECK: [[READ_X:%.+]] = moore.read %x : <l1>
  // CHECK: [[CONV_X:%.+]] = moore.conversion [[READ_X]] : !moore.l1 -> i1
  // CHECK: cf.cond_br [[CONV_X]], ^bb1, ^bb2
// CHECK: ^bb1:  // pred: ^bb0
  // CHECK: [[C1:%.+]] = moore.constant 1 : i32
  // CHECK: moore.blocking_assign %a, [[C1]] : i32
  // CHECK: cf.br ^bb2
// CHECK: ^bb2:  // 2 preds: ^bb0, ^bb1
  // CHECK:   moore.return
// CHECK: }
  assert (x) a = 1;

// CHECK: moore.procedure always {
  // CHECK: [[READ_X:%.+]] = moore.read %x : <l1>
  // CHECK: [[CONV_X:%.+]] = moore.conversion [[READ_X]] : !moore.l1 -> i1
  // CHECK: cf.cond_br [[CONV_X]], ^bb1, ^bb2
// CHECK: ^bb1:  // pred: ^bb0
  // CHECK: cf.br ^bb3
// CHECK: ^bb2:  // pred: ^bb0
  // CHECK: [[C0:%.+]] = moore.constant 0 : i32
  // CHECK: moore.blocking_assign %a, [[C0]] : i32
  // CHECK: cf.br ^bb3
// CHECK: ^bb3:  // 2 preds: ^bb1, ^bb2
  // CHECK: moore.return
// CHECK: }
  assert (x) else a = 0;

// CHECK: moore.procedure always {
  // CHECK: [[READ_X:%.+]] = moore.read %x : <l1>
  // CHECK: [[CONV_X:%.+]] = moore.conversion [[READ_X]] : !moore.l1 -> i1
  // CHECK: cf.cond_br [[CONV_X]], ^bb1, ^bb2
// CHECK: ^bb1:  // pred: ^bb0
  // CHECK: [[C1:%.+]] = moore.constant 1 : i32
  // CHECK: moore.blocking_assign %a, [[C1]] : i32
  // CHECK: cf.br ^bb3
// CHECK: ^bb2:  // pred: ^bb0
  // CHECK: [[C0:%.+]] = moore.constant 0 : i32
  // CHECK: moore.blocking_assign %a, [[C0]] : i32
  // CHECK: cf.br ^bb3
// CHECK: ^bb3:  // 2 preds: ^bb1, ^bb2
  // CHECK: moore.return
// CHECK: }
  assert (x) a = 1; else a = 0;
endmodule

// CHECK: [[TMP:%.+]] = moore.constant 42 : i32
// CHECK: dbg.variable "rootParam1", [[TMP]] : !moore.i32
parameter int rootParam1 = 42;

// CHECK: [[TMP:%.+]] = moore.constant 9001 : i32
// CHECK: dbg.variable "rootParam2", [[TMP]] : !moore.i32
localparam int rootParam2 = 9001;

package ParamPackage;
  // CHECK: [[TMP:%.+]] = moore.constant 42 : i32
  // CHECK: dbg.variable "ParamPackage::param1", [[TMP]] : !moore.i32
  parameter int param1 = 42;

  // CHECK: [[TMP:%.+]] = moore.constant 9001 : i32
  // CHECK: dbg.variable "ParamPackage::param2", [[TMP]] : !moore.i32
  localparam int param2 = 9001;
endpackage

// CHECK-LABEL: moore.module @PortCastA()
module PortCastA;
  bit [31:0] a, b;
  // CHECK: [[TMP1:%.+]] = moore.read %a : <i32>
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.array<1 x i32>
  // CHECK: [[TMP3:%.+]] = moore.instance "sub" @PortCastB(a: [[TMP2]]: !moore.array<1 x i32>)
  // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.array<1 x i32> -> !moore.i32
  // CHECK: moore.assign %b, [[TMP4]] : i32
  PortCastB sub(a, b);
endmodule

module PortCastB (input bit [0:0][31:0] a, output bit [0:0][31:0] b);
  assign b = a;
endmodule

// CHECK-LABEL: func.func private @SignCastsA(
// CHECK-SAME: %arg0: !moore.l16
function void SignCastsA(logic [15:0] value);
  // CHECK: [[TMP:%.+]] = moore.zext %arg0 : l16 -> l32
  // CHECK: call @SignCastsB([[TMP]])
  SignCastsB($unsigned(value));
  // CHECK: [[TMP:%.+]] = moore.sext %arg0 : l16 -> l32
  // CHECK: call @SignCastsB([[TMP]])
  SignCastsB($signed(value));

  // CHECK: [[TMP:%.+]] = moore.zext %arg0 : l16 -> l32
  // CHECK: call @SignCastsB([[TMP]])
  SignCastsB(unsigned'(value));
  // CHECK: [[TMP:%.+]] = moore.sext %arg0 : l16 -> l32
  // CHECK: call @SignCastsB([[TMP]])
  SignCastsB(signed'(value));
endfunction

function void SignCastsB(logic [31:0] value);
endfunction

// CHECK-LABEL: func.func private @AssignFuncArgs(
// CHECK-SAME: %arg0: !moore.i32
function void AssignFuncArgs(int x);
  // CHECK: [[ARG:%.+]] = moore.variable %arg0 : <i32>
  // CHECK: [[READ:%.+]] = moore.constant 1 : i32
  // CHECK: moore.blocking_assign [[ARG]], [[READ]] : i32
  x = 1;
endfunction

// CHECK-LABEL: func.func private @AssignFuncArgs2(
// CHECK-SAME: %arg0: !moore.i32, %arg1: !moore.i32
function int AssignFuncArgs2(int x, int y);
  // CHECK: [[X:%.+]] = moore.variable %arg0 : <i32>
  // CHECK: [[Y:%.+]] = moore.variable %arg1 : <i32>
  // CHECK: [[C1:%.+]] = moore.constant 1 : i32
  // CHECK: moore.blocking_assign [[X]], [[C1]] : i32
  x = 1;
  
  // CHECK: [[C2:%.+]] = moore.constant 2 : i32
  // CHECK: moore.blocking_assign [[Y]], [[C2]] : i32
  y = 2;

  // CHECK: [[READ_X:%.+]] = moore.read [[X]] : <i32>
  // CHECK: [[READ_Y:%.+]] = moore.read [[Y]] : <i32>
  // CHECK: [[ADD:%.+]] = moore.add [[READ_X]], [[READ_Y]] : i32
  return x+y;
endfunction
