// RUN: circt-verilog --ir-moore --allow-use-before-declare %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Broad expression coverage for module-body names resolved by Slang's
// `AllowUseBeforeDeclare` mode. The focused tests check exact IR. This file
// keeps a matrix of common rvalue, lvalue, interface, instance, and generate
// expression shapes importing successfully when declarations appear late.

interface ExprBus(input logic clk);
  logic sig;
  logic [7:0] data;
  int count;
  assign sig = clk;
  assign data = {7'b0, clk};
  assign count = 13;
endinterface

interface ExprWriteBus(input logic clk);
  logic sig;
  logic [7:0] data;
  int count;
endinterface

module ExprLeaf(input logic [7:0] in, output logic [7:0] out);
  assign out = in;
endmodule

typedef struct packed {
  logic [3:0] lo;
  logic [3:0] hi;
} expr_pair_t;

// CHECK-LABEL: moore.module @ExprAdd(
module ExprAdd(output int out);
  assign out = lhs + rhs;
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ExprSub(
module ExprSub(output int out);
  assign out = lhs - rhs;
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ExprMul(
module ExprMul(output int out);
  assign out = lhs * rhs;
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ExprDivMod(
module ExprDivMod(output int out);
  assign out = (lhs / rhs) + (lhs % rhs);
  int lhs;
  int rhs = 1;
endmodule

// CHECK-LABEL: moore.module @ExprBitAnd(
module ExprBitAnd(output logic [7:0] out);
  assign out = lhs & rhs;
  logic [7:0] lhs;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprBitOr(
module ExprBitOr(output logic [7:0] out);
  assign out = lhs | rhs;
  logic [7:0] lhs;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprBitXor(
module ExprBitXor(output logic [7:0] out);
  assign out = lhs ^ rhs;
  logic [7:0] lhs;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprBitNot(
module ExprBitNot(output logic [7:0] out);
  assign out = ~late;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprReductionAnd(
module ExprReductionAnd(output logic out);
  assign out = &late;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprReductionOr(
module ExprReductionOr(output logic out);
  assign out = |late;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprReductionXor(
module ExprReductionXor(output logic out);
  assign out = ^late;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprLogicalAnd(
module ExprLogicalAnd(output logic out);
  assign out = lhs && rhs;
  logic lhs;
  logic rhs;
endmodule

// CHECK-LABEL: moore.module @ExprLogicalOr(
module ExprLogicalOr(output logic out);
  assign out = lhs || rhs;
  logic lhs;
  logic rhs;
endmodule

// CHECK-LABEL: moore.module @ExprLogicalNot(
module ExprLogicalNot(output logic out);
  assign out = !late;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ExprEquality(
module ExprEquality(output logic out);
  assign out = lhs == rhs;
  logic [7:0] lhs;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprInequality(
module ExprInequality(output logic out);
  assign out = lhs != rhs;
  logic [7:0] lhs;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprCaseEquality(
module ExprCaseEquality(output logic out);
  assign out = lhs === rhs;
  logic [7:0] lhs;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprLessThan(
module ExprLessThan(output logic out);
  assign out = lhs < rhs;
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ExprGreaterEqual(
module ExprGreaterEqual(output logic out);
  assign out = lhs >= rhs;
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ExprShiftLeft(
module ExprShiftLeft(output logic [7:0] out);
  assign out = data << shift;
  logic [7:0] data;
  int shift;
endmodule

// CHECK-LABEL: moore.module @ExprShiftRight(
module ExprShiftRight(output logic [7:0] out);
  assign out = data >> shift;
  logic [7:0] data;
  int shift;
endmodule

// CHECK-LABEL: moore.module @ExprArithmeticShift(
module ExprArithmeticShift(output logic signed [7:0] out);
  assign out = data >>> shift;
  logic signed [7:0] data;
  int shift;
endmodule

// CHECK-LABEL: moore.module @ExprConcat(
module ExprConcat(output logic [7:0] out);
  assign out = {hi, lo};
  logic [3:0] hi;
  logic [3:0] lo;
endmodule

// CHECK-LABEL: moore.module @ExprNestedConcat(
module ExprNestedConcat(output logic [11:0] out);
  assign out = {prefix, {mid, suffix}};
  logic [3:0] prefix;
  logic [3:0] mid;
  logic [3:0] suffix;
endmodule

// CHECK-LABEL: moore.module @ExprReplication(
module ExprReplication(output logic [7:0] out);
  assign out = {4{late}};
  logic [1:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprPartSelect(
module ExprPartSelect(output logic [3:0] out);
  assign out = data[7:4];
  logic [7:0] data;
endmodule

// CHECK-LABEL: moore.module @ExprIndexedPartSelectPlus(
module ExprIndexedPartSelectPlus(output logic [3:0] out);
  assign out = data[base +: 4];
  logic [7:0] data;
  int base;
endmodule

// CHECK-LABEL: moore.module @ExprIndexedPartSelectMinus(
module ExprIndexedPartSelectMinus(output logic [3:0] out);
  assign out = data[base -: 4];
  logic [7:0] data;
  int base = 7;
endmodule

// CHECK-LABEL: moore.module @ExprArraySelect(
module ExprArraySelect(output logic [7:0] out);
  assign out = arr[index];
  logic [7:0] arr [0:3];
  int index;
endmodule

// CHECK-LABEL: moore.module @ExprArrayElementWrite(
module ExprArrayElementWrite(output logic [7:0] out);
  initial arr[index] = late;
  assign out = arr[index];
  logic [7:0] arr [0:3];
  int index;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprStructFieldLow(
module ExprStructFieldLow(output logic [3:0] out);
  assign out = pair.lo;
  expr_pair_t pair;
endmodule

// CHECK-LABEL: moore.module @ExprStructFieldHigh(
module ExprStructFieldHigh(output logic [3:0] out);
  assign out = pair.hi;
  expr_pair_t pair;
endmodule

// CHECK-LABEL: moore.module @ExprStructWrite(
module ExprStructWrite(output logic [3:0] out);
  initial pair.hi = late;
  assign out = pair.hi;
  expr_pair_t pair;
  logic [3:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprStructConcat(
module ExprStructConcat(output logic [7:0] out);
  assign out = {pair.hi, pair.lo};
  expr_pair_t pair;
endmodule

// CHECK-LABEL: moore.module @ExprConditional(
module ExprConditional(output logic [7:0] out);
  assign out = sel ? lhs : rhs;
  logic sel;
  logic [7:0] lhs;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprNestedConditional(
module ExprNestedConditional(output logic [7:0] out);
  assign out = a ? lhs : (b ? mid : rhs);
  logic a;
  logic b;
  logic [7:0] lhs;
  logic [7:0] mid;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprLogicCast(
module ExprLogicCast(output logic out);
  assign out = logic'(late);
  int late;
endmodule

// CHECK-LABEL: moore.module @ExprIntCast(
module ExprIntCast(output int out);
  assign out = int'(late);
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprSignedCast(
module ExprSignedCast(output logic signed [7:0] out);
  assign out = signed'(late);
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprUnsignedCast(
module ExprUnsignedCast(output logic [7:0] out);
  assign out = unsigned'(late);
  logic signed [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprInitializerBinary(
module ExprInitializerBinary(output int out);
  int captured = lhs + rhs;
  assign out = captured;
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ExprInitializerConcat(
module ExprInitializerConcat(output logic [7:0] out);
  logic [7:0] captured = {hi, lo};
  assign out = captured;
  logic [3:0] hi;
  logic [3:0] lo;
endmodule

// CHECK-LABEL: moore.module @ExprInitializerArray(
module ExprInitializerArray(output logic [7:0] out);
  logic [7:0] captured = arr[index];
  assign out = captured;
  logic [7:0] arr [0:3];
  int index;
endmodule

// CHECK-LABEL: moore.module @ExprInitializerStruct(
module ExprInitializerStruct(output logic [3:0] out);
  logic [3:0] captured = pair.lo;
  assign out = captured;
  expr_pair_t pair;
endmodule

// CHECK-LABEL: moore.module @ExprNetInitializerBinary(
module ExprNetInitializerBinary(output logic [31:0] out);
  wire [31:0] captured = lhs + rhs;
  assign out = captured;
  logic [31:0] lhs;
  logic [31:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ExprNetInitializerConcat(
module ExprNetInitializerConcat(output logic [7:0] out);
  wire [7:0] captured = {hi, lo};
  assign out = captured;
  logic [3:0] hi;
  logic [3:0] lo;
endmodule

// CHECK-LABEL: moore.module @ExprInterfaceSignal(
module ExprInterfaceSignal(output logic out);
  assign out = bus.sig ^ late;
  ExprBus bus(clk);
  logic clk;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ExprInterfaceData(
module ExprInterfaceData(output logic [7:0] out);
  assign out = bus.data + late;
  ExprBus bus(clk);
  logic clk;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprInterfaceCount(
module ExprInterfaceCount(output int out);
  assign out = bus.count + late;
  ExprBus bus(clk);
  logic clk;
  int late;
endmodule

// CHECK-LABEL: moore.module @ExprInterfaceWriteExpression(
module ExprInterfaceWriteExpression(output logic [7:0] out);
  initial bus.data[index +: 4] = late;
  assign out = bus.data;
  ExprWriteBus bus(clk);
  logic clk;
  int index;
  logic [3:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprInterfaceInitializer(
module ExprInterfaceInitializer(output logic [7:0] out);
  logic [7:0] captured = bus.data ^ late;
  assign out = captured;
  ExprBus bus(clk);
  logic clk;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprInstanceOutput(
module ExprInstanceOutput(output logic [7:0] out);
  assign out = u.out ^ late;
  ExprLeaf u(.in(source), .out(child_out));
  logic [7:0] source;
  logic [7:0] child_out;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprInstanceInput(
module ExprInstanceInput(output logic [7:0] out);
  assign out = u.out;
  ExprLeaf u(.in(source ^ mask), .out(child_out));
  logic [7:0] source;
  logic [7:0] mask;
  logic [7:0] child_out;
endmodule

// CHECK-LABEL: moore.module @ExprInstanceInitializer(
module ExprInstanceInitializer(output logic [7:0] out);
  logic [7:0] captured = u.out + late;
  assign out = captured;
  ExprLeaf u(.in(source), .out(child_out));
  logic [7:0] source;
  logic [7:0] child_out;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ExprGenerateIf(
module ExprGenerateIf(output logic [7:0] out);
  if (1) begin : g
    assign out = lhs + rhs;
    logic [7:0] lhs;
    logic [7:0] rhs;
  end
endmodule

// CHECK-LABEL: moore.module @ExprGenerateNestedIf(
module ExprGenerateNestedIf(output logic [7:0] out);
  if (1) begin : outer
    if (1) begin : inner
      assign out = lhs ^ rhs;
      logic [7:0] lhs;
      logic [7:0] rhs;
    end
  end
endmodule

// CHECK-LABEL: moore.module @ExprGenerateCase(
module ExprGenerateCase(output logic [7:0] out);
  case (1)
    1: begin : g
      assign out = {hi, lo};
      logic [3:0] hi;
      logic [3:0] lo;
    end
  endcase
endmodule

// CHECK-LABEL: moore.module @ExprGenerateLoop(
module ExprGenerateLoop(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    assign out[i] = lhs ^ rhs;
    logic lhs;
    logic rhs;
  end
endmodule

// CHECK-LABEL: moore.module @ExprGenerateLoopInitializer(
module ExprGenerateLoopInitializer(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    logic captured = lhs ^ bit'(i);
    assign out[i] = captured;
    logic lhs;
  end
endmodule

// CHECK-LABEL: moore.module @ExprGenerateInterface(
module ExprGenerateInterface(output logic [7:0] out);
  if (1) begin : g
    assign out = bus.data + late;
    ExprBus bus(clk);
    logic clk;
    logic [7:0] late;
  end
endmodule

// CHECK-LABEL: moore.module @ExprGenerateInstance(
module ExprGenerateInstance(output logic [7:0] out);
  if (1) begin : g
    assign out = u.out + late;
    ExprLeaf u(.in(source), .out(child_out));
    logic [7:0] source;
    logic [7:0] child_out;
    logic [7:0] late;
  end
endmodule

// CHECK-LABEL: moore.module @ExprLoopInterface(
module ExprLoopInterface(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    assign out[i] = bus.sig ^ late;
    ExprBus bus(clk);
    logic clk;
    logic late;
  end
endmodule

// CHECK-LABEL: moore.module @ExprLoopInstance(
module ExprLoopInstance(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    assign out[i] = u.out[0] ^ late;
    ExprLeaf u(.in(source), .out(child_out));
    logic [7:0] source;
    logic [7:0] child_out;
    logic late;
  end
endmodule
