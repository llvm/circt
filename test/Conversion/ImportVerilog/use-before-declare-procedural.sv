// RUN: circt-verilog --ir-moore --allow-use-before-declare %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Procedural compatibility coverage for module-body names resolved by Slang's
// `AllowUseBeforeDeclare` mode. These cases exercise control-flow, assignments,
// event controls, subroutines, interfaces, instances, and generated scopes where
// the referenced declarations appear after the procedural code.

interface ProcBus(input logic clk);
  logic sig;
  logic [7:0] data;
  assign sig = clk;
  assign data = {7'b0, clk};
endinterface

interface ProcWriteBus(input logic clk);
  logic sig;
  logic [7:0] data;
endinterface

module ProcLeaf(input logic [7:0] in, output logic [7:0] out);
  assign out = in;
endmodule

module ProcReadable(output int value);
  assign value = 21;
endmodule

typedef struct packed {
  logic [3:0] lo;
  logic [3:0] hi;
} proc_pair_t;

// CHECK-LABEL: moore.module @ProcInitialIf(
module ProcInitialIf(output logic out);
  initial if (sel) out = late;
  logic sel;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialIfElse(
module ProcInitialIfElse(output logic out);
  initial if (sel) out = lhs; else out = rhs;
  logic sel;
  logic lhs;
  logic rhs;
endmodule

// CHECK-LABEL: moore.module @ProcInitialNestedIf(
module ProcInitialNestedIf(output logic out);
  initial if (outer) if (inner) out = lhs; else out = rhs;
  logic outer;
  logic inner;
  logic lhs;
  logic rhs;
endmodule

// CHECK-LABEL: moore.module @ProcInitialCase(
module ProcInitialCase(output logic [3:0] out);
  initial case (sel)
    2'd0: out = a;
    2'd1: out = b;
    default: out = c;
  endcase
  logic [1:0] sel;
  logic [3:0] a;
  logic [3:0] b;
  logic [3:0] c;
endmodule

// CHECK-LABEL: moore.module @ProcInitialCasez(
module ProcInitialCasez(output logic out);
  initial casez (sel)
    2'b1?: out = a;
    default: out = b;
  endcase
  logic [1:0] sel;
  logic a;
  logic b;
endmodule

// CHECK-LABEL: moore.module @ProcInitialFor(
module ProcInitialFor(output logic [3:0] out);
  initial for (i = 0; i < 4; ++i) out[i] = data[i];
  int i;
  logic [3:0] data;
endmodule

// CHECK-LABEL: moore.module @ProcInitialForBlock(
module ProcInitialForBlock(output logic [3:0] out);
  initial begin
    for (i = 0; i < 4; ++i)
      out[i] = data[i] ^ mask[i];
  end
  int i;
  logic [3:0] data;
  logic [3:0] mask;
endmodule

// CHECK-LABEL: moore.module @ProcInitialRepeat(
module ProcInitialRepeat(output logic [3:0] out);
  initial repeat (count) out = data;
  int count;
  logic [3:0] data;
endmodule

// CHECK-LABEL: moore.module @ProcInitialWhile(
module ProcInitialWhile(output logic [3:0] out);
  initial while (enable) out = data;
  logic enable;
  logic [3:0] data;
endmodule

// CHECK-LABEL: moore.module @ProcInitialDoWhile(
module ProcInitialDoWhile(output logic out);
  initial do out = late; while (enable);
  logic late;
  logic enable;
endmodule

// CHECK-LABEL: moore.module @ProcInitialWait(
module ProcInitialWait(output logic out);
  initial wait (ready) out = late;
  logic ready;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialEventControl(
module ProcInitialEventControl(output logic out);
  initial @(posedge clk) out = late;
  logic clk;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialDelayControl(
module ProcInitialDelayControl(output logic out);
  initial #delay out = late;
  int delay;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialNamedBlock(
module ProcInitialNamedBlock(output int out);
  initial begin : body
    out = late;
  end
  int late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialSequentialBlock(
module ProcInitialSequentialBlock(output int out);
  initial begin
    tmp = lhs + rhs;
    out = tmp;
  end
  int tmp;
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ProcInitialNonblocking(
module ProcInitialNonblocking(output logic out);
  initial out <= late;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialCompoundAdd(
module ProcInitialCompoundAdd(output int out);
  initial out += late;
  int late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialCompoundXor(
module ProcInitialCompoundXor(output logic [7:0] out);
  initial out ^= late;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialIncrement(
module ProcInitialIncrement(output int out);
  initial begin
    late++;
    out = late;
  end
  int late;
endmodule

// CHECK-LABEL: moore.module @ProcInitialDecrement(
module ProcInitialDecrement(output int out);
  initial begin
    late--;
    out = late;
  end
  int late;
endmodule

// CHECK-LABEL: moore.module @ProcFinalIf(
module ProcFinalIf(output logic out);
  final if (sel) out = late;
  logic sel;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcFinalCase(
module ProcFinalCase(output logic [3:0] out);
  final case (sel)
    1'b0: out = lhs;
    default: out = rhs;
  endcase
  logic sel;
  logic [3:0] lhs;
  logic [3:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ProcAlwaysCombIf(
module ProcAlwaysCombIf(output logic out);
  always_comb if (sel) out = late; else out = other;
  logic sel;
  logic late;
  logic other;
endmodule

// CHECK-LABEL: moore.module @ProcAlwaysCombCase(
module ProcAlwaysCombCase(output logic [3:0] out);
  always_comb case (sel)
    2'd0: out = a;
    2'd1: out = b;
    default: out = c;
  endcase
  logic [1:0] sel;
  logic [3:0] a;
  logic [3:0] b;
  logic [3:0] c;
endmodule

// CHECK-LABEL: moore.module @ProcAlwaysAtStarBlock(
module ProcAlwaysAtStarBlock(output logic [7:0] out);
  always @(*) begin
    tmp = lhs ^ rhs;
    out = tmp;
  end
  logic [7:0] tmp;
  logic [7:0] lhs;
  logic [7:0] rhs;
endmodule

// CHECK-LABEL: moore.module @ProcAlwaysLatch(
module ProcAlwaysLatch(output logic out);
  always_latch if (enable) out = late;
  logic enable;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcAlwaysFF(
module ProcAlwaysFF(input logic clk, output logic out);
  always_ff @(posedge clk) out <= late;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcAlwaysFFReset(
module ProcAlwaysFFReset(input logic clk, output logic out);
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) out <= reset_value; else out <= late;
  logic rst_n;
  logic reset_value;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcAlwaysFFEnable(
module ProcAlwaysFFEnable(input logic clk, output logic out);
  always_ff @(posedge clk) if (enable) out <= late;
  logic enable;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcFunctionReturn(
module ProcFunctionReturn(output int out);
  initial out = compute();
  function int compute();
    return late;
  endfunction
  int late;
endmodule

// CHECK-LABEL: moore.module @ProcFunctionExpression(
module ProcFunctionExpression(output int out);
  initial out = compute();
  function int compute();
    return lhs + rhs;
  endfunction
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ProcFunctionStruct(
module ProcFunctionStruct(output logic [3:0] out);
  initial out = select_lo();
  function logic [3:0] select_lo();
    return pair.lo;
  endfunction
  proc_pair_t pair;
endmodule

// CHECK-LABEL: moore.module @ProcTaskAssign(
module ProcTaskAssign(output int out);
  initial drive();
  task drive();
    out = late;
  endtask
  int late;
endmodule

// CHECK-LABEL: moore.module @ProcTaskBlock(
module ProcTaskBlock(output int out);
  initial drive();
  task drive();
    tmp = lhs + rhs;
    out = tmp;
  endtask
  int tmp;
  int lhs;
  int rhs;
endmodule

// CHECK-LABEL: moore.module @ProcInterfaceInitialRead(
module ProcInterfaceInitialRead(output logic out);
  initial out = bus.sig;
  ProcBus bus(clk);
  logic clk;
endmodule

// CHECK-LABEL: moore.module @ProcInterfaceInitialWrite(
module ProcInterfaceInitialWrite(output logic out);
  initial bus.sig = late;
  assign out = bus.sig;
  ProcWriteBus bus(clk);
  logic clk;
  logic late;
endmodule

// CHECK-LABEL: moore.module @ProcInterfaceAlwaysComb(
module ProcInterfaceAlwaysComb(output logic [7:0] out);
  always_comb out = bus.data ^ late;
  ProcBus bus(clk);
  logic clk;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ProcInterfaceCase(
module ProcInterfaceCase(output logic [7:0] out);
  initial case (sel)
    1'b0: bus.data = a;
    default: bus.data = b;
  endcase
  assign out = bus.data;
  ProcWriteBus bus(clk);
  logic clk;
  logic sel;
  logic [7:0] a;
  logic [7:0] b;
endmodule

// CHECK-LABEL: moore.module @ProcInstanceInitialRead(
module ProcInstanceInitialRead(output int out);
  initial out = u.value;
  ProcReadable u(.value(child_value));
  int child_value;
endmodule

// CHECK-LABEL: moore.module @ProcInstanceInitialInput(
module ProcInstanceInitialInput(output logic [7:0] out);
  initial out = u.out;
  ProcLeaf u(.in(source ^ mask), .out(child_out));
  logic [7:0] source;
  logic [7:0] mask;
  logic [7:0] child_out;
endmodule

// CHECK-LABEL: moore.module @ProcInstanceAlwaysComb(
module ProcInstanceAlwaysComb(output logic [7:0] out);
  always_comb out = u.out + late;
  ProcLeaf u(.in(source), .out(child_out));
  logic [7:0] source;
  logic [7:0] child_out;
  logic [7:0] late;
endmodule

// CHECK-LABEL: moore.module @ProcGenerateIfInitial(
module ProcGenerateIfInitial(output logic out);
  if (1) begin : g
    initial out = late;
    logic late;
  end
endmodule

// CHECK-LABEL: moore.module @ProcGenerateIfAlways(
module ProcGenerateIfAlways(output logic out);
  if (1) begin : g
    always_comb out = lhs ^ rhs;
    logic lhs;
    logic rhs;
  end
endmodule

// CHECK-LABEL: moore.module @ProcGenerateIfInterface(
module ProcGenerateIfInterface(output logic out);
  if (1) begin : g
    initial out = bus.sig;
    ProcBus bus(clk);
    logic clk;
  end
endmodule

// CHECK-LABEL: moore.module @ProcGenerateIfInstance(
module ProcGenerateIfInstance(output int out);
  if (1) begin : g
    initial out = u.value;
    ProcReadable u(.value(child_value));
    int child_value;
  end
endmodule

// CHECK-LABEL: moore.module @ProcGenerateCaseInitial(
module ProcGenerateCaseInitial(output logic [3:0] out);
  case (1)
    1: begin : g
      initial out = late;
      logic [3:0] late;
    end
  endcase
endmodule

// CHECK-LABEL: moore.module @ProcGenerateCaseInterface(
module ProcGenerateCaseInterface(output logic out);
  case (1)
    1: begin : g
      initial out = bus.sig;
      ProcBus bus(clk);
      logic clk;
    end
  endcase
endmodule

// CHECK-LABEL: moore.module @ProcGenerateLoopInitial(
module ProcGenerateLoopInitial(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    initial out[i] = late ^ bit'(i);
    logic late;
  end
endmodule

// CHECK-LABEL: moore.module @ProcGenerateLoopAlways(
module ProcGenerateLoopAlways(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    always_comb out[i] = lhs ^ rhs;
    logic lhs;
    logic rhs;
  end
endmodule

// CHECK-LABEL: moore.module @ProcGenerateLoopInterface(
module ProcGenerateLoopInterface(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    initial out[i] = bus.sig;
    ProcBus bus(clk);
    logic clk;
  end
endmodule

// CHECK-LABEL: moore.module @ProcGenerateLoopInstance(
module ProcGenerateLoopInstance(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    initial out[i] = u.out[0];
    ProcLeaf u(.in(source), .out(child_out));
    logic [7:0] source;
    logic [7:0] child_out;
  end
endmodule

// CHECK-LABEL: moore.module @ProcNestedGenerateProcedure(
module ProcNestedGenerateProcedure(output logic [3:0] out);
  for (genvar i = 0; i < 2; ++i) begin : row
    for (genvar j = 0; j < 2; ++j) begin : col
      initial out[i * 2 + j] = late;
      logic late;
    end
  end
endmodule

// CHECK-LABEL: moore.module @ProcNestedGenerateInterface(
module ProcNestedGenerateInterface(output logic [3:0] out);
  for (genvar i = 0; i < 2; ++i) begin : row
    for (genvar j = 0; j < 2; ++j) begin : col
      initial out[i * 2 + j] = bus.sig;
      ProcBus bus(clk);
      logic clk;
    end
  end
endmodule

// CHECK-LABEL: moore.module @ProcMixedInterfaceInstance(
module ProcMixedInterfaceInstance(output logic [7:0] out);
  initial begin
    bus.data = seed;
    out = u.out ^ bus.data;
  end
  ProcWriteBus bus(clk);
  ProcLeaf u(.in(source), .out(child_out));
  logic clk;
  logic [7:0] seed;
  logic [7:0] source;
  logic [7:0] child_out;
endmodule
