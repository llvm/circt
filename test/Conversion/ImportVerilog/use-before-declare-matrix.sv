// RUN: circt-verilog --ir-moore --allow-use-before-declare %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// A compatibility matrix for common testbench shapes that depend on Slang's
// `AllowUseBeforeDeclare` mode. The more surgical tests in
// use-before-declare*.sv check exact IR for the new predeclaration paths; this
// file keeps broad coverage that these independently common source patterns
// continue to import successfully.

interface MatrixIf(input logic clk);
  logic sig;
  logic [3:0] data;
  assign sig = clk;
endinterface

interface MatrixWriteIf(input logic clk);
  logic sig;
  logic [3:0] data;
endinterface

module MatrixLeaf(input logic in, output logic out);
  assign out = in;
endmodule

module MatrixStorageBit;
  int bit_value;
endmodule

module MatrixStorageData;
  logic [3:0] data;
endmodule

module MatrixReadableStorage(output int bit_value, output logic [3:0] data);
  assign bit_value = 3;
  assign data = 4'h5;
endmodule

typedef struct packed {
  logic [3:0] lo;
  logic [3:0] hi;
} matrix_pair_t;

// CHECK-LABEL: moore.module @MatrixInitialBlock(
module MatrixInitialBlock(output logic out);
  initial out = late;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixInitialBeginBlock(
module MatrixInitialBeginBlock(output logic out);
  initial begin
    tmp = late;
    out = tmp;
  end
  logic tmp;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixAlwaysComb(
module MatrixAlwaysComb(output logic out);
  always_comb out = late;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixAlwaysAtStar(
module MatrixAlwaysAtStar(output logic out);
  always @(*) out = late;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixAlwaysFF(
module MatrixAlwaysFF(input logic clk, output logic out);
  always_ff @(posedge clk) out <= late;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixFinalBlock(
module MatrixFinalBlock(output logic out);
  final out = late;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixContinuousAssign(
module MatrixContinuousAssign(output logic out);
  assign out = late;
  wire late = 1'b1;
endmodule

// CHECK-LABEL: moore.module @MatrixNetDeclarationAssign(
module MatrixNetDeclarationAssign(output logic out);
  assign out = early;
  wire early = late;
  wire late = 1'b1;
endmodule

// CHECK-LABEL: moore.module @MatrixVariableInitializer(
module MatrixVariableInitializer(output int out);
  int early = late;
  int late = 32'd9;
  assign out = early;
endmodule

// CHECK-LABEL: moore.module @MatrixInitializerChain(
module MatrixInitializerChain(output int out);
  int a = b;
  int b = c;
  int c = 32'd12;
  assign out = a;
endmodule

// CHECK-LABEL: moore.module @MatrixArraySelect(
module MatrixArraySelect(output logic out);
  initial out = data[index];
  logic [3:0] data;
  int index = 1;
endmodule

// CHECK-LABEL: moore.module @MatrixArrayElementWrite(
module MatrixArrayElementWrite(output logic [3:0] out);
  initial data[index] = value;
  assign out = data;
  logic [3:0] data;
  int index = 2;
  logic value;
endmodule

// CHECK-LABEL: moore.module @MatrixStructRead(
module MatrixStructRead(output logic [3:0] out);
  initial out = pair.hi;
  matrix_pair_t pair;
endmodule

// CHECK-LABEL: moore.module @MatrixStructWrite(
module MatrixStructWrite(output logic [3:0] out);
  initial pair.lo = value;
  assign out = pair.lo;
  matrix_pair_t pair;
  logic [3:0] value;
endmodule

// CHECK-LABEL: moore.module @MatrixPackedSliceRead(
module MatrixPackedSliceRead(output logic [3:0] out);
  assign out = data[7:4];
  logic [7:0] data;
endmodule

// CHECK-LABEL: moore.module @MatrixPackedSliceWrite(
module MatrixPackedSliceWrite(output logic [7:0] out);
  initial data[3:0] = value;
  assign out = data;
  logic [7:0] data;
  logic [3:0] value;
endmodule

// CHECK-LABEL: moore.module @MatrixNamedGenerateInitial(
module MatrixNamedGenerateInitial(output logic out);
  if (1) begin : g
    initial out = late;
    logic late;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixNamedGenerateAssign(
module MatrixNamedGenerateAssign(output logic out);
  if (1) begin : g
    assign out = late;
    wire late = 1'b1;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixNamedGenerateInitializer(
module MatrixNamedGenerateInitializer(output int out);
  if (1) begin : g
    int early = late;
    int late = 32'd15;
    assign out = early;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixNestedGenerateInitial(
module MatrixNestedGenerateInitial(output logic out);
  if (1) begin : outer
    if (1) begin : inner
      initial out = late;
      logic late;
    end
  end
endmodule

// CHECK-LABEL: moore.module @MatrixNestedGenerateAssign(
module MatrixNestedGenerateAssign(output logic out);
  if (1) begin : outer
    if (1) begin : inner
      assign out = late;
      wire late = 1'b1;
    end
  end
endmodule

// CHECK-LABEL: moore.module @MatrixCaseGenerateDefault(
module MatrixCaseGenerateDefault(output logic out);
  parameter int Sel = 3;
  case (Sel)
    0: begin : a
      assign out = 1'b0;
    end
    default: begin : b
      initial out = late;
      logic late;
    end
  endcase
endmodule

// CHECK-LABEL: moore.module @MatrixLoopGenerateInitial(
module MatrixLoopGenerateInitial(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    initial bit_out = bit'(i);
    assign out[i] = bit_out;
    bit bit_out;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixLoopGenerateAssign(
module MatrixLoopGenerateAssign(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    assign out[i] = late;
    wire late = bit'(i);
  end
endmodule

// CHECK-LABEL: moore.module @MatrixNestedLoopGenerate(
module MatrixNestedLoopGenerate(output logic [3:0] out);
  for (genvar i = 0; i < 2; ++i) begin : row
    for (genvar j = 0; j < 2; ++j) begin : col
      initial bit_out = bit'(i ^ j);
      assign out[i * 2 + j] = bit_out;
      bit bit_out;
    end
  end
endmodule

// CHECK-LABEL: moore.module @MatrixInterfaceDirect(
module MatrixInterfaceDirect(output logic out);
  initial bus.sig = late;
  assign out = bus.sig;
  MatrixWriteIf bus(clk);
  logic clk;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixInterfacePortLateClock(
module MatrixInterfacePortLateClock(output logic out);
  assign out = bus.sig;
  MatrixIf bus(clk);
  logic clk;
endmodule

// CHECK-LABEL: moore.module @MatrixInterfaceInitializer(
module MatrixInterfaceInitializer(output logic out);
  logic captured = bus.sig;
  assign out = captured;
  MatrixIf bus(clk);
  logic clk;
endmodule

// CHECK-LABEL: moore.module @MatrixInterfaceDataSlice(
module MatrixInterfaceDataSlice(output logic out);
  initial bus.data[index] = late;
  assign out = bus.data[index];
  MatrixIf bus(clk);
  logic clk;
  int index = 2;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixInterfaceGenerate(
module MatrixInterfaceGenerate(output logic out);
  if (1) begin : g
    initial bus.sig = late;
    assign out = bus.sig;
    MatrixWriteIf bus(clk);
    logic clk;
    logic late;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixInterfaceLoopGenerate(
module MatrixInterfaceLoopGenerate(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    initial bus.sig = bit'(i);
    assign out[i] = bus.sig;
    MatrixWriteIf bus(clk);
    logic clk;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixHierRead(
module MatrixHierRead(output logic out);
  initial out = u.bit_value;
  MatrixReadableStorage u(.bit_value(storage_bit), .data(storage_data));
  int storage_bit;
  logic [3:0] storage_data;
endmodule

// CHECK-LABEL: moore.module @MatrixHierWrite(
module MatrixHierWrite(output logic out);
  initial u.bit_value = late;
  assign out = u.bit_value;
  MatrixStorageBit u();
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixHierDataSlice(
module MatrixHierDataSlice(output logic out);
  initial u.data[index] = late;
  assign out = u.data[index];
  MatrixStorageData u();
  int index = 1;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixHierInGenerate(
module MatrixHierInGenerate(output logic out);
  if (1) begin : g
    initial out = u.bit_value;
    MatrixReadableStorage u(.bit_value(storage_bit), .data(storage_data));
    int storage_bit;
    logic [3:0] storage_data;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixHierLoopGenerate(
module MatrixHierLoopGenerate(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    assign out[i] = u.bit_value;
    MatrixReadableStorage u(.bit_value(storage_bit), .data(storage_data));
    int storage_bit;
    logic [3:0] storage_data;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixInstanceInputLate(
module MatrixInstanceInputLate(output logic out);
  assign out = u.out;
  MatrixLeaf u(.in(late));
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixInstanceInGenerateInputLate(
module MatrixInstanceInGenerateInputLate(output logic out);
  if (1) begin : g
    assign out = u.out;
    MatrixLeaf u(.in(late));
    logic late;
  end
endmodule

// CHECK-LABEL: moore.module @MatrixMixedInterfaceAndInstance(
module MatrixMixedInterfaceAndInstance(output logic out);
  initial begin
    bus.sig = late;
    holder.bit_value = bus.sig;
  end
  assign out = holder.bit_value;
  MatrixWriteIf bus(clk);
  MatrixStorageBit holder();
  logic clk;
  logic late;
endmodule

// CHECK-LABEL: moore.module @MatrixMixedGenerateInterfaceAndInstance(
module MatrixMixedGenerateInterfaceAndInstance(output logic out);
  if (1) begin : g
    initial begin
      bus.sig = late;
      holder.bit_value = bus.sig;
    end
    assign out = holder.bit_value;
    MatrixWriteIf bus(clk);
    MatrixStorageBit holder();
    logic clk;
    logic late;
  end
endmodule
