`default_nettype none
`include "helper.sv"
module readA(
//Outputs.

//Inputs.

//MemrefType : port = r.
output reg[7:0] v0_addr,
output wire v0_rd_en,
input wire[31:0] v0_rd_data,
//MemrefType : port = w.
output reg[3:0] v1_addr[15:0],
output wire v1_wr_en[15:0],
output reg[31:0] v1_wr_data[15:0],
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

wire v0_addr_valid  [15:0] ;
wire [7:0] v0_addr_input  [15:0];
 always@(*) begin
if(v0_addr_valid[0] )
v0_addr = v0_addr_input[0];
else if (v0_addr_valid[1])
v0_addr = v0_addr_input[1];
else if (v0_addr_valid[2])
v0_addr = v0_addr_input[2];
else if (v0_addr_valid[3])
v0_addr = v0_addr_input[3];
else if (v0_addr_valid[4])
v0_addr = v0_addr_input[4];
else if (v0_addr_valid[5])
v0_addr = v0_addr_input[5];
else if (v0_addr_valid[6])
v0_addr = v0_addr_input[6];
else if (v0_addr_valid[7])
v0_addr = v0_addr_input[7];
else if (v0_addr_valid[8])
v0_addr = v0_addr_input[8];
else if (v0_addr_valid[9])
v0_addr = v0_addr_input[9];
else if (v0_addr_valid[10])
v0_addr = v0_addr_input[10];
else if (v0_addr_valid[11])
v0_addr = v0_addr_input[11];
else if (v0_addr_valid[12])
v0_addr = v0_addr_input[12];
else if (v0_addr_valid[13])
v0_addr = v0_addr_input[13];
else if (v0_addr_valid[14])
v0_addr = v0_addr_input[14];
else if (v0_addr_valid[15])
v0_addr = v0_addr_input[15];
else
 v0_addr = 'x;
end

wire [15:0] v0_rd_en_input ;
assign v0_rd_en  =| v0_rd_en_input ;


wire v1_addr_valid [15:0] [0:0] ;
wire [3:0] v1_addr_input [15:0] [0:0];
 generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
always@(*) begin
if(v1_addr_valid[i0][0] )
v1_addr[i0] = v1_addr_input[i0][0];
else
 v1_addr[i0] = 'x;
end
end
endgenerate

wire [0:0] v1_wr_en_input [15:0];
generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
assign v1_wr_en [i0] =| v1_wr_en_input [i0];
end
endgenerate
wire v1_wr_data_valid [15:0] [0:0] ;
wire [31:0] v1_wr_data_input [15:0] [0:0];
 generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
always@(*) begin
if(v1_wr_data_valid[i0][0] )
v1_wr_data[i0] = v1_wr_data_input[i0][0];
else
 v1_wr_data[i0] = 'x;
end
end
endgenerate


//printTimeOffset
reg tstartdelay[0:0] = '{default:0} ;
always@(*) tstartdelay[0] <= tstart;
generate
genvar i3;

for(i3 = 1; i3<= 0; i3= i3 + 1) begin
always@(posedge clk) begin
tstartdelay[i3] <= tstartdelay[i3-1];
end
end
endgenerate


//ConstantOp at loc("test/HIR/matmul.mlir":5:8)
//constant v4 = 1'd0;

//ConstantOp at loc("test/HIR/matmul.mlir":6:8)
//constant v5 = 1'd1;

//ConstantOp at loc("test/HIR/matmul.mlir":7:8)
//constant [1:0] v6 = 2'd2;

//ConstantOp at loc("test/HIR/matmul.mlir":8:8)
//constant [2:0] v7 = 3'd4;

//ConstantOp at loc("test/HIR/matmul.mlir":9:9)
//constant [4:0] v8 = 5'd16;

//ForOp at loc("test/HIR/matmul.mlir":12:3)

//{ Loop9

reg[31:0] idx9 ;
reg[4:0] ub9 ;
reg[0:0] step9 ;
wire tloop_in9;
reg tloop9;
reg tfinish9;
always@(posedge clk) begin
 if(/*tstart=*/ tstartdelay[0]) begin
   idx9 <= /*v4=*/ 1'd0; //lower bound.
   step9 <= /*v5=*/ 1'd1;
   ub9 <= /*v8=*/ 5'd16;
   tloop9 <= (/*v8=*/ 5'd16 > /*v4=*/ 1'd0);
   tfinish9 <=!(/*v8=*/ 5'd16 > /*v4=*/ 1'd0);
 end
 else if (tloop_in9) begin
   idx9 <= idx9 + step9; //increment
   tloop9 <= (idx9 + step9) < ub9;
   tfinish9 <= !((idx9 + step9) < ub9);
 end
 else begin
   tloop9 <= 1'b0;
   tfinish9 <= 1'b0;
 end
end
//Loop9 body
//printTimeOffset
reg tloop9delay[1:0] = '{default:0} ;
always@(*) tloop9delay[0] <= tloop9;
generate
genvar i10;

for(i10 = 1; i10<= 1; i10= i10 + 1) begin
always@(posedge clk) begin
tloop9delay[i10] <= tloop9delay[i10-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":13:13)

//{ Unrolled body 0 of loop11.
//DEBUG: /*idx11=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop12 = tloop9delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[0] = tloop9delay[0];
assign v0_addr_input[0] = {idx9[3:0], /*idx11=*/ 4'd0};
wire[31:0] v13 = v0_rd_data;
assign v0_rd_en_input[0] = tloop9delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg15[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg15[0] <= idx9;
always@(posedge clk) shiftreg15[/*v5=*/ 1:1] <= shiftreg15[/*v5=*/ 0:0];
wire [31:0] v14 = shiftreg15[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 0][0] = tloop9delay[1];
assign v1_addr_input[/*idx11=*/ 0][0] = {v14[3:0]};
assign v1_wr_en_input[/*idx11=*/ 0][0] = tloop9delay[1];
assign v1_wr_data_valid[/*idx11=*/ 0][0] = tloop9delay[1];
assign v1_wr_data_input[/*idx11=*/ 0][0] = v13;


//TerminatorOp

//} Unrolled body 0 of loop11.
//DEBUG: /*idx11=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop11.
//DEBUG: /*idx11=*/ 1'd1, expected 1
//printTimeOffset
reg tloop12delay[1:0] = '{default:0} ;
always@(*) tloop12delay[0] <= tloop12;
generate
genvar i17;

for(i17 = 1; i17<= 1; i17= i17 + 1) begin
always@(posedge clk) begin
tloop12delay[i17] <= tloop12delay[i17-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop16 = tloop12delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[1] = tloop12delay[0];
assign v0_addr_input[1] = {idx9[3:0], /*idx11=*/ 4'd1};
wire[31:0] v18 = v0_rd_data;
assign v0_rd_en_input[1] = tloop12delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg20[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg20[0] <= idx9;
always@(posedge clk) shiftreg20[/*v5=*/ 1:1] <= shiftreg20[/*v5=*/ 0:0];
wire [31:0] v19 = shiftreg20[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 1][0] = tloop12delay[1];
assign v1_addr_input[/*idx11=*/ 1][0] = {v19[3:0]};
assign v1_wr_en_input[/*idx11=*/ 1][0] = tloop12delay[1];
assign v1_wr_data_valid[/*idx11=*/ 1][0] = tloop12delay[1];
assign v1_wr_data_input[/*idx11=*/ 1][0] = v18;


//TerminatorOp

//} Unrolled body 1 of loop11.
//DEBUG: /*idx11=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop11.
//DEBUG: /*idx11=*/ 2'd2, expected 2
//printTimeOffset
reg tloop16delay[1:0] = '{default:0} ;
always@(*) tloop16delay[0] <= tloop16;
generate
genvar i22;

for(i22 = 1; i22<= 1; i22= i22 + 1) begin
always@(posedge clk) begin
tloop16delay[i22] <= tloop16delay[i22-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop21 = tloop16delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[2] = tloop16delay[0];
assign v0_addr_input[2] = {idx9[3:0], /*idx11=*/ 4'd2};
wire[31:0] v23 = v0_rd_data;
assign v0_rd_en_input[2] = tloop16delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg25[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg25[0] <= idx9;
always@(posedge clk) shiftreg25[/*v5=*/ 1:1] <= shiftreg25[/*v5=*/ 0:0];
wire [31:0] v24 = shiftreg25[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 2][0] = tloop16delay[1];
assign v1_addr_input[/*idx11=*/ 2][0] = {v24[3:0]};
assign v1_wr_en_input[/*idx11=*/ 2][0] = tloop16delay[1];
assign v1_wr_data_valid[/*idx11=*/ 2][0] = tloop16delay[1];
assign v1_wr_data_input[/*idx11=*/ 2][0] = v23;


//TerminatorOp

//} Unrolled body 2 of loop11.
//DEBUG: /*idx11=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop11.
//DEBUG: /*idx11=*/ 2'd3, expected 3
//printTimeOffset
reg tloop21delay[1:0] = '{default:0} ;
always@(*) tloop21delay[0] <= tloop21;
generate
genvar i27;

for(i27 = 1; i27<= 1; i27= i27 + 1) begin
always@(posedge clk) begin
tloop21delay[i27] <= tloop21delay[i27-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop26 = tloop21delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[3] = tloop21delay[0];
assign v0_addr_input[3] = {idx9[3:0], /*idx11=*/ 4'd3};
wire[31:0] v28 = v0_rd_data;
assign v0_rd_en_input[3] = tloop21delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg30[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg30[0] <= idx9;
always@(posedge clk) shiftreg30[/*v5=*/ 1:1] <= shiftreg30[/*v5=*/ 0:0];
wire [31:0] v29 = shiftreg30[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 3][0] = tloop21delay[1];
assign v1_addr_input[/*idx11=*/ 3][0] = {v29[3:0]};
assign v1_wr_en_input[/*idx11=*/ 3][0] = tloop21delay[1];
assign v1_wr_data_valid[/*idx11=*/ 3][0] = tloop21delay[1];
assign v1_wr_data_input[/*idx11=*/ 3][0] = v28;


//TerminatorOp

//} Unrolled body 3 of loop11.
//DEBUG: /*idx11=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop11.
//DEBUG: /*idx11=*/ 3'd4, expected 4
//printTimeOffset
reg tloop26delay[1:0] = '{default:0} ;
always@(*) tloop26delay[0] <= tloop26;
generate
genvar i32;

for(i32 = 1; i32<= 1; i32= i32 + 1) begin
always@(posedge clk) begin
tloop26delay[i32] <= tloop26delay[i32-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop31 = tloop26delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[4] = tloop26delay[0];
assign v0_addr_input[4] = {idx9[3:0], /*idx11=*/ 4'd4};
wire[31:0] v33 = v0_rd_data;
assign v0_rd_en_input[4] = tloop26delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg35[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg35[0] <= idx9;
always@(posedge clk) shiftreg35[/*v5=*/ 1:1] <= shiftreg35[/*v5=*/ 0:0];
wire [31:0] v34 = shiftreg35[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 4][0] = tloop26delay[1];
assign v1_addr_input[/*idx11=*/ 4][0] = {v34[3:0]};
assign v1_wr_en_input[/*idx11=*/ 4][0] = tloop26delay[1];
assign v1_wr_data_valid[/*idx11=*/ 4][0] = tloop26delay[1];
assign v1_wr_data_input[/*idx11=*/ 4][0] = v33;


//TerminatorOp

//} Unrolled body 4 of loop11.
//DEBUG: /*idx11=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop11.
//DEBUG: /*idx11=*/ 3'd5, expected 5
//printTimeOffset
reg tloop31delay[1:0] = '{default:0} ;
always@(*) tloop31delay[0] <= tloop31;
generate
genvar i37;

for(i37 = 1; i37<= 1; i37= i37 + 1) begin
always@(posedge clk) begin
tloop31delay[i37] <= tloop31delay[i37-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop36 = tloop31delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[5] = tloop31delay[0];
assign v0_addr_input[5] = {idx9[3:0], /*idx11=*/ 4'd5};
wire[31:0] v38 = v0_rd_data;
assign v0_rd_en_input[5] = tloop31delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg40[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg40[0] <= idx9;
always@(posedge clk) shiftreg40[/*v5=*/ 1:1] <= shiftreg40[/*v5=*/ 0:0];
wire [31:0] v39 = shiftreg40[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 5][0] = tloop31delay[1];
assign v1_addr_input[/*idx11=*/ 5][0] = {v39[3:0]};
assign v1_wr_en_input[/*idx11=*/ 5][0] = tloop31delay[1];
assign v1_wr_data_valid[/*idx11=*/ 5][0] = tloop31delay[1];
assign v1_wr_data_input[/*idx11=*/ 5][0] = v38;


//TerminatorOp

//} Unrolled body 5 of loop11.
//DEBUG: /*idx11=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop11.
//DEBUG: /*idx11=*/ 3'd6, expected 6
//printTimeOffset
reg tloop36delay[1:0] = '{default:0} ;
always@(*) tloop36delay[0] <= tloop36;
generate
genvar i42;

for(i42 = 1; i42<= 1; i42= i42 + 1) begin
always@(posedge clk) begin
tloop36delay[i42] <= tloop36delay[i42-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop41 = tloop36delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[6] = tloop36delay[0];
assign v0_addr_input[6] = {idx9[3:0], /*idx11=*/ 4'd6};
wire[31:0] v43 = v0_rd_data;
assign v0_rd_en_input[6] = tloop36delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg45[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg45[0] <= idx9;
always@(posedge clk) shiftreg45[/*v5=*/ 1:1] <= shiftreg45[/*v5=*/ 0:0];
wire [31:0] v44 = shiftreg45[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 6][0] = tloop36delay[1];
assign v1_addr_input[/*idx11=*/ 6][0] = {v44[3:0]};
assign v1_wr_en_input[/*idx11=*/ 6][0] = tloop36delay[1];
assign v1_wr_data_valid[/*idx11=*/ 6][0] = tloop36delay[1];
assign v1_wr_data_input[/*idx11=*/ 6][0] = v43;


//TerminatorOp

//} Unrolled body 6 of loop11.
//DEBUG: /*idx11=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop11.
//DEBUG: /*idx11=*/ 3'd7, expected 7
//printTimeOffset
reg tloop41delay[1:0] = '{default:0} ;
always@(*) tloop41delay[0] <= tloop41;
generate
genvar i47;

for(i47 = 1; i47<= 1; i47= i47 + 1) begin
always@(posedge clk) begin
tloop41delay[i47] <= tloop41delay[i47-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop46 = tloop41delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[7] = tloop41delay[0];
assign v0_addr_input[7] = {idx9[3:0], /*idx11=*/ 4'd7};
wire[31:0] v48 = v0_rd_data;
assign v0_rd_en_input[7] = tloop41delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg50[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg50[0] <= idx9;
always@(posedge clk) shiftreg50[/*v5=*/ 1:1] <= shiftreg50[/*v5=*/ 0:0];
wire [31:0] v49 = shiftreg50[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 7][0] = tloop41delay[1];
assign v1_addr_input[/*idx11=*/ 7][0] = {v49[3:0]};
assign v1_wr_en_input[/*idx11=*/ 7][0] = tloop41delay[1];
assign v1_wr_data_valid[/*idx11=*/ 7][0] = tloop41delay[1];
assign v1_wr_data_input[/*idx11=*/ 7][0] = v48;


//TerminatorOp

//} Unrolled body 7 of loop11.
//DEBUG: /*idx11=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop11.
//DEBUG: /*idx11=*/ 4'd8, expected 8
//printTimeOffset
reg tloop46delay[1:0] = '{default:0} ;
always@(*) tloop46delay[0] <= tloop46;
generate
genvar i52;

for(i52 = 1; i52<= 1; i52= i52 + 1) begin
always@(posedge clk) begin
tloop46delay[i52] <= tloop46delay[i52-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop51 = tloop46delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[8] = tloop46delay[0];
assign v0_addr_input[8] = {idx9[3:0], /*idx11=*/ 4'd8};
wire[31:0] v53 = v0_rd_data;
assign v0_rd_en_input[8] = tloop46delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg55[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg55[0] <= idx9;
always@(posedge clk) shiftreg55[/*v5=*/ 1:1] <= shiftreg55[/*v5=*/ 0:0];
wire [31:0] v54 = shiftreg55[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 8][0] = tloop46delay[1];
assign v1_addr_input[/*idx11=*/ 8][0] = {v54[3:0]};
assign v1_wr_en_input[/*idx11=*/ 8][0] = tloop46delay[1];
assign v1_wr_data_valid[/*idx11=*/ 8][0] = tloop46delay[1];
assign v1_wr_data_input[/*idx11=*/ 8][0] = v53;


//TerminatorOp

//} Unrolled body 8 of loop11.
//DEBUG: /*idx11=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop11.
//DEBUG: /*idx11=*/ 4'd9, expected 9
//printTimeOffset
reg tloop51delay[1:0] = '{default:0} ;
always@(*) tloop51delay[0] <= tloop51;
generate
genvar i57;

for(i57 = 1; i57<= 1; i57= i57 + 1) begin
always@(posedge clk) begin
tloop51delay[i57] <= tloop51delay[i57-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop56 = tloop51delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[9] = tloop51delay[0];
assign v0_addr_input[9] = {idx9[3:0], /*idx11=*/ 4'd9};
wire[31:0] v58 = v0_rd_data;
assign v0_rd_en_input[9] = tloop51delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg60[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg60[0] <= idx9;
always@(posedge clk) shiftreg60[/*v5=*/ 1:1] <= shiftreg60[/*v5=*/ 0:0];
wire [31:0] v59 = shiftreg60[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 9][0] = tloop51delay[1];
assign v1_addr_input[/*idx11=*/ 9][0] = {v59[3:0]};
assign v1_wr_en_input[/*idx11=*/ 9][0] = tloop51delay[1];
assign v1_wr_data_valid[/*idx11=*/ 9][0] = tloop51delay[1];
assign v1_wr_data_input[/*idx11=*/ 9][0] = v58;


//TerminatorOp

//} Unrolled body 9 of loop11.
//DEBUG: /*idx11=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop11.
//DEBUG: /*idx11=*/ 4'd10, expected 10
//printTimeOffset
reg tloop56delay[1:0] = '{default:0} ;
always@(*) tloop56delay[0] <= tloop56;
generate
genvar i62;

for(i62 = 1; i62<= 1; i62= i62 + 1) begin
always@(posedge clk) begin
tloop56delay[i62] <= tloop56delay[i62-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop61 = tloop56delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[10] = tloop56delay[0];
assign v0_addr_input[10] = {idx9[3:0], /*idx11=*/ 4'd10};
wire[31:0] v63 = v0_rd_data;
assign v0_rd_en_input[10] = tloop56delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg65[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg65[0] <= idx9;
always@(posedge clk) shiftreg65[/*v5=*/ 1:1] <= shiftreg65[/*v5=*/ 0:0];
wire [31:0] v64 = shiftreg65[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 10][0] = tloop56delay[1];
assign v1_addr_input[/*idx11=*/ 10][0] = {v64[3:0]};
assign v1_wr_en_input[/*idx11=*/ 10][0] = tloop56delay[1];
assign v1_wr_data_valid[/*idx11=*/ 10][0] = tloop56delay[1];
assign v1_wr_data_input[/*idx11=*/ 10][0] = v63;


//TerminatorOp

//} Unrolled body 10 of loop11.
//DEBUG: /*idx11=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop11.
//DEBUG: /*idx11=*/ 4'd11, expected 11
//printTimeOffset
reg tloop61delay[1:0] = '{default:0} ;
always@(*) tloop61delay[0] <= tloop61;
generate
genvar i67;

for(i67 = 1; i67<= 1; i67= i67 + 1) begin
always@(posedge clk) begin
tloop61delay[i67] <= tloop61delay[i67-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop66 = tloop61delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[11] = tloop61delay[0];
assign v0_addr_input[11] = {idx9[3:0], /*idx11=*/ 4'd11};
wire[31:0] v68 = v0_rd_data;
assign v0_rd_en_input[11] = tloop61delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg70[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg70[0] <= idx9;
always@(posedge clk) shiftreg70[/*v5=*/ 1:1] <= shiftreg70[/*v5=*/ 0:0];
wire [31:0] v69 = shiftreg70[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 11][0] = tloop61delay[1];
assign v1_addr_input[/*idx11=*/ 11][0] = {v69[3:0]};
assign v1_wr_en_input[/*idx11=*/ 11][0] = tloop61delay[1];
assign v1_wr_data_valid[/*idx11=*/ 11][0] = tloop61delay[1];
assign v1_wr_data_input[/*idx11=*/ 11][0] = v68;


//TerminatorOp

//} Unrolled body 11 of loop11.
//DEBUG: /*idx11=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop11.
//DEBUG: /*idx11=*/ 4'd12, expected 12
//printTimeOffset
reg tloop66delay[1:0] = '{default:0} ;
always@(*) tloop66delay[0] <= tloop66;
generate
genvar i72;

for(i72 = 1; i72<= 1; i72= i72 + 1) begin
always@(posedge clk) begin
tloop66delay[i72] <= tloop66delay[i72-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop71 = tloop66delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[12] = tloop66delay[0];
assign v0_addr_input[12] = {idx9[3:0], /*idx11=*/ 4'd12};
wire[31:0] v73 = v0_rd_data;
assign v0_rd_en_input[12] = tloop66delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg75[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg75[0] <= idx9;
always@(posedge clk) shiftreg75[/*v5=*/ 1:1] <= shiftreg75[/*v5=*/ 0:0];
wire [31:0] v74 = shiftreg75[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 12][0] = tloop66delay[1];
assign v1_addr_input[/*idx11=*/ 12][0] = {v74[3:0]};
assign v1_wr_en_input[/*idx11=*/ 12][0] = tloop66delay[1];
assign v1_wr_data_valid[/*idx11=*/ 12][0] = tloop66delay[1];
assign v1_wr_data_input[/*idx11=*/ 12][0] = v73;


//TerminatorOp

//} Unrolled body 12 of loop11.
//DEBUG: /*idx11=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop11.
//DEBUG: /*idx11=*/ 4'd13, expected 13
//printTimeOffset
reg tloop71delay[1:0] = '{default:0} ;
always@(*) tloop71delay[0] <= tloop71;
generate
genvar i77;

for(i77 = 1; i77<= 1; i77= i77 + 1) begin
always@(posedge clk) begin
tloop71delay[i77] <= tloop71delay[i77-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop76 = tloop71delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[13] = tloop71delay[0];
assign v0_addr_input[13] = {idx9[3:0], /*idx11=*/ 4'd13};
wire[31:0] v78 = v0_rd_data;
assign v0_rd_en_input[13] = tloop71delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg80[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg80[0] <= idx9;
always@(posedge clk) shiftreg80[/*v5=*/ 1:1] <= shiftreg80[/*v5=*/ 0:0];
wire [31:0] v79 = shiftreg80[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 13][0] = tloop71delay[1];
assign v1_addr_input[/*idx11=*/ 13][0] = {v79[3:0]};
assign v1_wr_en_input[/*idx11=*/ 13][0] = tloop71delay[1];
assign v1_wr_data_valid[/*idx11=*/ 13][0] = tloop71delay[1];
assign v1_wr_data_input[/*idx11=*/ 13][0] = v78;


//TerminatorOp

//} Unrolled body 13 of loop11.
//DEBUG: /*idx11=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop11.
//DEBUG: /*idx11=*/ 4'd14, expected 14
//printTimeOffset
reg tloop76delay[1:0] = '{default:0} ;
always@(*) tloop76delay[0] <= tloop76;
generate
genvar i82;

for(i82 = 1; i82<= 1; i82= i82 + 1) begin
always@(posedge clk) begin
tloop76delay[i82] <= tloop76delay[i82-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop81 = tloop76delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[14] = tloop76delay[0];
assign v0_addr_input[14] = {idx9[3:0], /*idx11=*/ 4'd14};
wire[31:0] v83 = v0_rd_data;
assign v0_rd_en_input[14] = tloop76delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg85[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg85[0] <= idx9;
always@(posedge clk) shiftreg85[/*v5=*/ 1:1] <= shiftreg85[/*v5=*/ 0:0];
wire [31:0] v84 = shiftreg85[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 14][0] = tloop76delay[1];
assign v1_addr_input[/*idx11=*/ 14][0] = {v84[3:0]};
assign v1_wr_en_input[/*idx11=*/ 14][0] = tloop76delay[1];
assign v1_wr_data_valid[/*idx11=*/ 14][0] = tloop76delay[1];
assign v1_wr_data_input[/*idx11=*/ 14][0] = v83;


//TerminatorOp

//} Unrolled body 14 of loop11.
//DEBUG: /*idx11=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop11.
//DEBUG: /*idx11=*/ 4'd15, expected 15
//printTimeOffset
reg tloop81delay[1:0] = '{default:0} ;
always@(*) tloop81delay[0] <= tloop81;
generate
genvar i87;

for(i87 = 1; i87<= 1; i87= i87 + 1) begin
always@(posedge clk) begin
tloop81delay[i87] <= tloop81delay[i87-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":14:7)
wire tloop86 = tloop81delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":16:13)
assign v0_addr_valid[15] = tloop81delay[0];
assign v0_addr_input[15] = {idx9[3:0], /*idx11=*/ 4'd15};
wire[31:0] v88 = v0_rd_data;
assign v0_rd_en_input[15] = tloop81delay[0];


//DelayOp at loc("test/HIR/matmul.mlir":17:13)
reg[31:0]shiftreg90[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg90[0] <= idx9;
always@(posedge clk) shiftreg90[/*v5=*/ 1:1] <= shiftreg90[/*v5=*/ 0:0];
wire [31:0] v89 = shiftreg90[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/matmul.mlir":18:7)
assign v1_addr_valid[/*idx11=*/ 15][0] = tloop81delay[1];
assign v1_addr_input[/*idx11=*/ 15][0] = {v89[3:0]};
assign v1_wr_en_input[/*idx11=*/ 15][0] = tloop81delay[1];
assign v1_wr_data_valid[/*idx11=*/ 15][0] = tloop81delay[1];
assign v1_wr_data_input[/*idx11=*/ 15][0] = v88;


//TerminatorOp

//} Unrolled body 15 of loop11.
//DEBUG: /*idx11=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t91;
assign t91 = tloop86;
//printTimeOffset
reg t91delay[0:0] = '{default:0} ;
always@(*) t91delay[0] <= t91;
generate
genvar i92;

for(i92 = 1; i92<= 0; i92= i92 + 1) begin
always@(posedge clk) begin
t91delay[i92] <= t91delay[i92-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":20:5)
assign tloop_in9 = t91delay[0];

//TerminatorOp

//} Loop9
//printTimeOffset
reg tfinish9delay[0:0] = '{default:0} ;
always@(*) tfinish9delay[0] <= tfinish9;
generate
genvar i93;

for(i93 = 1; i93<= 0; i93= i93 + 1) begin
always@(posedge clk) begin
tfinish9delay[i93] <= tfinish9delay[i93-1];
end
end
endgenerate


//ReturnOp at loc("test/HIR/matmul.mlir":22:3)
endmodule
module readB(
//Outputs.

output wire[0:0] out0
//Inputs.
,
//MemrefType : port = r.
output reg[7:0] v0_addr,
output wire v0_rd_en,
input wire[31:0] v0_rd_data,
//MemrefType : port = w.
output wire v1_wr_en[15:0][15:0],
output reg[31:0] v1_wr_data[15:0][15:0],
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

wire v0_addr_valid  [0:0] ;
wire [7:0] v0_addr_input  [0:0];
 always@(*) begin
if(v0_addr_valid[0] )
v0_addr = v0_addr_input[0];
else
 v0_addr = 'x;
end

wire [0:0] v0_rd_en_input ;
assign v0_rd_en  =| v0_rd_en_input ;


wire [0:0] v1_wr_en_input [15:0][15:0];
generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
for(genvar i1 = 0; i1 < 16;i1=i1 + 1) begin
assign v1_wr_en [i0][i1] =| v1_wr_en_input [i0][i1];
end
end
endgenerate
wire v1_wr_data_valid [15:0][15:0] [0:0] ;
wire [31:0] v1_wr_data_input [15:0][15:0] [0:0];
 generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
for(genvar i1 = 0; i1 < 16;i1=i1 + 1) begin
always@(*) begin
if(v1_wr_data_valid[i0][i1][0] )
v1_wr_data[i0][i1] = v1_wr_data_input[i0][i1][0];
else
 v1_wr_data[i0][i1] = 'x;
end
end
end
endgenerate


//printTimeOffset
reg tstartdelay[0:0] = '{default:0} ;
always@(*) tstartdelay[0] <= tstart;
generate
genvar i3;

for(i3 = 1; i3<= 0; i3= i3 + 1) begin
always@(posedge clk) begin
tstartdelay[i3] <= tstartdelay[i3-1];
end
end
endgenerate


//ConstantOp at loc("test/HIR/matmul.mlir":29:8)
//constant v4 = 1'd0;

//ConstantOp at loc("test/HIR/matmul.mlir":30:8)
//constant v5 = 1'd1;

//ConstantOp at loc("test/HIR/matmul.mlir":31:8)
//constant [1:0] v6 = 2'd2;

//ConstantOp at loc("test/HIR/matmul.mlir":32:8)
//constant [1:0] v7 = 2'd3;

//ConstantOp at loc("test/HIR/matmul.mlir":33:8)
//constant [2:0] v8 = 3'd4;

//ConstantOp at loc("test/HIR/matmul.mlir":34:9)
//constant [4:0] v9 = 5'd16;

//AllocOp at loc("test/HIR/matmul.mlir":37:18)
//strMemrefInstDecl
wire v10_rd_en[0:0];
logic[31:0] v10_rd_data[0:0];
//strMemrefSelDecl
wire [255:0] v10_rd_en_input [0:0];
generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
assign v10_rd_en [i0] =| v10_rd_en_input [i0];
end
endgenerate


//strMemrefInstDecl
 wire v11_wr_en[0:0];
reg[31:0] v11_wr_data[0:0];
//strMemrefSelDecl
wire [0:0] v11_wr_en_input [0:0];
generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
assign v11_wr_en [i0] =| v11_wr_en_input [i0];
end
endgenerate
wire v11_wr_data_valid [0:0] [0:0] ;
wire [31:0] v11_wr_data_input [0:0] [0:0];
 generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
always@(*) begin
if(v11_wr_data_valid[i0][0] )
v11_wr_data[i0] = v11_wr_data_input[i0][0];
else
 v11_wr_data[i0] = 'x;
end
end
endgenerate



 //Instantiate Memory.
for(genvar i0= 0; i0<1;i0+=1) begin
always@(posedge clk) begin
  if(v11_wr_en[i0]) v10_rd_data[i0] <= v11_wr_data[i0];
end
end

//ForOp at loc("test/HIR/matmul.mlir":39:3)

//{ Loop12

reg[31:0] idx12 ;
reg[4:0] ub12 ;
reg[0:0] step12 ;
wire tloop_in12;
reg tloop12;
reg tfinish12;
always@(posedge clk) begin
 if(/*tstart=*/ tstartdelay[0]) begin
   idx12 <= /*v4=*/ 1'd0; //lower bound.
   step12 <= /*v5=*/ 1'd1;
   ub12 <= /*v9=*/ 5'd16;
   tloop12 <= (/*v9=*/ 5'd16 > /*v4=*/ 1'd0);
   tfinish12 <=!(/*v9=*/ 5'd16 > /*v4=*/ 1'd0);
 end
 else if (tloop_in12) begin
   idx12 <= idx12 + step12; //increment
   tloop12 <= (idx12 + step12) < ub12;
   tfinish12 <= !((idx12 + step12) < ub12);
 end
 else begin
   tloop12 <= 1'b0;
   tfinish12 <= 1'b0;
 end
end
//Loop12 body
//printTimeOffset
reg tloop12delay[0:0] = '{default:0} ;
always@(*) tloop12delay[0] <= tloop12;
generate
genvar i13;

for(i13 = 1; i13<= 0; i13= i13 + 1) begin
always@(posedge clk) begin
tloop12delay[i13] <= tloop12delay[i13-1];
end
end
endgenerate


//ForOp at loc("test/HIR/matmul.mlir":40:14)

//{ Loop14

reg[31:0] idx14 ;
reg[4:0] ub14 ;
reg[0:0] step14 ;
wire tloop_in14;
reg tloop14;
reg tfinish14;
always@(posedge clk) begin
 if(/*tstart=*/ tloop12delay[0]) begin
   idx14 <= /*v4=*/ 1'd0; //lower bound.
   step14 <= /*v5=*/ 1'd1;
   ub14 <= /*v9=*/ 5'd16;
   tloop14 <= (/*v9=*/ 5'd16 > /*v4=*/ 1'd0);
   tfinish14 <=!(/*v9=*/ 5'd16 > /*v4=*/ 1'd0);
 end
 else if (tloop_in14) begin
   idx14 <= idx14 + step14; //increment
   tloop14 <= (idx14 + step14) < ub14;
   tfinish14 <= !((idx14 + step14) < ub14);
 end
 else begin
   tloop14 <= 1'b0;
   tfinish14 <= 1'b0;
 end
end
//Loop14 body
//printTimeOffset
reg tloop14delay[1:0] = '{default:0} ;
always@(*) tloop14delay[0] <= tloop14;
generate
genvar i15;

for(i15 = 1; i15<= 1; i15= i15 + 1) begin
always@(posedge clk) begin
tloop14delay[i15] <= tloop14delay[i15-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":41:7)
assign tloop_in14 = tloop14delay[0];

//MemReadOp at loc("test/HIR/matmul.mlir":42:13)
assign v0_addr_valid[0] = tloop14delay[0];
assign v0_addr_input[0] = {idx14[3:0], idx12[3:0]};
wire[31:0] v16 = v0_rd_data;
assign v0_rd_en_input[0] = tloop14delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":43:7)
assign v11_wr_en_input[/*v4=*/ 0][0] = tloop14delay[1];
assign v11_wr_data_valid[/*v4=*/ 0][0] = tloop14delay[1];
assign v11_wr_data_input[/*v4=*/ 0][0] = v16;


//TerminatorOp

//} Loop14
//printTimeOffset
reg tfinish14delay[0:0] = '{default:0} ;
always@(*) tfinish14delay[0] <= tfinish14;
generate
genvar i17;

for(i17 = 1; i17<= 0; i17= i17 + 1) begin
always@(posedge clk) begin
tfinish14delay[i17] <= tfinish14delay[i17-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":45:5)
assign tloop_in12 = tfinish14delay[0];

//TerminatorOp

//} Loop12
//printTimeOffset
reg tfinish12delay[0:0] = '{default:0} ;
always@(*) tfinish12delay[0] <= tfinish12;
generate
genvar i18;

for(i18 = 1; i18<= 0; i18= i18 + 1) begin
always@(posedge clk) begin
tfinish12delay[i18] <= tfinish12delay[i18-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":48:9)
reg[0:0]shiftreg20[/*v7=*/ 3:0] = '{default:0};
always@(*) shiftreg20[0] <= tstart;
always@(posedge clk) shiftreg20[/*v7=*/ 3:1] <= shiftreg20[/*v7=*/ 2:0];
wire v19 = shiftreg20[/*v7=*/ 3];
//printTimeOffset
reg v19delay[0:0] = '{default:0} ;
always@(*) v19delay[0] <= v19;
generate
genvar i21;

for(i21 = 1; i21<= 0; i21= i21 + 1) begin
always@(posedge clk) begin
v19delay[i21] <= v19delay[i21-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":49:11)

//{ Unrolled body 0 of loop22.
//DEBUG: /*idx22=*/ 1'd0, expected 0

//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg25[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg25[0] <= v19;
always@(posedge clk) shiftreg25[/*v5=*/ 1:1] <= shiftreg25[/*v5=*/ 0:0];
wire v24 = shiftreg25[/*v5=*/ 1];
//printTimeOffset
reg v24delay[1:0] = '{default:0} ;
always@(*) v24delay[0] <= v24;
generate
genvar i26;

for(i26 = 1; i26<= 1; i26= i26 + 1) begin
always@(posedge clk) begin
v24delay[i26] <= v24delay[i26-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop27.
//DEBUG: /*idx27=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop28 = v24delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v29 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][0] = v24delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 0][/*idx22=*/ 0][0] = v24delay[0];
assign v1_wr_data_valid[/*idx27=*/ 0][/*idx22=*/ 0][0] = v24delay[0];
assign v1_wr_data_input[/*idx27=*/ 0][/*idx22=*/ 0][0] = v29;


//TerminatorOp

//} Unrolled body 0 of loop27.
//DEBUG: /*idx27=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop27.
//DEBUG: /*idx27=*/ 1'd1, expected 1
//printTimeOffset
reg tloop28delay[1:0] = '{default:0} ;
always@(*) tloop28delay[0] <= tloop28;
generate
genvar i31;

for(i31 = 1; i31<= 1; i31= i31 + 1) begin
always@(posedge clk) begin
tloop28delay[i31] <= tloop28delay[i31-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop30 = tloop28delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v32 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][1] = tloop28delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 1][/*idx22=*/ 0][0] = tloop28delay[0];
assign v1_wr_data_valid[/*idx27=*/ 1][/*idx22=*/ 0][0] = tloop28delay[0];
assign v1_wr_data_input[/*idx27=*/ 1][/*idx22=*/ 0][0] = v32;


//TerminatorOp

//} Unrolled body 1 of loop27.
//DEBUG: /*idx27=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop27.
//DEBUG: /*idx27=*/ 2'd2, expected 2
//printTimeOffset
reg tloop30delay[1:0] = '{default:0} ;
always@(*) tloop30delay[0] <= tloop30;
generate
genvar i34;

for(i34 = 1; i34<= 1; i34= i34 + 1) begin
always@(posedge clk) begin
tloop30delay[i34] <= tloop30delay[i34-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop33 = tloop30delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v35 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][2] = tloop30delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 2][/*idx22=*/ 0][0] = tloop30delay[0];
assign v1_wr_data_valid[/*idx27=*/ 2][/*idx22=*/ 0][0] = tloop30delay[0];
assign v1_wr_data_input[/*idx27=*/ 2][/*idx22=*/ 0][0] = v35;


//TerminatorOp

//} Unrolled body 2 of loop27.
//DEBUG: /*idx27=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop27.
//DEBUG: /*idx27=*/ 2'd3, expected 3
//printTimeOffset
reg tloop33delay[1:0] = '{default:0} ;
always@(*) tloop33delay[0] <= tloop33;
generate
genvar i37;

for(i37 = 1; i37<= 1; i37= i37 + 1) begin
always@(posedge clk) begin
tloop33delay[i37] <= tloop33delay[i37-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop36 = tloop33delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v38 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][3] = tloop33delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 3][/*idx22=*/ 0][0] = tloop33delay[0];
assign v1_wr_data_valid[/*idx27=*/ 3][/*idx22=*/ 0][0] = tloop33delay[0];
assign v1_wr_data_input[/*idx27=*/ 3][/*idx22=*/ 0][0] = v38;


//TerminatorOp

//} Unrolled body 3 of loop27.
//DEBUG: /*idx27=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop27.
//DEBUG: /*idx27=*/ 3'd4, expected 4
//printTimeOffset
reg tloop36delay[1:0] = '{default:0} ;
always@(*) tloop36delay[0] <= tloop36;
generate
genvar i40;

for(i40 = 1; i40<= 1; i40= i40 + 1) begin
always@(posedge clk) begin
tloop36delay[i40] <= tloop36delay[i40-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop39 = tloop36delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v41 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][4] = tloop36delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 4][/*idx22=*/ 0][0] = tloop36delay[0];
assign v1_wr_data_valid[/*idx27=*/ 4][/*idx22=*/ 0][0] = tloop36delay[0];
assign v1_wr_data_input[/*idx27=*/ 4][/*idx22=*/ 0][0] = v41;


//TerminatorOp

//} Unrolled body 4 of loop27.
//DEBUG: /*idx27=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop27.
//DEBUG: /*idx27=*/ 3'd5, expected 5
//printTimeOffset
reg tloop39delay[1:0] = '{default:0} ;
always@(*) tloop39delay[0] <= tloop39;
generate
genvar i43;

for(i43 = 1; i43<= 1; i43= i43 + 1) begin
always@(posedge clk) begin
tloop39delay[i43] <= tloop39delay[i43-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop42 = tloop39delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v44 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][5] = tloop39delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 5][/*idx22=*/ 0][0] = tloop39delay[0];
assign v1_wr_data_valid[/*idx27=*/ 5][/*idx22=*/ 0][0] = tloop39delay[0];
assign v1_wr_data_input[/*idx27=*/ 5][/*idx22=*/ 0][0] = v44;


//TerminatorOp

//} Unrolled body 5 of loop27.
//DEBUG: /*idx27=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop27.
//DEBUG: /*idx27=*/ 3'd6, expected 6
//printTimeOffset
reg tloop42delay[1:0] = '{default:0} ;
always@(*) tloop42delay[0] <= tloop42;
generate
genvar i46;

for(i46 = 1; i46<= 1; i46= i46 + 1) begin
always@(posedge clk) begin
tloop42delay[i46] <= tloop42delay[i46-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop45 = tloop42delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v47 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][6] = tloop42delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 6][/*idx22=*/ 0][0] = tloop42delay[0];
assign v1_wr_data_valid[/*idx27=*/ 6][/*idx22=*/ 0][0] = tloop42delay[0];
assign v1_wr_data_input[/*idx27=*/ 6][/*idx22=*/ 0][0] = v47;


//TerminatorOp

//} Unrolled body 6 of loop27.
//DEBUG: /*idx27=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop27.
//DEBUG: /*idx27=*/ 3'd7, expected 7
//printTimeOffset
reg tloop45delay[1:0] = '{default:0} ;
always@(*) tloop45delay[0] <= tloop45;
generate
genvar i49;

for(i49 = 1; i49<= 1; i49= i49 + 1) begin
always@(posedge clk) begin
tloop45delay[i49] <= tloop45delay[i49-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop48 = tloop45delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v50 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][7] = tloop45delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 7][/*idx22=*/ 0][0] = tloop45delay[0];
assign v1_wr_data_valid[/*idx27=*/ 7][/*idx22=*/ 0][0] = tloop45delay[0];
assign v1_wr_data_input[/*idx27=*/ 7][/*idx22=*/ 0][0] = v50;


//TerminatorOp

//} Unrolled body 7 of loop27.
//DEBUG: /*idx27=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop27.
//DEBUG: /*idx27=*/ 4'd8, expected 8
//printTimeOffset
reg tloop48delay[1:0] = '{default:0} ;
always@(*) tloop48delay[0] <= tloop48;
generate
genvar i52;

for(i52 = 1; i52<= 1; i52= i52 + 1) begin
always@(posedge clk) begin
tloop48delay[i52] <= tloop48delay[i52-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop51 = tloop48delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v53 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][8] = tloop48delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 8][/*idx22=*/ 0][0] = tloop48delay[0];
assign v1_wr_data_valid[/*idx27=*/ 8][/*idx22=*/ 0][0] = tloop48delay[0];
assign v1_wr_data_input[/*idx27=*/ 8][/*idx22=*/ 0][0] = v53;


//TerminatorOp

//} Unrolled body 8 of loop27.
//DEBUG: /*idx27=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop27.
//DEBUG: /*idx27=*/ 4'd9, expected 9
//printTimeOffset
reg tloop51delay[1:0] = '{default:0} ;
always@(*) tloop51delay[0] <= tloop51;
generate
genvar i55;

for(i55 = 1; i55<= 1; i55= i55 + 1) begin
always@(posedge clk) begin
tloop51delay[i55] <= tloop51delay[i55-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop54 = tloop51delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v56 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][9] = tloop51delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 9][/*idx22=*/ 0][0] = tloop51delay[0];
assign v1_wr_data_valid[/*idx27=*/ 9][/*idx22=*/ 0][0] = tloop51delay[0];
assign v1_wr_data_input[/*idx27=*/ 9][/*idx22=*/ 0][0] = v56;


//TerminatorOp

//} Unrolled body 9 of loop27.
//DEBUG: /*idx27=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop27.
//DEBUG: /*idx27=*/ 4'd10, expected 10
//printTimeOffset
reg tloop54delay[1:0] = '{default:0} ;
always@(*) tloop54delay[0] <= tloop54;
generate
genvar i58;

for(i58 = 1; i58<= 1; i58= i58 + 1) begin
always@(posedge clk) begin
tloop54delay[i58] <= tloop54delay[i58-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop57 = tloop54delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v59 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][10] = tloop54delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 10][/*idx22=*/ 0][0] = tloop54delay[0];
assign v1_wr_data_valid[/*idx27=*/ 10][/*idx22=*/ 0][0] = tloop54delay[0];
assign v1_wr_data_input[/*idx27=*/ 10][/*idx22=*/ 0][0] = v59;


//TerminatorOp

//} Unrolled body 10 of loop27.
//DEBUG: /*idx27=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop27.
//DEBUG: /*idx27=*/ 4'd11, expected 11
//printTimeOffset
reg tloop57delay[1:0] = '{default:0} ;
always@(*) tloop57delay[0] <= tloop57;
generate
genvar i61;

for(i61 = 1; i61<= 1; i61= i61 + 1) begin
always@(posedge clk) begin
tloop57delay[i61] <= tloop57delay[i61-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop60 = tloop57delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v62 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][11] = tloop57delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 11][/*idx22=*/ 0][0] = tloop57delay[0];
assign v1_wr_data_valid[/*idx27=*/ 11][/*idx22=*/ 0][0] = tloop57delay[0];
assign v1_wr_data_input[/*idx27=*/ 11][/*idx22=*/ 0][0] = v62;


//TerminatorOp

//} Unrolled body 11 of loop27.
//DEBUG: /*idx27=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop27.
//DEBUG: /*idx27=*/ 4'd12, expected 12
//printTimeOffset
reg tloop60delay[1:0] = '{default:0} ;
always@(*) tloop60delay[0] <= tloop60;
generate
genvar i64;

for(i64 = 1; i64<= 1; i64= i64 + 1) begin
always@(posedge clk) begin
tloop60delay[i64] <= tloop60delay[i64-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop63 = tloop60delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v65 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][12] = tloop60delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 12][/*idx22=*/ 0][0] = tloop60delay[0];
assign v1_wr_data_valid[/*idx27=*/ 12][/*idx22=*/ 0][0] = tloop60delay[0];
assign v1_wr_data_input[/*idx27=*/ 12][/*idx22=*/ 0][0] = v65;


//TerminatorOp

//} Unrolled body 12 of loop27.
//DEBUG: /*idx27=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop27.
//DEBUG: /*idx27=*/ 4'd13, expected 13
//printTimeOffset
reg tloop63delay[1:0] = '{default:0} ;
always@(*) tloop63delay[0] <= tloop63;
generate
genvar i67;

for(i67 = 1; i67<= 1; i67= i67 + 1) begin
always@(posedge clk) begin
tloop63delay[i67] <= tloop63delay[i67-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop66 = tloop63delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v68 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][13] = tloop63delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 13][/*idx22=*/ 0][0] = tloop63delay[0];
assign v1_wr_data_valid[/*idx27=*/ 13][/*idx22=*/ 0][0] = tloop63delay[0];
assign v1_wr_data_input[/*idx27=*/ 13][/*idx22=*/ 0][0] = v68;


//TerminatorOp

//} Unrolled body 13 of loop27.
//DEBUG: /*idx27=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop27.
//DEBUG: /*idx27=*/ 4'd14, expected 14
//printTimeOffset
reg tloop66delay[1:0] = '{default:0} ;
always@(*) tloop66delay[0] <= tloop66;
generate
genvar i70;

for(i70 = 1; i70<= 1; i70= i70 + 1) begin
always@(posedge clk) begin
tloop66delay[i70] <= tloop66delay[i70-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop69 = tloop66delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v71 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][14] = tloop66delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 14][/*idx22=*/ 0][0] = tloop66delay[0];
assign v1_wr_data_valid[/*idx27=*/ 14][/*idx22=*/ 0][0] = tloop66delay[0];
assign v1_wr_data_input[/*idx27=*/ 14][/*idx22=*/ 0][0] = v71;


//TerminatorOp

//} Unrolled body 14 of loop27.
//DEBUG: /*idx27=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop27.
//DEBUG: /*idx27=*/ 4'd15, expected 15
//printTimeOffset
reg tloop69delay[1:0] = '{default:0} ;
always@(*) tloop69delay[0] <= tloop69;
generate
genvar i73;

for(i73 = 1; i73<= 1; i73= i73 + 1) begin
always@(posedge clk) begin
tloop69delay[i73] <= tloop69delay[i73-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop72 = tloop69delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v74 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][15] = tloop69delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx27=*/ 15][/*idx22=*/ 0][0] = tloop69delay[0];
assign v1_wr_data_valid[/*idx27=*/ 15][/*idx22=*/ 0][0] = tloop69delay[0];
assign v1_wr_data_input[/*idx27=*/ 15][/*idx22=*/ 0][0] = v74;


//TerminatorOp

//} Unrolled body 15 of loop27.
//DEBUG: /*idx27=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t75;
assign t75 = tloop72;
//printTimeOffset
reg t75delay[1:0] = '{default:0} ;
always@(*) t75delay[0] <= t75;
generate
genvar i76;

for(i76 = 1; i76<= 1; i76= i76 + 1) begin
always@(posedge clk) begin
t75delay[i76] <= t75delay[i76-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop23 = t75delay[1];

//TerminatorOp

//} Unrolled body 0 of loop22.
//DEBUG: /*idx22=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop22.
//DEBUG: /*idx22=*/ 1'd1, expected 1
//printTimeOffset
reg tloop23delay[0:0] = '{default:0} ;
always@(*) tloop23delay[0] <= tloop23;
generate
genvar i78;

for(i78 = 1; i78<= 0; i78= i78 + 1) begin
always@(posedge clk) begin
tloop23delay[i78] <= tloop23delay[i78-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg80[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg80[0] <= tloop23;
always@(posedge clk) shiftreg80[/*v5=*/ 1:1] <= shiftreg80[/*v5=*/ 0:0];
wire v79 = shiftreg80[/*v5=*/ 1];
//printTimeOffset
reg v79delay[1:0] = '{default:0} ;
always@(*) v79delay[0] <= v79;
generate
genvar i81;

for(i81 = 1; i81<= 1; i81= i81 + 1) begin
always@(posedge clk) begin
v79delay[i81] <= v79delay[i81-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop82.
//DEBUG: /*idx82=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop83 = v79delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v84 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][16] = v79delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 0][/*idx22=*/ 1][0] = v79delay[0];
assign v1_wr_data_valid[/*idx82=*/ 0][/*idx22=*/ 1][0] = v79delay[0];
assign v1_wr_data_input[/*idx82=*/ 0][/*idx22=*/ 1][0] = v84;


//TerminatorOp

//} Unrolled body 0 of loop82.
//DEBUG: /*idx82=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop82.
//DEBUG: /*idx82=*/ 1'd1, expected 1
//printTimeOffset
reg tloop83delay[1:0] = '{default:0} ;
always@(*) tloop83delay[0] <= tloop83;
generate
genvar i86;

for(i86 = 1; i86<= 1; i86= i86 + 1) begin
always@(posedge clk) begin
tloop83delay[i86] <= tloop83delay[i86-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop85 = tloop83delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v87 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][17] = tloop83delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 1][/*idx22=*/ 1][0] = tloop83delay[0];
assign v1_wr_data_valid[/*idx82=*/ 1][/*idx22=*/ 1][0] = tloop83delay[0];
assign v1_wr_data_input[/*idx82=*/ 1][/*idx22=*/ 1][0] = v87;


//TerminatorOp

//} Unrolled body 1 of loop82.
//DEBUG: /*idx82=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop82.
//DEBUG: /*idx82=*/ 2'd2, expected 2
//printTimeOffset
reg tloop85delay[1:0] = '{default:0} ;
always@(*) tloop85delay[0] <= tloop85;
generate
genvar i89;

for(i89 = 1; i89<= 1; i89= i89 + 1) begin
always@(posedge clk) begin
tloop85delay[i89] <= tloop85delay[i89-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop88 = tloop85delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v90 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][18] = tloop85delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 2][/*idx22=*/ 1][0] = tloop85delay[0];
assign v1_wr_data_valid[/*idx82=*/ 2][/*idx22=*/ 1][0] = tloop85delay[0];
assign v1_wr_data_input[/*idx82=*/ 2][/*idx22=*/ 1][0] = v90;


//TerminatorOp

//} Unrolled body 2 of loop82.
//DEBUG: /*idx82=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop82.
//DEBUG: /*idx82=*/ 2'd3, expected 3
//printTimeOffset
reg tloop88delay[1:0] = '{default:0} ;
always@(*) tloop88delay[0] <= tloop88;
generate
genvar i92;

for(i92 = 1; i92<= 1; i92= i92 + 1) begin
always@(posedge clk) begin
tloop88delay[i92] <= tloop88delay[i92-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop91 = tloop88delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v93 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][19] = tloop88delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 3][/*idx22=*/ 1][0] = tloop88delay[0];
assign v1_wr_data_valid[/*idx82=*/ 3][/*idx22=*/ 1][0] = tloop88delay[0];
assign v1_wr_data_input[/*idx82=*/ 3][/*idx22=*/ 1][0] = v93;


//TerminatorOp

//} Unrolled body 3 of loop82.
//DEBUG: /*idx82=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop82.
//DEBUG: /*idx82=*/ 3'd4, expected 4
//printTimeOffset
reg tloop91delay[1:0] = '{default:0} ;
always@(*) tloop91delay[0] <= tloop91;
generate
genvar i95;

for(i95 = 1; i95<= 1; i95= i95 + 1) begin
always@(posedge clk) begin
tloop91delay[i95] <= tloop91delay[i95-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop94 = tloop91delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v96 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][20] = tloop91delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 4][/*idx22=*/ 1][0] = tloop91delay[0];
assign v1_wr_data_valid[/*idx82=*/ 4][/*idx22=*/ 1][0] = tloop91delay[0];
assign v1_wr_data_input[/*idx82=*/ 4][/*idx22=*/ 1][0] = v96;


//TerminatorOp

//} Unrolled body 4 of loop82.
//DEBUG: /*idx82=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop82.
//DEBUG: /*idx82=*/ 3'd5, expected 5
//printTimeOffset
reg tloop94delay[1:0] = '{default:0} ;
always@(*) tloop94delay[0] <= tloop94;
generate
genvar i98;

for(i98 = 1; i98<= 1; i98= i98 + 1) begin
always@(posedge clk) begin
tloop94delay[i98] <= tloop94delay[i98-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop97 = tloop94delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v99 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][21] = tloop94delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 5][/*idx22=*/ 1][0] = tloop94delay[0];
assign v1_wr_data_valid[/*idx82=*/ 5][/*idx22=*/ 1][0] = tloop94delay[0];
assign v1_wr_data_input[/*idx82=*/ 5][/*idx22=*/ 1][0] = v99;


//TerminatorOp

//} Unrolled body 5 of loop82.
//DEBUG: /*idx82=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop82.
//DEBUG: /*idx82=*/ 3'd6, expected 6
//printTimeOffset
reg tloop97delay[1:0] = '{default:0} ;
always@(*) tloop97delay[0] <= tloop97;
generate
genvar i101;

for(i101 = 1; i101<= 1; i101= i101 + 1) begin
always@(posedge clk) begin
tloop97delay[i101] <= tloop97delay[i101-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop100 = tloop97delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v102 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][22] = tloop97delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 6][/*idx22=*/ 1][0] = tloop97delay[0];
assign v1_wr_data_valid[/*idx82=*/ 6][/*idx22=*/ 1][0] = tloop97delay[0];
assign v1_wr_data_input[/*idx82=*/ 6][/*idx22=*/ 1][0] = v102;


//TerminatorOp

//} Unrolled body 6 of loop82.
//DEBUG: /*idx82=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop82.
//DEBUG: /*idx82=*/ 3'd7, expected 7
//printTimeOffset
reg tloop100delay[1:0] = '{default:0} ;
always@(*) tloop100delay[0] <= tloop100;
generate
genvar i104;

for(i104 = 1; i104<= 1; i104= i104 + 1) begin
always@(posedge clk) begin
tloop100delay[i104] <= tloop100delay[i104-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop103 = tloop100delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v105 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][23] = tloop100delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 7][/*idx22=*/ 1][0] = tloop100delay[0];
assign v1_wr_data_valid[/*idx82=*/ 7][/*idx22=*/ 1][0] = tloop100delay[0];
assign v1_wr_data_input[/*idx82=*/ 7][/*idx22=*/ 1][0] = v105;


//TerminatorOp

//} Unrolled body 7 of loop82.
//DEBUG: /*idx82=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop82.
//DEBUG: /*idx82=*/ 4'd8, expected 8
//printTimeOffset
reg tloop103delay[1:0] = '{default:0} ;
always@(*) tloop103delay[0] <= tloop103;
generate
genvar i107;

for(i107 = 1; i107<= 1; i107= i107 + 1) begin
always@(posedge clk) begin
tloop103delay[i107] <= tloop103delay[i107-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop106 = tloop103delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v108 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][24] = tloop103delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 8][/*idx22=*/ 1][0] = tloop103delay[0];
assign v1_wr_data_valid[/*idx82=*/ 8][/*idx22=*/ 1][0] = tloop103delay[0];
assign v1_wr_data_input[/*idx82=*/ 8][/*idx22=*/ 1][0] = v108;


//TerminatorOp

//} Unrolled body 8 of loop82.
//DEBUG: /*idx82=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop82.
//DEBUG: /*idx82=*/ 4'd9, expected 9
//printTimeOffset
reg tloop106delay[1:0] = '{default:0} ;
always@(*) tloop106delay[0] <= tloop106;
generate
genvar i110;

for(i110 = 1; i110<= 1; i110= i110 + 1) begin
always@(posedge clk) begin
tloop106delay[i110] <= tloop106delay[i110-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop109 = tloop106delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v111 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][25] = tloop106delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 9][/*idx22=*/ 1][0] = tloop106delay[0];
assign v1_wr_data_valid[/*idx82=*/ 9][/*idx22=*/ 1][0] = tloop106delay[0];
assign v1_wr_data_input[/*idx82=*/ 9][/*idx22=*/ 1][0] = v111;


//TerminatorOp

//} Unrolled body 9 of loop82.
//DEBUG: /*idx82=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop82.
//DEBUG: /*idx82=*/ 4'd10, expected 10
//printTimeOffset
reg tloop109delay[1:0] = '{default:0} ;
always@(*) tloop109delay[0] <= tloop109;
generate
genvar i113;

for(i113 = 1; i113<= 1; i113= i113 + 1) begin
always@(posedge clk) begin
tloop109delay[i113] <= tloop109delay[i113-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop112 = tloop109delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v114 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][26] = tloop109delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 10][/*idx22=*/ 1][0] = tloop109delay[0];
assign v1_wr_data_valid[/*idx82=*/ 10][/*idx22=*/ 1][0] = tloop109delay[0];
assign v1_wr_data_input[/*idx82=*/ 10][/*idx22=*/ 1][0] = v114;


//TerminatorOp

//} Unrolled body 10 of loop82.
//DEBUG: /*idx82=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop82.
//DEBUG: /*idx82=*/ 4'd11, expected 11
//printTimeOffset
reg tloop112delay[1:0] = '{default:0} ;
always@(*) tloop112delay[0] <= tloop112;
generate
genvar i116;

for(i116 = 1; i116<= 1; i116= i116 + 1) begin
always@(posedge clk) begin
tloop112delay[i116] <= tloop112delay[i116-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop115 = tloop112delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v117 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][27] = tloop112delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 11][/*idx22=*/ 1][0] = tloop112delay[0];
assign v1_wr_data_valid[/*idx82=*/ 11][/*idx22=*/ 1][0] = tloop112delay[0];
assign v1_wr_data_input[/*idx82=*/ 11][/*idx22=*/ 1][0] = v117;


//TerminatorOp

//} Unrolled body 11 of loop82.
//DEBUG: /*idx82=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop82.
//DEBUG: /*idx82=*/ 4'd12, expected 12
//printTimeOffset
reg tloop115delay[1:0] = '{default:0} ;
always@(*) tloop115delay[0] <= tloop115;
generate
genvar i119;

for(i119 = 1; i119<= 1; i119= i119 + 1) begin
always@(posedge clk) begin
tloop115delay[i119] <= tloop115delay[i119-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop118 = tloop115delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v120 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][28] = tloop115delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 12][/*idx22=*/ 1][0] = tloop115delay[0];
assign v1_wr_data_valid[/*idx82=*/ 12][/*idx22=*/ 1][0] = tloop115delay[0];
assign v1_wr_data_input[/*idx82=*/ 12][/*idx22=*/ 1][0] = v120;


//TerminatorOp

//} Unrolled body 12 of loop82.
//DEBUG: /*idx82=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop82.
//DEBUG: /*idx82=*/ 4'd13, expected 13
//printTimeOffset
reg tloop118delay[1:0] = '{default:0} ;
always@(*) tloop118delay[0] <= tloop118;
generate
genvar i122;

for(i122 = 1; i122<= 1; i122= i122 + 1) begin
always@(posedge clk) begin
tloop118delay[i122] <= tloop118delay[i122-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop121 = tloop118delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v123 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][29] = tloop118delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 13][/*idx22=*/ 1][0] = tloop118delay[0];
assign v1_wr_data_valid[/*idx82=*/ 13][/*idx22=*/ 1][0] = tloop118delay[0];
assign v1_wr_data_input[/*idx82=*/ 13][/*idx22=*/ 1][0] = v123;


//TerminatorOp

//} Unrolled body 13 of loop82.
//DEBUG: /*idx82=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop82.
//DEBUG: /*idx82=*/ 4'd14, expected 14
//printTimeOffset
reg tloop121delay[1:0] = '{default:0} ;
always@(*) tloop121delay[0] <= tloop121;
generate
genvar i125;

for(i125 = 1; i125<= 1; i125= i125 + 1) begin
always@(posedge clk) begin
tloop121delay[i125] <= tloop121delay[i125-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop124 = tloop121delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v126 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][30] = tloop121delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 14][/*idx22=*/ 1][0] = tloop121delay[0];
assign v1_wr_data_valid[/*idx82=*/ 14][/*idx22=*/ 1][0] = tloop121delay[0];
assign v1_wr_data_input[/*idx82=*/ 14][/*idx22=*/ 1][0] = v126;


//TerminatorOp

//} Unrolled body 14 of loop82.
//DEBUG: /*idx82=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop82.
//DEBUG: /*idx82=*/ 4'd15, expected 15
//printTimeOffset
reg tloop124delay[1:0] = '{default:0} ;
always@(*) tloop124delay[0] <= tloop124;
generate
genvar i128;

for(i128 = 1; i128<= 1; i128= i128 + 1) begin
always@(posedge clk) begin
tloop124delay[i128] <= tloop124delay[i128-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop127 = tloop124delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v129 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][31] = tloop124delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx82=*/ 15][/*idx22=*/ 1][0] = tloop124delay[0];
assign v1_wr_data_valid[/*idx82=*/ 15][/*idx22=*/ 1][0] = tloop124delay[0];
assign v1_wr_data_input[/*idx82=*/ 15][/*idx22=*/ 1][0] = v129;


//TerminatorOp

//} Unrolled body 15 of loop82.
//DEBUG: /*idx82=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t130;
assign t130 = tloop127;
//printTimeOffset
reg t130delay[1:0] = '{default:0} ;
always@(*) t130delay[0] <= t130;
generate
genvar i131;

for(i131 = 1; i131<= 1; i131= i131 + 1) begin
always@(posedge clk) begin
t130delay[i131] <= t130delay[i131-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop77 = t130delay[1];

//TerminatorOp

//} Unrolled body 1 of loop22.
//DEBUG: /*idx22=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop22.
//DEBUG: /*idx22=*/ 2'd2, expected 2
//printTimeOffset
reg tloop77delay[0:0] = '{default:0} ;
always@(*) tloop77delay[0] <= tloop77;
generate
genvar i133;

for(i133 = 1; i133<= 0; i133= i133 + 1) begin
always@(posedge clk) begin
tloop77delay[i133] <= tloop77delay[i133-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg135[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg135[0] <= tloop77;
always@(posedge clk) shiftreg135[/*v5=*/ 1:1] <= shiftreg135[/*v5=*/ 0:0];
wire v134 = shiftreg135[/*v5=*/ 1];
//printTimeOffset
reg v134delay[1:0] = '{default:0} ;
always@(*) v134delay[0] <= v134;
generate
genvar i136;

for(i136 = 1; i136<= 1; i136= i136 + 1) begin
always@(posedge clk) begin
v134delay[i136] <= v134delay[i136-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop137.
//DEBUG: /*idx137=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop138 = v134delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v139 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][32] = v134delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 0][/*idx22=*/ 2][0] = v134delay[0];
assign v1_wr_data_valid[/*idx137=*/ 0][/*idx22=*/ 2][0] = v134delay[0];
assign v1_wr_data_input[/*idx137=*/ 0][/*idx22=*/ 2][0] = v139;


//TerminatorOp

//} Unrolled body 0 of loop137.
//DEBUG: /*idx137=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop137.
//DEBUG: /*idx137=*/ 1'd1, expected 1
//printTimeOffset
reg tloop138delay[1:0] = '{default:0} ;
always@(*) tloop138delay[0] <= tloop138;
generate
genvar i141;

for(i141 = 1; i141<= 1; i141= i141 + 1) begin
always@(posedge clk) begin
tloop138delay[i141] <= tloop138delay[i141-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop140 = tloop138delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v142 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][33] = tloop138delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 1][/*idx22=*/ 2][0] = tloop138delay[0];
assign v1_wr_data_valid[/*idx137=*/ 1][/*idx22=*/ 2][0] = tloop138delay[0];
assign v1_wr_data_input[/*idx137=*/ 1][/*idx22=*/ 2][0] = v142;


//TerminatorOp

//} Unrolled body 1 of loop137.
//DEBUG: /*idx137=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop137.
//DEBUG: /*idx137=*/ 2'd2, expected 2
//printTimeOffset
reg tloop140delay[1:0] = '{default:0} ;
always@(*) tloop140delay[0] <= tloop140;
generate
genvar i144;

for(i144 = 1; i144<= 1; i144= i144 + 1) begin
always@(posedge clk) begin
tloop140delay[i144] <= tloop140delay[i144-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop143 = tloop140delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v145 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][34] = tloop140delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 2][/*idx22=*/ 2][0] = tloop140delay[0];
assign v1_wr_data_valid[/*idx137=*/ 2][/*idx22=*/ 2][0] = tloop140delay[0];
assign v1_wr_data_input[/*idx137=*/ 2][/*idx22=*/ 2][0] = v145;


//TerminatorOp

//} Unrolled body 2 of loop137.
//DEBUG: /*idx137=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop137.
//DEBUG: /*idx137=*/ 2'd3, expected 3
//printTimeOffset
reg tloop143delay[1:0] = '{default:0} ;
always@(*) tloop143delay[0] <= tloop143;
generate
genvar i147;

for(i147 = 1; i147<= 1; i147= i147 + 1) begin
always@(posedge clk) begin
tloop143delay[i147] <= tloop143delay[i147-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop146 = tloop143delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v148 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][35] = tloop143delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 3][/*idx22=*/ 2][0] = tloop143delay[0];
assign v1_wr_data_valid[/*idx137=*/ 3][/*idx22=*/ 2][0] = tloop143delay[0];
assign v1_wr_data_input[/*idx137=*/ 3][/*idx22=*/ 2][0] = v148;


//TerminatorOp

//} Unrolled body 3 of loop137.
//DEBUG: /*idx137=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop137.
//DEBUG: /*idx137=*/ 3'd4, expected 4
//printTimeOffset
reg tloop146delay[1:0] = '{default:0} ;
always@(*) tloop146delay[0] <= tloop146;
generate
genvar i150;

for(i150 = 1; i150<= 1; i150= i150 + 1) begin
always@(posedge clk) begin
tloop146delay[i150] <= tloop146delay[i150-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop149 = tloop146delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v151 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][36] = tloop146delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 4][/*idx22=*/ 2][0] = tloop146delay[0];
assign v1_wr_data_valid[/*idx137=*/ 4][/*idx22=*/ 2][0] = tloop146delay[0];
assign v1_wr_data_input[/*idx137=*/ 4][/*idx22=*/ 2][0] = v151;


//TerminatorOp

//} Unrolled body 4 of loop137.
//DEBUG: /*idx137=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop137.
//DEBUG: /*idx137=*/ 3'd5, expected 5
//printTimeOffset
reg tloop149delay[1:0] = '{default:0} ;
always@(*) tloop149delay[0] <= tloop149;
generate
genvar i153;

for(i153 = 1; i153<= 1; i153= i153 + 1) begin
always@(posedge clk) begin
tloop149delay[i153] <= tloop149delay[i153-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop152 = tloop149delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v154 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][37] = tloop149delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 5][/*idx22=*/ 2][0] = tloop149delay[0];
assign v1_wr_data_valid[/*idx137=*/ 5][/*idx22=*/ 2][0] = tloop149delay[0];
assign v1_wr_data_input[/*idx137=*/ 5][/*idx22=*/ 2][0] = v154;


//TerminatorOp

//} Unrolled body 5 of loop137.
//DEBUG: /*idx137=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop137.
//DEBUG: /*idx137=*/ 3'd6, expected 6
//printTimeOffset
reg tloop152delay[1:0] = '{default:0} ;
always@(*) tloop152delay[0] <= tloop152;
generate
genvar i156;

for(i156 = 1; i156<= 1; i156= i156 + 1) begin
always@(posedge clk) begin
tloop152delay[i156] <= tloop152delay[i156-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop155 = tloop152delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v157 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][38] = tloop152delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 6][/*idx22=*/ 2][0] = tloop152delay[0];
assign v1_wr_data_valid[/*idx137=*/ 6][/*idx22=*/ 2][0] = tloop152delay[0];
assign v1_wr_data_input[/*idx137=*/ 6][/*idx22=*/ 2][0] = v157;


//TerminatorOp

//} Unrolled body 6 of loop137.
//DEBUG: /*idx137=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop137.
//DEBUG: /*idx137=*/ 3'd7, expected 7
//printTimeOffset
reg tloop155delay[1:0] = '{default:0} ;
always@(*) tloop155delay[0] <= tloop155;
generate
genvar i159;

for(i159 = 1; i159<= 1; i159= i159 + 1) begin
always@(posedge clk) begin
tloop155delay[i159] <= tloop155delay[i159-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop158 = tloop155delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v160 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][39] = tloop155delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 7][/*idx22=*/ 2][0] = tloop155delay[0];
assign v1_wr_data_valid[/*idx137=*/ 7][/*idx22=*/ 2][0] = tloop155delay[0];
assign v1_wr_data_input[/*idx137=*/ 7][/*idx22=*/ 2][0] = v160;


//TerminatorOp

//} Unrolled body 7 of loop137.
//DEBUG: /*idx137=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop137.
//DEBUG: /*idx137=*/ 4'd8, expected 8
//printTimeOffset
reg tloop158delay[1:0] = '{default:0} ;
always@(*) tloop158delay[0] <= tloop158;
generate
genvar i162;

for(i162 = 1; i162<= 1; i162= i162 + 1) begin
always@(posedge clk) begin
tloop158delay[i162] <= tloop158delay[i162-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop161 = tloop158delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v163 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][40] = tloop158delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 8][/*idx22=*/ 2][0] = tloop158delay[0];
assign v1_wr_data_valid[/*idx137=*/ 8][/*idx22=*/ 2][0] = tloop158delay[0];
assign v1_wr_data_input[/*idx137=*/ 8][/*idx22=*/ 2][0] = v163;


//TerminatorOp

//} Unrolled body 8 of loop137.
//DEBUG: /*idx137=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop137.
//DEBUG: /*idx137=*/ 4'd9, expected 9
//printTimeOffset
reg tloop161delay[1:0] = '{default:0} ;
always@(*) tloop161delay[0] <= tloop161;
generate
genvar i165;

for(i165 = 1; i165<= 1; i165= i165 + 1) begin
always@(posedge clk) begin
tloop161delay[i165] <= tloop161delay[i165-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop164 = tloop161delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v166 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][41] = tloop161delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 9][/*idx22=*/ 2][0] = tloop161delay[0];
assign v1_wr_data_valid[/*idx137=*/ 9][/*idx22=*/ 2][0] = tloop161delay[0];
assign v1_wr_data_input[/*idx137=*/ 9][/*idx22=*/ 2][0] = v166;


//TerminatorOp

//} Unrolled body 9 of loop137.
//DEBUG: /*idx137=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop137.
//DEBUG: /*idx137=*/ 4'd10, expected 10
//printTimeOffset
reg tloop164delay[1:0] = '{default:0} ;
always@(*) tloop164delay[0] <= tloop164;
generate
genvar i168;

for(i168 = 1; i168<= 1; i168= i168 + 1) begin
always@(posedge clk) begin
tloop164delay[i168] <= tloop164delay[i168-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop167 = tloop164delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v169 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][42] = tloop164delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 10][/*idx22=*/ 2][0] = tloop164delay[0];
assign v1_wr_data_valid[/*idx137=*/ 10][/*idx22=*/ 2][0] = tloop164delay[0];
assign v1_wr_data_input[/*idx137=*/ 10][/*idx22=*/ 2][0] = v169;


//TerminatorOp

//} Unrolled body 10 of loop137.
//DEBUG: /*idx137=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop137.
//DEBUG: /*idx137=*/ 4'd11, expected 11
//printTimeOffset
reg tloop167delay[1:0] = '{default:0} ;
always@(*) tloop167delay[0] <= tloop167;
generate
genvar i171;

for(i171 = 1; i171<= 1; i171= i171 + 1) begin
always@(posedge clk) begin
tloop167delay[i171] <= tloop167delay[i171-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop170 = tloop167delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v172 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][43] = tloop167delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 11][/*idx22=*/ 2][0] = tloop167delay[0];
assign v1_wr_data_valid[/*idx137=*/ 11][/*idx22=*/ 2][0] = tloop167delay[0];
assign v1_wr_data_input[/*idx137=*/ 11][/*idx22=*/ 2][0] = v172;


//TerminatorOp

//} Unrolled body 11 of loop137.
//DEBUG: /*idx137=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop137.
//DEBUG: /*idx137=*/ 4'd12, expected 12
//printTimeOffset
reg tloop170delay[1:0] = '{default:0} ;
always@(*) tloop170delay[0] <= tloop170;
generate
genvar i174;

for(i174 = 1; i174<= 1; i174= i174 + 1) begin
always@(posedge clk) begin
tloop170delay[i174] <= tloop170delay[i174-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop173 = tloop170delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v175 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][44] = tloop170delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 12][/*idx22=*/ 2][0] = tloop170delay[0];
assign v1_wr_data_valid[/*idx137=*/ 12][/*idx22=*/ 2][0] = tloop170delay[0];
assign v1_wr_data_input[/*idx137=*/ 12][/*idx22=*/ 2][0] = v175;


//TerminatorOp

//} Unrolled body 12 of loop137.
//DEBUG: /*idx137=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop137.
//DEBUG: /*idx137=*/ 4'd13, expected 13
//printTimeOffset
reg tloop173delay[1:0] = '{default:0} ;
always@(*) tloop173delay[0] <= tloop173;
generate
genvar i177;

for(i177 = 1; i177<= 1; i177= i177 + 1) begin
always@(posedge clk) begin
tloop173delay[i177] <= tloop173delay[i177-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop176 = tloop173delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v178 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][45] = tloop173delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 13][/*idx22=*/ 2][0] = tloop173delay[0];
assign v1_wr_data_valid[/*idx137=*/ 13][/*idx22=*/ 2][0] = tloop173delay[0];
assign v1_wr_data_input[/*idx137=*/ 13][/*idx22=*/ 2][0] = v178;


//TerminatorOp

//} Unrolled body 13 of loop137.
//DEBUG: /*idx137=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop137.
//DEBUG: /*idx137=*/ 4'd14, expected 14
//printTimeOffset
reg tloop176delay[1:0] = '{default:0} ;
always@(*) tloop176delay[0] <= tloop176;
generate
genvar i180;

for(i180 = 1; i180<= 1; i180= i180 + 1) begin
always@(posedge clk) begin
tloop176delay[i180] <= tloop176delay[i180-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop179 = tloop176delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v181 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][46] = tloop176delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 14][/*idx22=*/ 2][0] = tloop176delay[0];
assign v1_wr_data_valid[/*idx137=*/ 14][/*idx22=*/ 2][0] = tloop176delay[0];
assign v1_wr_data_input[/*idx137=*/ 14][/*idx22=*/ 2][0] = v181;


//TerminatorOp

//} Unrolled body 14 of loop137.
//DEBUG: /*idx137=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop137.
//DEBUG: /*idx137=*/ 4'd15, expected 15
//printTimeOffset
reg tloop179delay[1:0] = '{default:0} ;
always@(*) tloop179delay[0] <= tloop179;
generate
genvar i183;

for(i183 = 1; i183<= 1; i183= i183 + 1) begin
always@(posedge clk) begin
tloop179delay[i183] <= tloop179delay[i183-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop182 = tloop179delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v184 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][47] = tloop179delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx137=*/ 15][/*idx22=*/ 2][0] = tloop179delay[0];
assign v1_wr_data_valid[/*idx137=*/ 15][/*idx22=*/ 2][0] = tloop179delay[0];
assign v1_wr_data_input[/*idx137=*/ 15][/*idx22=*/ 2][0] = v184;


//TerminatorOp

//} Unrolled body 15 of loop137.
//DEBUG: /*idx137=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t185;
assign t185 = tloop182;
//printTimeOffset
reg t185delay[1:0] = '{default:0} ;
always@(*) t185delay[0] <= t185;
generate
genvar i186;

for(i186 = 1; i186<= 1; i186= i186 + 1) begin
always@(posedge clk) begin
t185delay[i186] <= t185delay[i186-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop132 = t185delay[1];

//TerminatorOp

//} Unrolled body 2 of loop22.
//DEBUG: /*idx22=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop22.
//DEBUG: /*idx22=*/ 2'd3, expected 3
//printTimeOffset
reg tloop132delay[0:0] = '{default:0} ;
always@(*) tloop132delay[0] <= tloop132;
generate
genvar i188;

for(i188 = 1; i188<= 0; i188= i188 + 1) begin
always@(posedge clk) begin
tloop132delay[i188] <= tloop132delay[i188-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg190[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg190[0] <= tloop132;
always@(posedge clk) shiftreg190[/*v5=*/ 1:1] <= shiftreg190[/*v5=*/ 0:0];
wire v189 = shiftreg190[/*v5=*/ 1];
//printTimeOffset
reg v189delay[1:0] = '{default:0} ;
always@(*) v189delay[0] <= v189;
generate
genvar i191;

for(i191 = 1; i191<= 1; i191= i191 + 1) begin
always@(posedge clk) begin
v189delay[i191] <= v189delay[i191-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop192.
//DEBUG: /*idx192=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop193 = v189delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v194 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][48] = v189delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 0][/*idx22=*/ 3][0] = v189delay[0];
assign v1_wr_data_valid[/*idx192=*/ 0][/*idx22=*/ 3][0] = v189delay[0];
assign v1_wr_data_input[/*idx192=*/ 0][/*idx22=*/ 3][0] = v194;


//TerminatorOp

//} Unrolled body 0 of loop192.
//DEBUG: /*idx192=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop192.
//DEBUG: /*idx192=*/ 1'd1, expected 1
//printTimeOffset
reg tloop193delay[1:0] = '{default:0} ;
always@(*) tloop193delay[0] <= tloop193;
generate
genvar i196;

for(i196 = 1; i196<= 1; i196= i196 + 1) begin
always@(posedge clk) begin
tloop193delay[i196] <= tloop193delay[i196-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop195 = tloop193delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v197 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][49] = tloop193delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 1][/*idx22=*/ 3][0] = tloop193delay[0];
assign v1_wr_data_valid[/*idx192=*/ 1][/*idx22=*/ 3][0] = tloop193delay[0];
assign v1_wr_data_input[/*idx192=*/ 1][/*idx22=*/ 3][0] = v197;


//TerminatorOp

//} Unrolled body 1 of loop192.
//DEBUG: /*idx192=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop192.
//DEBUG: /*idx192=*/ 2'd2, expected 2
//printTimeOffset
reg tloop195delay[1:0] = '{default:0} ;
always@(*) tloop195delay[0] <= tloop195;
generate
genvar i199;

for(i199 = 1; i199<= 1; i199= i199 + 1) begin
always@(posedge clk) begin
tloop195delay[i199] <= tloop195delay[i199-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop198 = tloop195delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v200 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][50] = tloop195delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 2][/*idx22=*/ 3][0] = tloop195delay[0];
assign v1_wr_data_valid[/*idx192=*/ 2][/*idx22=*/ 3][0] = tloop195delay[0];
assign v1_wr_data_input[/*idx192=*/ 2][/*idx22=*/ 3][0] = v200;


//TerminatorOp

//} Unrolled body 2 of loop192.
//DEBUG: /*idx192=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop192.
//DEBUG: /*idx192=*/ 2'd3, expected 3
//printTimeOffset
reg tloop198delay[1:0] = '{default:0} ;
always@(*) tloop198delay[0] <= tloop198;
generate
genvar i202;

for(i202 = 1; i202<= 1; i202= i202 + 1) begin
always@(posedge clk) begin
tloop198delay[i202] <= tloop198delay[i202-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop201 = tloop198delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v203 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][51] = tloop198delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 3][/*idx22=*/ 3][0] = tloop198delay[0];
assign v1_wr_data_valid[/*idx192=*/ 3][/*idx22=*/ 3][0] = tloop198delay[0];
assign v1_wr_data_input[/*idx192=*/ 3][/*idx22=*/ 3][0] = v203;


//TerminatorOp

//} Unrolled body 3 of loop192.
//DEBUG: /*idx192=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop192.
//DEBUG: /*idx192=*/ 3'd4, expected 4
//printTimeOffset
reg tloop201delay[1:0] = '{default:0} ;
always@(*) tloop201delay[0] <= tloop201;
generate
genvar i205;

for(i205 = 1; i205<= 1; i205= i205 + 1) begin
always@(posedge clk) begin
tloop201delay[i205] <= tloop201delay[i205-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop204 = tloop201delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v206 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][52] = tloop201delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 4][/*idx22=*/ 3][0] = tloop201delay[0];
assign v1_wr_data_valid[/*idx192=*/ 4][/*idx22=*/ 3][0] = tloop201delay[0];
assign v1_wr_data_input[/*idx192=*/ 4][/*idx22=*/ 3][0] = v206;


//TerminatorOp

//} Unrolled body 4 of loop192.
//DEBUG: /*idx192=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop192.
//DEBUG: /*idx192=*/ 3'd5, expected 5
//printTimeOffset
reg tloop204delay[1:0] = '{default:0} ;
always@(*) tloop204delay[0] <= tloop204;
generate
genvar i208;

for(i208 = 1; i208<= 1; i208= i208 + 1) begin
always@(posedge clk) begin
tloop204delay[i208] <= tloop204delay[i208-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop207 = tloop204delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v209 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][53] = tloop204delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 5][/*idx22=*/ 3][0] = tloop204delay[0];
assign v1_wr_data_valid[/*idx192=*/ 5][/*idx22=*/ 3][0] = tloop204delay[0];
assign v1_wr_data_input[/*idx192=*/ 5][/*idx22=*/ 3][0] = v209;


//TerminatorOp

//} Unrolled body 5 of loop192.
//DEBUG: /*idx192=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop192.
//DEBUG: /*idx192=*/ 3'd6, expected 6
//printTimeOffset
reg tloop207delay[1:0] = '{default:0} ;
always@(*) tloop207delay[0] <= tloop207;
generate
genvar i211;

for(i211 = 1; i211<= 1; i211= i211 + 1) begin
always@(posedge clk) begin
tloop207delay[i211] <= tloop207delay[i211-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop210 = tloop207delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v212 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][54] = tloop207delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 6][/*idx22=*/ 3][0] = tloop207delay[0];
assign v1_wr_data_valid[/*idx192=*/ 6][/*idx22=*/ 3][0] = tloop207delay[0];
assign v1_wr_data_input[/*idx192=*/ 6][/*idx22=*/ 3][0] = v212;


//TerminatorOp

//} Unrolled body 6 of loop192.
//DEBUG: /*idx192=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop192.
//DEBUG: /*idx192=*/ 3'd7, expected 7
//printTimeOffset
reg tloop210delay[1:0] = '{default:0} ;
always@(*) tloop210delay[0] <= tloop210;
generate
genvar i214;

for(i214 = 1; i214<= 1; i214= i214 + 1) begin
always@(posedge clk) begin
tloop210delay[i214] <= tloop210delay[i214-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop213 = tloop210delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v215 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][55] = tloop210delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 7][/*idx22=*/ 3][0] = tloop210delay[0];
assign v1_wr_data_valid[/*idx192=*/ 7][/*idx22=*/ 3][0] = tloop210delay[0];
assign v1_wr_data_input[/*idx192=*/ 7][/*idx22=*/ 3][0] = v215;


//TerminatorOp

//} Unrolled body 7 of loop192.
//DEBUG: /*idx192=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop192.
//DEBUG: /*idx192=*/ 4'd8, expected 8
//printTimeOffset
reg tloop213delay[1:0] = '{default:0} ;
always@(*) tloop213delay[0] <= tloop213;
generate
genvar i217;

for(i217 = 1; i217<= 1; i217= i217 + 1) begin
always@(posedge clk) begin
tloop213delay[i217] <= tloop213delay[i217-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop216 = tloop213delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v218 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][56] = tloop213delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 8][/*idx22=*/ 3][0] = tloop213delay[0];
assign v1_wr_data_valid[/*idx192=*/ 8][/*idx22=*/ 3][0] = tloop213delay[0];
assign v1_wr_data_input[/*idx192=*/ 8][/*idx22=*/ 3][0] = v218;


//TerminatorOp

//} Unrolled body 8 of loop192.
//DEBUG: /*idx192=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop192.
//DEBUG: /*idx192=*/ 4'd9, expected 9
//printTimeOffset
reg tloop216delay[1:0] = '{default:0} ;
always@(*) tloop216delay[0] <= tloop216;
generate
genvar i220;

for(i220 = 1; i220<= 1; i220= i220 + 1) begin
always@(posedge clk) begin
tloop216delay[i220] <= tloop216delay[i220-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop219 = tloop216delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v221 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][57] = tloop216delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 9][/*idx22=*/ 3][0] = tloop216delay[0];
assign v1_wr_data_valid[/*idx192=*/ 9][/*idx22=*/ 3][0] = tloop216delay[0];
assign v1_wr_data_input[/*idx192=*/ 9][/*idx22=*/ 3][0] = v221;


//TerminatorOp

//} Unrolled body 9 of loop192.
//DEBUG: /*idx192=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop192.
//DEBUG: /*idx192=*/ 4'd10, expected 10
//printTimeOffset
reg tloop219delay[1:0] = '{default:0} ;
always@(*) tloop219delay[0] <= tloop219;
generate
genvar i223;

for(i223 = 1; i223<= 1; i223= i223 + 1) begin
always@(posedge clk) begin
tloop219delay[i223] <= tloop219delay[i223-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop222 = tloop219delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v224 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][58] = tloop219delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 10][/*idx22=*/ 3][0] = tloop219delay[0];
assign v1_wr_data_valid[/*idx192=*/ 10][/*idx22=*/ 3][0] = tloop219delay[0];
assign v1_wr_data_input[/*idx192=*/ 10][/*idx22=*/ 3][0] = v224;


//TerminatorOp

//} Unrolled body 10 of loop192.
//DEBUG: /*idx192=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop192.
//DEBUG: /*idx192=*/ 4'd11, expected 11
//printTimeOffset
reg tloop222delay[1:0] = '{default:0} ;
always@(*) tloop222delay[0] <= tloop222;
generate
genvar i226;

for(i226 = 1; i226<= 1; i226= i226 + 1) begin
always@(posedge clk) begin
tloop222delay[i226] <= tloop222delay[i226-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop225 = tloop222delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v227 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][59] = tloop222delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 11][/*idx22=*/ 3][0] = tloop222delay[0];
assign v1_wr_data_valid[/*idx192=*/ 11][/*idx22=*/ 3][0] = tloop222delay[0];
assign v1_wr_data_input[/*idx192=*/ 11][/*idx22=*/ 3][0] = v227;


//TerminatorOp

//} Unrolled body 11 of loop192.
//DEBUG: /*idx192=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop192.
//DEBUG: /*idx192=*/ 4'd12, expected 12
//printTimeOffset
reg tloop225delay[1:0] = '{default:0} ;
always@(*) tloop225delay[0] <= tloop225;
generate
genvar i229;

for(i229 = 1; i229<= 1; i229= i229 + 1) begin
always@(posedge clk) begin
tloop225delay[i229] <= tloop225delay[i229-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop228 = tloop225delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v230 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][60] = tloop225delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 12][/*idx22=*/ 3][0] = tloop225delay[0];
assign v1_wr_data_valid[/*idx192=*/ 12][/*idx22=*/ 3][0] = tloop225delay[0];
assign v1_wr_data_input[/*idx192=*/ 12][/*idx22=*/ 3][0] = v230;


//TerminatorOp

//} Unrolled body 12 of loop192.
//DEBUG: /*idx192=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop192.
//DEBUG: /*idx192=*/ 4'd13, expected 13
//printTimeOffset
reg tloop228delay[1:0] = '{default:0} ;
always@(*) tloop228delay[0] <= tloop228;
generate
genvar i232;

for(i232 = 1; i232<= 1; i232= i232 + 1) begin
always@(posedge clk) begin
tloop228delay[i232] <= tloop228delay[i232-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop231 = tloop228delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v233 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][61] = tloop228delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 13][/*idx22=*/ 3][0] = tloop228delay[0];
assign v1_wr_data_valid[/*idx192=*/ 13][/*idx22=*/ 3][0] = tloop228delay[0];
assign v1_wr_data_input[/*idx192=*/ 13][/*idx22=*/ 3][0] = v233;


//TerminatorOp

//} Unrolled body 13 of loop192.
//DEBUG: /*idx192=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop192.
//DEBUG: /*idx192=*/ 4'd14, expected 14
//printTimeOffset
reg tloop231delay[1:0] = '{default:0} ;
always@(*) tloop231delay[0] <= tloop231;
generate
genvar i235;

for(i235 = 1; i235<= 1; i235= i235 + 1) begin
always@(posedge clk) begin
tloop231delay[i235] <= tloop231delay[i235-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop234 = tloop231delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v236 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][62] = tloop231delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 14][/*idx22=*/ 3][0] = tloop231delay[0];
assign v1_wr_data_valid[/*idx192=*/ 14][/*idx22=*/ 3][0] = tloop231delay[0];
assign v1_wr_data_input[/*idx192=*/ 14][/*idx22=*/ 3][0] = v236;


//TerminatorOp

//} Unrolled body 14 of loop192.
//DEBUG: /*idx192=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop192.
//DEBUG: /*idx192=*/ 4'd15, expected 15
//printTimeOffset
reg tloop234delay[1:0] = '{default:0} ;
always@(*) tloop234delay[0] <= tloop234;
generate
genvar i238;

for(i238 = 1; i238<= 1; i238= i238 + 1) begin
always@(posedge clk) begin
tloop234delay[i238] <= tloop234delay[i238-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop237 = tloop234delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v239 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][63] = tloop234delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx192=*/ 15][/*idx22=*/ 3][0] = tloop234delay[0];
assign v1_wr_data_valid[/*idx192=*/ 15][/*idx22=*/ 3][0] = tloop234delay[0];
assign v1_wr_data_input[/*idx192=*/ 15][/*idx22=*/ 3][0] = v239;


//TerminatorOp

//} Unrolled body 15 of loop192.
//DEBUG: /*idx192=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t240;
assign t240 = tloop237;
//printTimeOffset
reg t240delay[1:0] = '{default:0} ;
always@(*) t240delay[0] <= t240;
generate
genvar i241;

for(i241 = 1; i241<= 1; i241= i241 + 1) begin
always@(posedge clk) begin
t240delay[i241] <= t240delay[i241-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop187 = t240delay[1];

//TerminatorOp

//} Unrolled body 3 of loop22.
//DEBUG: /*idx22=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop22.
//DEBUG: /*idx22=*/ 3'd4, expected 4
//printTimeOffset
reg tloop187delay[0:0] = '{default:0} ;
always@(*) tloop187delay[0] <= tloop187;
generate
genvar i243;

for(i243 = 1; i243<= 0; i243= i243 + 1) begin
always@(posedge clk) begin
tloop187delay[i243] <= tloop187delay[i243-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg245[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg245[0] <= tloop187;
always@(posedge clk) shiftreg245[/*v5=*/ 1:1] <= shiftreg245[/*v5=*/ 0:0];
wire v244 = shiftreg245[/*v5=*/ 1];
//printTimeOffset
reg v244delay[1:0] = '{default:0} ;
always@(*) v244delay[0] <= v244;
generate
genvar i246;

for(i246 = 1; i246<= 1; i246= i246 + 1) begin
always@(posedge clk) begin
v244delay[i246] <= v244delay[i246-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop247.
//DEBUG: /*idx247=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop248 = v244delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v249 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][64] = v244delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 0][/*idx22=*/ 4][0] = v244delay[0];
assign v1_wr_data_valid[/*idx247=*/ 0][/*idx22=*/ 4][0] = v244delay[0];
assign v1_wr_data_input[/*idx247=*/ 0][/*idx22=*/ 4][0] = v249;


//TerminatorOp

//} Unrolled body 0 of loop247.
//DEBUG: /*idx247=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop247.
//DEBUG: /*idx247=*/ 1'd1, expected 1
//printTimeOffset
reg tloop248delay[1:0] = '{default:0} ;
always@(*) tloop248delay[0] <= tloop248;
generate
genvar i251;

for(i251 = 1; i251<= 1; i251= i251 + 1) begin
always@(posedge clk) begin
tloop248delay[i251] <= tloop248delay[i251-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop250 = tloop248delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v252 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][65] = tloop248delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 1][/*idx22=*/ 4][0] = tloop248delay[0];
assign v1_wr_data_valid[/*idx247=*/ 1][/*idx22=*/ 4][0] = tloop248delay[0];
assign v1_wr_data_input[/*idx247=*/ 1][/*idx22=*/ 4][0] = v252;


//TerminatorOp

//} Unrolled body 1 of loop247.
//DEBUG: /*idx247=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop247.
//DEBUG: /*idx247=*/ 2'd2, expected 2
//printTimeOffset
reg tloop250delay[1:0] = '{default:0} ;
always@(*) tloop250delay[0] <= tloop250;
generate
genvar i254;

for(i254 = 1; i254<= 1; i254= i254 + 1) begin
always@(posedge clk) begin
tloop250delay[i254] <= tloop250delay[i254-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop253 = tloop250delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v255 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][66] = tloop250delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 2][/*idx22=*/ 4][0] = tloop250delay[0];
assign v1_wr_data_valid[/*idx247=*/ 2][/*idx22=*/ 4][0] = tloop250delay[0];
assign v1_wr_data_input[/*idx247=*/ 2][/*idx22=*/ 4][0] = v255;


//TerminatorOp

//} Unrolled body 2 of loop247.
//DEBUG: /*idx247=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop247.
//DEBUG: /*idx247=*/ 2'd3, expected 3
//printTimeOffset
reg tloop253delay[1:0] = '{default:0} ;
always@(*) tloop253delay[0] <= tloop253;
generate
genvar i257;

for(i257 = 1; i257<= 1; i257= i257 + 1) begin
always@(posedge clk) begin
tloop253delay[i257] <= tloop253delay[i257-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop256 = tloop253delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v258 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][67] = tloop253delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 3][/*idx22=*/ 4][0] = tloop253delay[0];
assign v1_wr_data_valid[/*idx247=*/ 3][/*idx22=*/ 4][0] = tloop253delay[0];
assign v1_wr_data_input[/*idx247=*/ 3][/*idx22=*/ 4][0] = v258;


//TerminatorOp

//} Unrolled body 3 of loop247.
//DEBUG: /*idx247=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop247.
//DEBUG: /*idx247=*/ 3'd4, expected 4
//printTimeOffset
reg tloop256delay[1:0] = '{default:0} ;
always@(*) tloop256delay[0] <= tloop256;
generate
genvar i260;

for(i260 = 1; i260<= 1; i260= i260 + 1) begin
always@(posedge clk) begin
tloop256delay[i260] <= tloop256delay[i260-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop259 = tloop256delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v261 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][68] = tloop256delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 4][/*idx22=*/ 4][0] = tloop256delay[0];
assign v1_wr_data_valid[/*idx247=*/ 4][/*idx22=*/ 4][0] = tloop256delay[0];
assign v1_wr_data_input[/*idx247=*/ 4][/*idx22=*/ 4][0] = v261;


//TerminatorOp

//} Unrolled body 4 of loop247.
//DEBUG: /*idx247=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop247.
//DEBUG: /*idx247=*/ 3'd5, expected 5
//printTimeOffset
reg tloop259delay[1:0] = '{default:0} ;
always@(*) tloop259delay[0] <= tloop259;
generate
genvar i263;

for(i263 = 1; i263<= 1; i263= i263 + 1) begin
always@(posedge clk) begin
tloop259delay[i263] <= tloop259delay[i263-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop262 = tloop259delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v264 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][69] = tloop259delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 5][/*idx22=*/ 4][0] = tloop259delay[0];
assign v1_wr_data_valid[/*idx247=*/ 5][/*idx22=*/ 4][0] = tloop259delay[0];
assign v1_wr_data_input[/*idx247=*/ 5][/*idx22=*/ 4][0] = v264;


//TerminatorOp

//} Unrolled body 5 of loop247.
//DEBUG: /*idx247=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop247.
//DEBUG: /*idx247=*/ 3'd6, expected 6
//printTimeOffset
reg tloop262delay[1:0] = '{default:0} ;
always@(*) tloop262delay[0] <= tloop262;
generate
genvar i266;

for(i266 = 1; i266<= 1; i266= i266 + 1) begin
always@(posedge clk) begin
tloop262delay[i266] <= tloop262delay[i266-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop265 = tloop262delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v267 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][70] = tloop262delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 6][/*idx22=*/ 4][0] = tloop262delay[0];
assign v1_wr_data_valid[/*idx247=*/ 6][/*idx22=*/ 4][0] = tloop262delay[0];
assign v1_wr_data_input[/*idx247=*/ 6][/*idx22=*/ 4][0] = v267;


//TerminatorOp

//} Unrolled body 6 of loop247.
//DEBUG: /*idx247=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop247.
//DEBUG: /*idx247=*/ 3'd7, expected 7
//printTimeOffset
reg tloop265delay[1:0] = '{default:0} ;
always@(*) tloop265delay[0] <= tloop265;
generate
genvar i269;

for(i269 = 1; i269<= 1; i269= i269 + 1) begin
always@(posedge clk) begin
tloop265delay[i269] <= tloop265delay[i269-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop268 = tloop265delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v270 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][71] = tloop265delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 7][/*idx22=*/ 4][0] = tloop265delay[0];
assign v1_wr_data_valid[/*idx247=*/ 7][/*idx22=*/ 4][0] = tloop265delay[0];
assign v1_wr_data_input[/*idx247=*/ 7][/*idx22=*/ 4][0] = v270;


//TerminatorOp

//} Unrolled body 7 of loop247.
//DEBUG: /*idx247=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop247.
//DEBUG: /*idx247=*/ 4'd8, expected 8
//printTimeOffset
reg tloop268delay[1:0] = '{default:0} ;
always@(*) tloop268delay[0] <= tloop268;
generate
genvar i272;

for(i272 = 1; i272<= 1; i272= i272 + 1) begin
always@(posedge clk) begin
tloop268delay[i272] <= tloop268delay[i272-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop271 = tloop268delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v273 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][72] = tloop268delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 8][/*idx22=*/ 4][0] = tloop268delay[0];
assign v1_wr_data_valid[/*idx247=*/ 8][/*idx22=*/ 4][0] = tloop268delay[0];
assign v1_wr_data_input[/*idx247=*/ 8][/*idx22=*/ 4][0] = v273;


//TerminatorOp

//} Unrolled body 8 of loop247.
//DEBUG: /*idx247=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop247.
//DEBUG: /*idx247=*/ 4'd9, expected 9
//printTimeOffset
reg tloop271delay[1:0] = '{default:0} ;
always@(*) tloop271delay[0] <= tloop271;
generate
genvar i275;

for(i275 = 1; i275<= 1; i275= i275 + 1) begin
always@(posedge clk) begin
tloop271delay[i275] <= tloop271delay[i275-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop274 = tloop271delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v276 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][73] = tloop271delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 9][/*idx22=*/ 4][0] = tloop271delay[0];
assign v1_wr_data_valid[/*idx247=*/ 9][/*idx22=*/ 4][0] = tloop271delay[0];
assign v1_wr_data_input[/*idx247=*/ 9][/*idx22=*/ 4][0] = v276;


//TerminatorOp

//} Unrolled body 9 of loop247.
//DEBUG: /*idx247=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop247.
//DEBUG: /*idx247=*/ 4'd10, expected 10
//printTimeOffset
reg tloop274delay[1:0] = '{default:0} ;
always@(*) tloop274delay[0] <= tloop274;
generate
genvar i278;

for(i278 = 1; i278<= 1; i278= i278 + 1) begin
always@(posedge clk) begin
tloop274delay[i278] <= tloop274delay[i278-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop277 = tloop274delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v279 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][74] = tloop274delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 10][/*idx22=*/ 4][0] = tloop274delay[0];
assign v1_wr_data_valid[/*idx247=*/ 10][/*idx22=*/ 4][0] = tloop274delay[0];
assign v1_wr_data_input[/*idx247=*/ 10][/*idx22=*/ 4][0] = v279;


//TerminatorOp

//} Unrolled body 10 of loop247.
//DEBUG: /*idx247=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop247.
//DEBUG: /*idx247=*/ 4'd11, expected 11
//printTimeOffset
reg tloop277delay[1:0] = '{default:0} ;
always@(*) tloop277delay[0] <= tloop277;
generate
genvar i281;

for(i281 = 1; i281<= 1; i281= i281 + 1) begin
always@(posedge clk) begin
tloop277delay[i281] <= tloop277delay[i281-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop280 = tloop277delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v282 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][75] = tloop277delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 11][/*idx22=*/ 4][0] = tloop277delay[0];
assign v1_wr_data_valid[/*idx247=*/ 11][/*idx22=*/ 4][0] = tloop277delay[0];
assign v1_wr_data_input[/*idx247=*/ 11][/*idx22=*/ 4][0] = v282;


//TerminatorOp

//} Unrolled body 11 of loop247.
//DEBUG: /*idx247=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop247.
//DEBUG: /*idx247=*/ 4'd12, expected 12
//printTimeOffset
reg tloop280delay[1:0] = '{default:0} ;
always@(*) tloop280delay[0] <= tloop280;
generate
genvar i284;

for(i284 = 1; i284<= 1; i284= i284 + 1) begin
always@(posedge clk) begin
tloop280delay[i284] <= tloop280delay[i284-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop283 = tloop280delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v285 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][76] = tloop280delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 12][/*idx22=*/ 4][0] = tloop280delay[0];
assign v1_wr_data_valid[/*idx247=*/ 12][/*idx22=*/ 4][0] = tloop280delay[0];
assign v1_wr_data_input[/*idx247=*/ 12][/*idx22=*/ 4][0] = v285;


//TerminatorOp

//} Unrolled body 12 of loop247.
//DEBUG: /*idx247=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop247.
//DEBUG: /*idx247=*/ 4'd13, expected 13
//printTimeOffset
reg tloop283delay[1:0] = '{default:0} ;
always@(*) tloop283delay[0] <= tloop283;
generate
genvar i287;

for(i287 = 1; i287<= 1; i287= i287 + 1) begin
always@(posedge clk) begin
tloop283delay[i287] <= tloop283delay[i287-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop286 = tloop283delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v288 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][77] = tloop283delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 13][/*idx22=*/ 4][0] = tloop283delay[0];
assign v1_wr_data_valid[/*idx247=*/ 13][/*idx22=*/ 4][0] = tloop283delay[0];
assign v1_wr_data_input[/*idx247=*/ 13][/*idx22=*/ 4][0] = v288;


//TerminatorOp

//} Unrolled body 13 of loop247.
//DEBUG: /*idx247=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop247.
//DEBUG: /*idx247=*/ 4'd14, expected 14
//printTimeOffset
reg tloop286delay[1:0] = '{default:0} ;
always@(*) tloop286delay[0] <= tloop286;
generate
genvar i290;

for(i290 = 1; i290<= 1; i290= i290 + 1) begin
always@(posedge clk) begin
tloop286delay[i290] <= tloop286delay[i290-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop289 = tloop286delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v291 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][78] = tloop286delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 14][/*idx22=*/ 4][0] = tloop286delay[0];
assign v1_wr_data_valid[/*idx247=*/ 14][/*idx22=*/ 4][0] = tloop286delay[0];
assign v1_wr_data_input[/*idx247=*/ 14][/*idx22=*/ 4][0] = v291;


//TerminatorOp

//} Unrolled body 14 of loop247.
//DEBUG: /*idx247=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop247.
//DEBUG: /*idx247=*/ 4'd15, expected 15
//printTimeOffset
reg tloop289delay[1:0] = '{default:0} ;
always@(*) tloop289delay[0] <= tloop289;
generate
genvar i293;

for(i293 = 1; i293<= 1; i293= i293 + 1) begin
always@(posedge clk) begin
tloop289delay[i293] <= tloop289delay[i293-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop292 = tloop289delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v294 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][79] = tloop289delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx247=*/ 15][/*idx22=*/ 4][0] = tloop289delay[0];
assign v1_wr_data_valid[/*idx247=*/ 15][/*idx22=*/ 4][0] = tloop289delay[0];
assign v1_wr_data_input[/*idx247=*/ 15][/*idx22=*/ 4][0] = v294;


//TerminatorOp

//} Unrolled body 15 of loop247.
//DEBUG: /*idx247=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t295;
assign t295 = tloop292;
//printTimeOffset
reg t295delay[1:0] = '{default:0} ;
always@(*) t295delay[0] <= t295;
generate
genvar i296;

for(i296 = 1; i296<= 1; i296= i296 + 1) begin
always@(posedge clk) begin
t295delay[i296] <= t295delay[i296-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop242 = t295delay[1];

//TerminatorOp

//} Unrolled body 4 of loop22.
//DEBUG: /*idx22=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop22.
//DEBUG: /*idx22=*/ 3'd5, expected 5
//printTimeOffset
reg tloop242delay[0:0] = '{default:0} ;
always@(*) tloop242delay[0] <= tloop242;
generate
genvar i298;

for(i298 = 1; i298<= 0; i298= i298 + 1) begin
always@(posedge clk) begin
tloop242delay[i298] <= tloop242delay[i298-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg300[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg300[0] <= tloop242;
always@(posedge clk) shiftreg300[/*v5=*/ 1:1] <= shiftreg300[/*v5=*/ 0:0];
wire v299 = shiftreg300[/*v5=*/ 1];
//printTimeOffset
reg v299delay[1:0] = '{default:0} ;
always@(*) v299delay[0] <= v299;
generate
genvar i301;

for(i301 = 1; i301<= 1; i301= i301 + 1) begin
always@(posedge clk) begin
v299delay[i301] <= v299delay[i301-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop302.
//DEBUG: /*idx302=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop303 = v299delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v304 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][80] = v299delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 0][/*idx22=*/ 5][0] = v299delay[0];
assign v1_wr_data_valid[/*idx302=*/ 0][/*idx22=*/ 5][0] = v299delay[0];
assign v1_wr_data_input[/*idx302=*/ 0][/*idx22=*/ 5][0] = v304;


//TerminatorOp

//} Unrolled body 0 of loop302.
//DEBUG: /*idx302=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop302.
//DEBUG: /*idx302=*/ 1'd1, expected 1
//printTimeOffset
reg tloop303delay[1:0] = '{default:0} ;
always@(*) tloop303delay[0] <= tloop303;
generate
genvar i306;

for(i306 = 1; i306<= 1; i306= i306 + 1) begin
always@(posedge clk) begin
tloop303delay[i306] <= tloop303delay[i306-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop305 = tloop303delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v307 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][81] = tloop303delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 1][/*idx22=*/ 5][0] = tloop303delay[0];
assign v1_wr_data_valid[/*idx302=*/ 1][/*idx22=*/ 5][0] = tloop303delay[0];
assign v1_wr_data_input[/*idx302=*/ 1][/*idx22=*/ 5][0] = v307;


//TerminatorOp

//} Unrolled body 1 of loop302.
//DEBUG: /*idx302=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop302.
//DEBUG: /*idx302=*/ 2'd2, expected 2
//printTimeOffset
reg tloop305delay[1:0] = '{default:0} ;
always@(*) tloop305delay[0] <= tloop305;
generate
genvar i309;

for(i309 = 1; i309<= 1; i309= i309 + 1) begin
always@(posedge clk) begin
tloop305delay[i309] <= tloop305delay[i309-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop308 = tloop305delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v310 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][82] = tloop305delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 2][/*idx22=*/ 5][0] = tloop305delay[0];
assign v1_wr_data_valid[/*idx302=*/ 2][/*idx22=*/ 5][0] = tloop305delay[0];
assign v1_wr_data_input[/*idx302=*/ 2][/*idx22=*/ 5][0] = v310;


//TerminatorOp

//} Unrolled body 2 of loop302.
//DEBUG: /*idx302=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop302.
//DEBUG: /*idx302=*/ 2'd3, expected 3
//printTimeOffset
reg tloop308delay[1:0] = '{default:0} ;
always@(*) tloop308delay[0] <= tloop308;
generate
genvar i312;

for(i312 = 1; i312<= 1; i312= i312 + 1) begin
always@(posedge clk) begin
tloop308delay[i312] <= tloop308delay[i312-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop311 = tloop308delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v313 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][83] = tloop308delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 3][/*idx22=*/ 5][0] = tloop308delay[0];
assign v1_wr_data_valid[/*idx302=*/ 3][/*idx22=*/ 5][0] = tloop308delay[0];
assign v1_wr_data_input[/*idx302=*/ 3][/*idx22=*/ 5][0] = v313;


//TerminatorOp

//} Unrolled body 3 of loop302.
//DEBUG: /*idx302=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop302.
//DEBUG: /*idx302=*/ 3'd4, expected 4
//printTimeOffset
reg tloop311delay[1:0] = '{default:0} ;
always@(*) tloop311delay[0] <= tloop311;
generate
genvar i315;

for(i315 = 1; i315<= 1; i315= i315 + 1) begin
always@(posedge clk) begin
tloop311delay[i315] <= tloop311delay[i315-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop314 = tloop311delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v316 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][84] = tloop311delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 4][/*idx22=*/ 5][0] = tloop311delay[0];
assign v1_wr_data_valid[/*idx302=*/ 4][/*idx22=*/ 5][0] = tloop311delay[0];
assign v1_wr_data_input[/*idx302=*/ 4][/*idx22=*/ 5][0] = v316;


//TerminatorOp

//} Unrolled body 4 of loop302.
//DEBUG: /*idx302=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop302.
//DEBUG: /*idx302=*/ 3'd5, expected 5
//printTimeOffset
reg tloop314delay[1:0] = '{default:0} ;
always@(*) tloop314delay[0] <= tloop314;
generate
genvar i318;

for(i318 = 1; i318<= 1; i318= i318 + 1) begin
always@(posedge clk) begin
tloop314delay[i318] <= tloop314delay[i318-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop317 = tloop314delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v319 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][85] = tloop314delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 5][/*idx22=*/ 5][0] = tloop314delay[0];
assign v1_wr_data_valid[/*idx302=*/ 5][/*idx22=*/ 5][0] = tloop314delay[0];
assign v1_wr_data_input[/*idx302=*/ 5][/*idx22=*/ 5][0] = v319;


//TerminatorOp

//} Unrolled body 5 of loop302.
//DEBUG: /*idx302=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop302.
//DEBUG: /*idx302=*/ 3'd6, expected 6
//printTimeOffset
reg tloop317delay[1:0] = '{default:0} ;
always@(*) tloop317delay[0] <= tloop317;
generate
genvar i321;

for(i321 = 1; i321<= 1; i321= i321 + 1) begin
always@(posedge clk) begin
tloop317delay[i321] <= tloop317delay[i321-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop320 = tloop317delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v322 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][86] = tloop317delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 6][/*idx22=*/ 5][0] = tloop317delay[0];
assign v1_wr_data_valid[/*idx302=*/ 6][/*idx22=*/ 5][0] = tloop317delay[0];
assign v1_wr_data_input[/*idx302=*/ 6][/*idx22=*/ 5][0] = v322;


//TerminatorOp

//} Unrolled body 6 of loop302.
//DEBUG: /*idx302=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop302.
//DEBUG: /*idx302=*/ 3'd7, expected 7
//printTimeOffset
reg tloop320delay[1:0] = '{default:0} ;
always@(*) tloop320delay[0] <= tloop320;
generate
genvar i324;

for(i324 = 1; i324<= 1; i324= i324 + 1) begin
always@(posedge clk) begin
tloop320delay[i324] <= tloop320delay[i324-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop323 = tloop320delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v325 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][87] = tloop320delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 7][/*idx22=*/ 5][0] = tloop320delay[0];
assign v1_wr_data_valid[/*idx302=*/ 7][/*idx22=*/ 5][0] = tloop320delay[0];
assign v1_wr_data_input[/*idx302=*/ 7][/*idx22=*/ 5][0] = v325;


//TerminatorOp

//} Unrolled body 7 of loop302.
//DEBUG: /*idx302=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop302.
//DEBUG: /*idx302=*/ 4'd8, expected 8
//printTimeOffset
reg tloop323delay[1:0] = '{default:0} ;
always@(*) tloop323delay[0] <= tloop323;
generate
genvar i327;

for(i327 = 1; i327<= 1; i327= i327 + 1) begin
always@(posedge clk) begin
tloop323delay[i327] <= tloop323delay[i327-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop326 = tloop323delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v328 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][88] = tloop323delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 8][/*idx22=*/ 5][0] = tloop323delay[0];
assign v1_wr_data_valid[/*idx302=*/ 8][/*idx22=*/ 5][0] = tloop323delay[0];
assign v1_wr_data_input[/*idx302=*/ 8][/*idx22=*/ 5][0] = v328;


//TerminatorOp

//} Unrolled body 8 of loop302.
//DEBUG: /*idx302=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop302.
//DEBUG: /*idx302=*/ 4'd9, expected 9
//printTimeOffset
reg tloop326delay[1:0] = '{default:0} ;
always@(*) tloop326delay[0] <= tloop326;
generate
genvar i330;

for(i330 = 1; i330<= 1; i330= i330 + 1) begin
always@(posedge clk) begin
tloop326delay[i330] <= tloop326delay[i330-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop329 = tloop326delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v331 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][89] = tloop326delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 9][/*idx22=*/ 5][0] = tloop326delay[0];
assign v1_wr_data_valid[/*idx302=*/ 9][/*idx22=*/ 5][0] = tloop326delay[0];
assign v1_wr_data_input[/*idx302=*/ 9][/*idx22=*/ 5][0] = v331;


//TerminatorOp

//} Unrolled body 9 of loop302.
//DEBUG: /*idx302=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop302.
//DEBUG: /*idx302=*/ 4'd10, expected 10
//printTimeOffset
reg tloop329delay[1:0] = '{default:0} ;
always@(*) tloop329delay[0] <= tloop329;
generate
genvar i333;

for(i333 = 1; i333<= 1; i333= i333 + 1) begin
always@(posedge clk) begin
tloop329delay[i333] <= tloop329delay[i333-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop332 = tloop329delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v334 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][90] = tloop329delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 10][/*idx22=*/ 5][0] = tloop329delay[0];
assign v1_wr_data_valid[/*idx302=*/ 10][/*idx22=*/ 5][0] = tloop329delay[0];
assign v1_wr_data_input[/*idx302=*/ 10][/*idx22=*/ 5][0] = v334;


//TerminatorOp

//} Unrolled body 10 of loop302.
//DEBUG: /*idx302=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop302.
//DEBUG: /*idx302=*/ 4'd11, expected 11
//printTimeOffset
reg tloop332delay[1:0] = '{default:0} ;
always@(*) tloop332delay[0] <= tloop332;
generate
genvar i336;

for(i336 = 1; i336<= 1; i336= i336 + 1) begin
always@(posedge clk) begin
tloop332delay[i336] <= tloop332delay[i336-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop335 = tloop332delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v337 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][91] = tloop332delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 11][/*idx22=*/ 5][0] = tloop332delay[0];
assign v1_wr_data_valid[/*idx302=*/ 11][/*idx22=*/ 5][0] = tloop332delay[0];
assign v1_wr_data_input[/*idx302=*/ 11][/*idx22=*/ 5][0] = v337;


//TerminatorOp

//} Unrolled body 11 of loop302.
//DEBUG: /*idx302=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop302.
//DEBUG: /*idx302=*/ 4'd12, expected 12
//printTimeOffset
reg tloop335delay[1:0] = '{default:0} ;
always@(*) tloop335delay[0] <= tloop335;
generate
genvar i339;

for(i339 = 1; i339<= 1; i339= i339 + 1) begin
always@(posedge clk) begin
tloop335delay[i339] <= tloop335delay[i339-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop338 = tloop335delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v340 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][92] = tloop335delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 12][/*idx22=*/ 5][0] = tloop335delay[0];
assign v1_wr_data_valid[/*idx302=*/ 12][/*idx22=*/ 5][0] = tloop335delay[0];
assign v1_wr_data_input[/*idx302=*/ 12][/*idx22=*/ 5][0] = v340;


//TerminatorOp

//} Unrolled body 12 of loop302.
//DEBUG: /*idx302=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop302.
//DEBUG: /*idx302=*/ 4'd13, expected 13
//printTimeOffset
reg tloop338delay[1:0] = '{default:0} ;
always@(*) tloop338delay[0] <= tloop338;
generate
genvar i342;

for(i342 = 1; i342<= 1; i342= i342 + 1) begin
always@(posedge clk) begin
tloop338delay[i342] <= tloop338delay[i342-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop341 = tloop338delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v343 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][93] = tloop338delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 13][/*idx22=*/ 5][0] = tloop338delay[0];
assign v1_wr_data_valid[/*idx302=*/ 13][/*idx22=*/ 5][0] = tloop338delay[0];
assign v1_wr_data_input[/*idx302=*/ 13][/*idx22=*/ 5][0] = v343;


//TerminatorOp

//} Unrolled body 13 of loop302.
//DEBUG: /*idx302=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop302.
//DEBUG: /*idx302=*/ 4'd14, expected 14
//printTimeOffset
reg tloop341delay[1:0] = '{default:0} ;
always@(*) tloop341delay[0] <= tloop341;
generate
genvar i345;

for(i345 = 1; i345<= 1; i345= i345 + 1) begin
always@(posedge clk) begin
tloop341delay[i345] <= tloop341delay[i345-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop344 = tloop341delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v346 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][94] = tloop341delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 14][/*idx22=*/ 5][0] = tloop341delay[0];
assign v1_wr_data_valid[/*idx302=*/ 14][/*idx22=*/ 5][0] = tloop341delay[0];
assign v1_wr_data_input[/*idx302=*/ 14][/*idx22=*/ 5][0] = v346;


//TerminatorOp

//} Unrolled body 14 of loop302.
//DEBUG: /*idx302=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop302.
//DEBUG: /*idx302=*/ 4'd15, expected 15
//printTimeOffset
reg tloop344delay[1:0] = '{default:0} ;
always@(*) tloop344delay[0] <= tloop344;
generate
genvar i348;

for(i348 = 1; i348<= 1; i348= i348 + 1) begin
always@(posedge clk) begin
tloop344delay[i348] <= tloop344delay[i348-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop347 = tloop344delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v349 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][95] = tloop344delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx302=*/ 15][/*idx22=*/ 5][0] = tloop344delay[0];
assign v1_wr_data_valid[/*idx302=*/ 15][/*idx22=*/ 5][0] = tloop344delay[0];
assign v1_wr_data_input[/*idx302=*/ 15][/*idx22=*/ 5][0] = v349;


//TerminatorOp

//} Unrolled body 15 of loop302.
//DEBUG: /*idx302=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t350;
assign t350 = tloop347;
//printTimeOffset
reg t350delay[1:0] = '{default:0} ;
always@(*) t350delay[0] <= t350;
generate
genvar i351;

for(i351 = 1; i351<= 1; i351= i351 + 1) begin
always@(posedge clk) begin
t350delay[i351] <= t350delay[i351-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop297 = t350delay[1];

//TerminatorOp

//} Unrolled body 5 of loop22.
//DEBUG: /*idx22=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop22.
//DEBUG: /*idx22=*/ 3'd6, expected 6
//printTimeOffset
reg tloop297delay[0:0] = '{default:0} ;
always@(*) tloop297delay[0] <= tloop297;
generate
genvar i353;

for(i353 = 1; i353<= 0; i353= i353 + 1) begin
always@(posedge clk) begin
tloop297delay[i353] <= tloop297delay[i353-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg355[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg355[0] <= tloop297;
always@(posedge clk) shiftreg355[/*v5=*/ 1:1] <= shiftreg355[/*v5=*/ 0:0];
wire v354 = shiftreg355[/*v5=*/ 1];
//printTimeOffset
reg v354delay[1:0] = '{default:0} ;
always@(*) v354delay[0] <= v354;
generate
genvar i356;

for(i356 = 1; i356<= 1; i356= i356 + 1) begin
always@(posedge clk) begin
v354delay[i356] <= v354delay[i356-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop357.
//DEBUG: /*idx357=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop358 = v354delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v359 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][96] = v354delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 0][/*idx22=*/ 6][0] = v354delay[0];
assign v1_wr_data_valid[/*idx357=*/ 0][/*idx22=*/ 6][0] = v354delay[0];
assign v1_wr_data_input[/*idx357=*/ 0][/*idx22=*/ 6][0] = v359;


//TerminatorOp

//} Unrolled body 0 of loop357.
//DEBUG: /*idx357=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop357.
//DEBUG: /*idx357=*/ 1'd1, expected 1
//printTimeOffset
reg tloop358delay[1:0] = '{default:0} ;
always@(*) tloop358delay[0] <= tloop358;
generate
genvar i361;

for(i361 = 1; i361<= 1; i361= i361 + 1) begin
always@(posedge clk) begin
tloop358delay[i361] <= tloop358delay[i361-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop360 = tloop358delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v362 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][97] = tloop358delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 1][/*idx22=*/ 6][0] = tloop358delay[0];
assign v1_wr_data_valid[/*idx357=*/ 1][/*idx22=*/ 6][0] = tloop358delay[0];
assign v1_wr_data_input[/*idx357=*/ 1][/*idx22=*/ 6][0] = v362;


//TerminatorOp

//} Unrolled body 1 of loop357.
//DEBUG: /*idx357=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop357.
//DEBUG: /*idx357=*/ 2'd2, expected 2
//printTimeOffset
reg tloop360delay[1:0] = '{default:0} ;
always@(*) tloop360delay[0] <= tloop360;
generate
genvar i364;

for(i364 = 1; i364<= 1; i364= i364 + 1) begin
always@(posedge clk) begin
tloop360delay[i364] <= tloop360delay[i364-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop363 = tloop360delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v365 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][98] = tloop360delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 2][/*idx22=*/ 6][0] = tloop360delay[0];
assign v1_wr_data_valid[/*idx357=*/ 2][/*idx22=*/ 6][0] = tloop360delay[0];
assign v1_wr_data_input[/*idx357=*/ 2][/*idx22=*/ 6][0] = v365;


//TerminatorOp

//} Unrolled body 2 of loop357.
//DEBUG: /*idx357=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop357.
//DEBUG: /*idx357=*/ 2'd3, expected 3
//printTimeOffset
reg tloop363delay[1:0] = '{default:0} ;
always@(*) tloop363delay[0] <= tloop363;
generate
genvar i367;

for(i367 = 1; i367<= 1; i367= i367 + 1) begin
always@(posedge clk) begin
tloop363delay[i367] <= tloop363delay[i367-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop366 = tloop363delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v368 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][99] = tloop363delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 3][/*idx22=*/ 6][0] = tloop363delay[0];
assign v1_wr_data_valid[/*idx357=*/ 3][/*idx22=*/ 6][0] = tloop363delay[0];
assign v1_wr_data_input[/*idx357=*/ 3][/*idx22=*/ 6][0] = v368;


//TerminatorOp

//} Unrolled body 3 of loop357.
//DEBUG: /*idx357=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop357.
//DEBUG: /*idx357=*/ 3'd4, expected 4
//printTimeOffset
reg tloop366delay[1:0] = '{default:0} ;
always@(*) tloop366delay[0] <= tloop366;
generate
genvar i370;

for(i370 = 1; i370<= 1; i370= i370 + 1) begin
always@(posedge clk) begin
tloop366delay[i370] <= tloop366delay[i370-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop369 = tloop366delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v371 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][100] = tloop366delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 4][/*idx22=*/ 6][0] = tloop366delay[0];
assign v1_wr_data_valid[/*idx357=*/ 4][/*idx22=*/ 6][0] = tloop366delay[0];
assign v1_wr_data_input[/*idx357=*/ 4][/*idx22=*/ 6][0] = v371;


//TerminatorOp

//} Unrolled body 4 of loop357.
//DEBUG: /*idx357=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop357.
//DEBUG: /*idx357=*/ 3'd5, expected 5
//printTimeOffset
reg tloop369delay[1:0] = '{default:0} ;
always@(*) tloop369delay[0] <= tloop369;
generate
genvar i373;

for(i373 = 1; i373<= 1; i373= i373 + 1) begin
always@(posedge clk) begin
tloop369delay[i373] <= tloop369delay[i373-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop372 = tloop369delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v374 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][101] = tloop369delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 5][/*idx22=*/ 6][0] = tloop369delay[0];
assign v1_wr_data_valid[/*idx357=*/ 5][/*idx22=*/ 6][0] = tloop369delay[0];
assign v1_wr_data_input[/*idx357=*/ 5][/*idx22=*/ 6][0] = v374;


//TerminatorOp

//} Unrolled body 5 of loop357.
//DEBUG: /*idx357=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop357.
//DEBUG: /*idx357=*/ 3'd6, expected 6
//printTimeOffset
reg tloop372delay[1:0] = '{default:0} ;
always@(*) tloop372delay[0] <= tloop372;
generate
genvar i376;

for(i376 = 1; i376<= 1; i376= i376 + 1) begin
always@(posedge clk) begin
tloop372delay[i376] <= tloop372delay[i376-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop375 = tloop372delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v377 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][102] = tloop372delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 6][/*idx22=*/ 6][0] = tloop372delay[0];
assign v1_wr_data_valid[/*idx357=*/ 6][/*idx22=*/ 6][0] = tloop372delay[0];
assign v1_wr_data_input[/*idx357=*/ 6][/*idx22=*/ 6][0] = v377;


//TerminatorOp

//} Unrolled body 6 of loop357.
//DEBUG: /*idx357=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop357.
//DEBUG: /*idx357=*/ 3'd7, expected 7
//printTimeOffset
reg tloop375delay[1:0] = '{default:0} ;
always@(*) tloop375delay[0] <= tloop375;
generate
genvar i379;

for(i379 = 1; i379<= 1; i379= i379 + 1) begin
always@(posedge clk) begin
tloop375delay[i379] <= tloop375delay[i379-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop378 = tloop375delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v380 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][103] = tloop375delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 7][/*idx22=*/ 6][0] = tloop375delay[0];
assign v1_wr_data_valid[/*idx357=*/ 7][/*idx22=*/ 6][0] = tloop375delay[0];
assign v1_wr_data_input[/*idx357=*/ 7][/*idx22=*/ 6][0] = v380;


//TerminatorOp

//} Unrolled body 7 of loop357.
//DEBUG: /*idx357=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop357.
//DEBUG: /*idx357=*/ 4'd8, expected 8
//printTimeOffset
reg tloop378delay[1:0] = '{default:0} ;
always@(*) tloop378delay[0] <= tloop378;
generate
genvar i382;

for(i382 = 1; i382<= 1; i382= i382 + 1) begin
always@(posedge clk) begin
tloop378delay[i382] <= tloop378delay[i382-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop381 = tloop378delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v383 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][104] = tloop378delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 8][/*idx22=*/ 6][0] = tloop378delay[0];
assign v1_wr_data_valid[/*idx357=*/ 8][/*idx22=*/ 6][0] = tloop378delay[0];
assign v1_wr_data_input[/*idx357=*/ 8][/*idx22=*/ 6][0] = v383;


//TerminatorOp

//} Unrolled body 8 of loop357.
//DEBUG: /*idx357=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop357.
//DEBUG: /*idx357=*/ 4'd9, expected 9
//printTimeOffset
reg tloop381delay[1:0] = '{default:0} ;
always@(*) tloop381delay[0] <= tloop381;
generate
genvar i385;

for(i385 = 1; i385<= 1; i385= i385 + 1) begin
always@(posedge clk) begin
tloop381delay[i385] <= tloop381delay[i385-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop384 = tloop381delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v386 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][105] = tloop381delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 9][/*idx22=*/ 6][0] = tloop381delay[0];
assign v1_wr_data_valid[/*idx357=*/ 9][/*idx22=*/ 6][0] = tloop381delay[0];
assign v1_wr_data_input[/*idx357=*/ 9][/*idx22=*/ 6][0] = v386;


//TerminatorOp

//} Unrolled body 9 of loop357.
//DEBUG: /*idx357=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop357.
//DEBUG: /*idx357=*/ 4'd10, expected 10
//printTimeOffset
reg tloop384delay[1:0] = '{default:0} ;
always@(*) tloop384delay[0] <= tloop384;
generate
genvar i388;

for(i388 = 1; i388<= 1; i388= i388 + 1) begin
always@(posedge clk) begin
tloop384delay[i388] <= tloop384delay[i388-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop387 = tloop384delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v389 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][106] = tloop384delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 10][/*idx22=*/ 6][0] = tloop384delay[0];
assign v1_wr_data_valid[/*idx357=*/ 10][/*idx22=*/ 6][0] = tloop384delay[0];
assign v1_wr_data_input[/*idx357=*/ 10][/*idx22=*/ 6][0] = v389;


//TerminatorOp

//} Unrolled body 10 of loop357.
//DEBUG: /*idx357=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop357.
//DEBUG: /*idx357=*/ 4'd11, expected 11
//printTimeOffset
reg tloop387delay[1:0] = '{default:0} ;
always@(*) tloop387delay[0] <= tloop387;
generate
genvar i391;

for(i391 = 1; i391<= 1; i391= i391 + 1) begin
always@(posedge clk) begin
tloop387delay[i391] <= tloop387delay[i391-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop390 = tloop387delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v392 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][107] = tloop387delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 11][/*idx22=*/ 6][0] = tloop387delay[0];
assign v1_wr_data_valid[/*idx357=*/ 11][/*idx22=*/ 6][0] = tloop387delay[0];
assign v1_wr_data_input[/*idx357=*/ 11][/*idx22=*/ 6][0] = v392;


//TerminatorOp

//} Unrolled body 11 of loop357.
//DEBUG: /*idx357=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop357.
//DEBUG: /*idx357=*/ 4'd12, expected 12
//printTimeOffset
reg tloop390delay[1:0] = '{default:0} ;
always@(*) tloop390delay[0] <= tloop390;
generate
genvar i394;

for(i394 = 1; i394<= 1; i394= i394 + 1) begin
always@(posedge clk) begin
tloop390delay[i394] <= tloop390delay[i394-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop393 = tloop390delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v395 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][108] = tloop390delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 12][/*idx22=*/ 6][0] = tloop390delay[0];
assign v1_wr_data_valid[/*idx357=*/ 12][/*idx22=*/ 6][0] = tloop390delay[0];
assign v1_wr_data_input[/*idx357=*/ 12][/*idx22=*/ 6][0] = v395;


//TerminatorOp

//} Unrolled body 12 of loop357.
//DEBUG: /*idx357=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop357.
//DEBUG: /*idx357=*/ 4'd13, expected 13
//printTimeOffset
reg tloop393delay[1:0] = '{default:0} ;
always@(*) tloop393delay[0] <= tloop393;
generate
genvar i397;

for(i397 = 1; i397<= 1; i397= i397 + 1) begin
always@(posedge clk) begin
tloop393delay[i397] <= tloop393delay[i397-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop396 = tloop393delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v398 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][109] = tloop393delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 13][/*idx22=*/ 6][0] = tloop393delay[0];
assign v1_wr_data_valid[/*idx357=*/ 13][/*idx22=*/ 6][0] = tloop393delay[0];
assign v1_wr_data_input[/*idx357=*/ 13][/*idx22=*/ 6][0] = v398;


//TerminatorOp

//} Unrolled body 13 of loop357.
//DEBUG: /*idx357=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop357.
//DEBUG: /*idx357=*/ 4'd14, expected 14
//printTimeOffset
reg tloop396delay[1:0] = '{default:0} ;
always@(*) tloop396delay[0] <= tloop396;
generate
genvar i400;

for(i400 = 1; i400<= 1; i400= i400 + 1) begin
always@(posedge clk) begin
tloop396delay[i400] <= tloop396delay[i400-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop399 = tloop396delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v401 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][110] = tloop396delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 14][/*idx22=*/ 6][0] = tloop396delay[0];
assign v1_wr_data_valid[/*idx357=*/ 14][/*idx22=*/ 6][0] = tloop396delay[0];
assign v1_wr_data_input[/*idx357=*/ 14][/*idx22=*/ 6][0] = v401;


//TerminatorOp

//} Unrolled body 14 of loop357.
//DEBUG: /*idx357=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop357.
//DEBUG: /*idx357=*/ 4'd15, expected 15
//printTimeOffset
reg tloop399delay[1:0] = '{default:0} ;
always@(*) tloop399delay[0] <= tloop399;
generate
genvar i403;

for(i403 = 1; i403<= 1; i403= i403 + 1) begin
always@(posedge clk) begin
tloop399delay[i403] <= tloop399delay[i403-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop402 = tloop399delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v404 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][111] = tloop399delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx357=*/ 15][/*idx22=*/ 6][0] = tloop399delay[0];
assign v1_wr_data_valid[/*idx357=*/ 15][/*idx22=*/ 6][0] = tloop399delay[0];
assign v1_wr_data_input[/*idx357=*/ 15][/*idx22=*/ 6][0] = v404;


//TerminatorOp

//} Unrolled body 15 of loop357.
//DEBUG: /*idx357=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t405;
assign t405 = tloop402;
//printTimeOffset
reg t405delay[1:0] = '{default:0} ;
always@(*) t405delay[0] <= t405;
generate
genvar i406;

for(i406 = 1; i406<= 1; i406= i406 + 1) begin
always@(posedge clk) begin
t405delay[i406] <= t405delay[i406-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop352 = t405delay[1];

//TerminatorOp

//} Unrolled body 6 of loop22.
//DEBUG: /*idx22=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop22.
//DEBUG: /*idx22=*/ 3'd7, expected 7
//printTimeOffset
reg tloop352delay[0:0] = '{default:0} ;
always@(*) tloop352delay[0] <= tloop352;
generate
genvar i408;

for(i408 = 1; i408<= 0; i408= i408 + 1) begin
always@(posedge clk) begin
tloop352delay[i408] <= tloop352delay[i408-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg410[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg410[0] <= tloop352;
always@(posedge clk) shiftreg410[/*v5=*/ 1:1] <= shiftreg410[/*v5=*/ 0:0];
wire v409 = shiftreg410[/*v5=*/ 1];
//printTimeOffset
reg v409delay[1:0] = '{default:0} ;
always@(*) v409delay[0] <= v409;
generate
genvar i411;

for(i411 = 1; i411<= 1; i411= i411 + 1) begin
always@(posedge clk) begin
v409delay[i411] <= v409delay[i411-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop412.
//DEBUG: /*idx412=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop413 = v409delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v414 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][112] = v409delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 0][/*idx22=*/ 7][0] = v409delay[0];
assign v1_wr_data_valid[/*idx412=*/ 0][/*idx22=*/ 7][0] = v409delay[0];
assign v1_wr_data_input[/*idx412=*/ 0][/*idx22=*/ 7][0] = v414;


//TerminatorOp

//} Unrolled body 0 of loop412.
//DEBUG: /*idx412=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop412.
//DEBUG: /*idx412=*/ 1'd1, expected 1
//printTimeOffset
reg tloop413delay[1:0] = '{default:0} ;
always@(*) tloop413delay[0] <= tloop413;
generate
genvar i416;

for(i416 = 1; i416<= 1; i416= i416 + 1) begin
always@(posedge clk) begin
tloop413delay[i416] <= tloop413delay[i416-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop415 = tloop413delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v417 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][113] = tloop413delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 1][/*idx22=*/ 7][0] = tloop413delay[0];
assign v1_wr_data_valid[/*idx412=*/ 1][/*idx22=*/ 7][0] = tloop413delay[0];
assign v1_wr_data_input[/*idx412=*/ 1][/*idx22=*/ 7][0] = v417;


//TerminatorOp

//} Unrolled body 1 of loop412.
//DEBUG: /*idx412=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop412.
//DEBUG: /*idx412=*/ 2'd2, expected 2
//printTimeOffset
reg tloop415delay[1:0] = '{default:0} ;
always@(*) tloop415delay[0] <= tloop415;
generate
genvar i419;

for(i419 = 1; i419<= 1; i419= i419 + 1) begin
always@(posedge clk) begin
tloop415delay[i419] <= tloop415delay[i419-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop418 = tloop415delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v420 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][114] = tloop415delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 2][/*idx22=*/ 7][0] = tloop415delay[0];
assign v1_wr_data_valid[/*idx412=*/ 2][/*idx22=*/ 7][0] = tloop415delay[0];
assign v1_wr_data_input[/*idx412=*/ 2][/*idx22=*/ 7][0] = v420;


//TerminatorOp

//} Unrolled body 2 of loop412.
//DEBUG: /*idx412=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop412.
//DEBUG: /*idx412=*/ 2'd3, expected 3
//printTimeOffset
reg tloop418delay[1:0] = '{default:0} ;
always@(*) tloop418delay[0] <= tloop418;
generate
genvar i422;

for(i422 = 1; i422<= 1; i422= i422 + 1) begin
always@(posedge clk) begin
tloop418delay[i422] <= tloop418delay[i422-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop421 = tloop418delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v423 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][115] = tloop418delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 3][/*idx22=*/ 7][0] = tloop418delay[0];
assign v1_wr_data_valid[/*idx412=*/ 3][/*idx22=*/ 7][0] = tloop418delay[0];
assign v1_wr_data_input[/*idx412=*/ 3][/*idx22=*/ 7][0] = v423;


//TerminatorOp

//} Unrolled body 3 of loop412.
//DEBUG: /*idx412=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop412.
//DEBUG: /*idx412=*/ 3'd4, expected 4
//printTimeOffset
reg tloop421delay[1:0] = '{default:0} ;
always@(*) tloop421delay[0] <= tloop421;
generate
genvar i425;

for(i425 = 1; i425<= 1; i425= i425 + 1) begin
always@(posedge clk) begin
tloop421delay[i425] <= tloop421delay[i425-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop424 = tloop421delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v426 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][116] = tloop421delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 4][/*idx22=*/ 7][0] = tloop421delay[0];
assign v1_wr_data_valid[/*idx412=*/ 4][/*idx22=*/ 7][0] = tloop421delay[0];
assign v1_wr_data_input[/*idx412=*/ 4][/*idx22=*/ 7][0] = v426;


//TerminatorOp

//} Unrolled body 4 of loop412.
//DEBUG: /*idx412=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop412.
//DEBUG: /*idx412=*/ 3'd5, expected 5
//printTimeOffset
reg tloop424delay[1:0] = '{default:0} ;
always@(*) tloop424delay[0] <= tloop424;
generate
genvar i428;

for(i428 = 1; i428<= 1; i428= i428 + 1) begin
always@(posedge clk) begin
tloop424delay[i428] <= tloop424delay[i428-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop427 = tloop424delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v429 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][117] = tloop424delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 5][/*idx22=*/ 7][0] = tloop424delay[0];
assign v1_wr_data_valid[/*idx412=*/ 5][/*idx22=*/ 7][0] = tloop424delay[0];
assign v1_wr_data_input[/*idx412=*/ 5][/*idx22=*/ 7][0] = v429;


//TerminatorOp

//} Unrolled body 5 of loop412.
//DEBUG: /*idx412=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop412.
//DEBUG: /*idx412=*/ 3'd6, expected 6
//printTimeOffset
reg tloop427delay[1:0] = '{default:0} ;
always@(*) tloop427delay[0] <= tloop427;
generate
genvar i431;

for(i431 = 1; i431<= 1; i431= i431 + 1) begin
always@(posedge clk) begin
tloop427delay[i431] <= tloop427delay[i431-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop430 = tloop427delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v432 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][118] = tloop427delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 6][/*idx22=*/ 7][0] = tloop427delay[0];
assign v1_wr_data_valid[/*idx412=*/ 6][/*idx22=*/ 7][0] = tloop427delay[0];
assign v1_wr_data_input[/*idx412=*/ 6][/*idx22=*/ 7][0] = v432;


//TerminatorOp

//} Unrolled body 6 of loop412.
//DEBUG: /*idx412=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop412.
//DEBUG: /*idx412=*/ 3'd7, expected 7
//printTimeOffset
reg tloop430delay[1:0] = '{default:0} ;
always@(*) tloop430delay[0] <= tloop430;
generate
genvar i434;

for(i434 = 1; i434<= 1; i434= i434 + 1) begin
always@(posedge clk) begin
tloop430delay[i434] <= tloop430delay[i434-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop433 = tloop430delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v435 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][119] = tloop430delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 7][/*idx22=*/ 7][0] = tloop430delay[0];
assign v1_wr_data_valid[/*idx412=*/ 7][/*idx22=*/ 7][0] = tloop430delay[0];
assign v1_wr_data_input[/*idx412=*/ 7][/*idx22=*/ 7][0] = v435;


//TerminatorOp

//} Unrolled body 7 of loop412.
//DEBUG: /*idx412=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop412.
//DEBUG: /*idx412=*/ 4'd8, expected 8
//printTimeOffset
reg tloop433delay[1:0] = '{default:0} ;
always@(*) tloop433delay[0] <= tloop433;
generate
genvar i437;

for(i437 = 1; i437<= 1; i437= i437 + 1) begin
always@(posedge clk) begin
tloop433delay[i437] <= tloop433delay[i437-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop436 = tloop433delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v438 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][120] = tloop433delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 8][/*idx22=*/ 7][0] = tloop433delay[0];
assign v1_wr_data_valid[/*idx412=*/ 8][/*idx22=*/ 7][0] = tloop433delay[0];
assign v1_wr_data_input[/*idx412=*/ 8][/*idx22=*/ 7][0] = v438;


//TerminatorOp

//} Unrolled body 8 of loop412.
//DEBUG: /*idx412=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop412.
//DEBUG: /*idx412=*/ 4'd9, expected 9
//printTimeOffset
reg tloop436delay[1:0] = '{default:0} ;
always@(*) tloop436delay[0] <= tloop436;
generate
genvar i440;

for(i440 = 1; i440<= 1; i440= i440 + 1) begin
always@(posedge clk) begin
tloop436delay[i440] <= tloop436delay[i440-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop439 = tloop436delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v441 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][121] = tloop436delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 9][/*idx22=*/ 7][0] = tloop436delay[0];
assign v1_wr_data_valid[/*idx412=*/ 9][/*idx22=*/ 7][0] = tloop436delay[0];
assign v1_wr_data_input[/*idx412=*/ 9][/*idx22=*/ 7][0] = v441;


//TerminatorOp

//} Unrolled body 9 of loop412.
//DEBUG: /*idx412=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop412.
//DEBUG: /*idx412=*/ 4'd10, expected 10
//printTimeOffset
reg tloop439delay[1:0] = '{default:0} ;
always@(*) tloop439delay[0] <= tloop439;
generate
genvar i443;

for(i443 = 1; i443<= 1; i443= i443 + 1) begin
always@(posedge clk) begin
tloop439delay[i443] <= tloop439delay[i443-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop442 = tloop439delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v444 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][122] = tloop439delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 10][/*idx22=*/ 7][0] = tloop439delay[0];
assign v1_wr_data_valid[/*idx412=*/ 10][/*idx22=*/ 7][0] = tloop439delay[0];
assign v1_wr_data_input[/*idx412=*/ 10][/*idx22=*/ 7][0] = v444;


//TerminatorOp

//} Unrolled body 10 of loop412.
//DEBUG: /*idx412=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop412.
//DEBUG: /*idx412=*/ 4'd11, expected 11
//printTimeOffset
reg tloop442delay[1:0] = '{default:0} ;
always@(*) tloop442delay[0] <= tloop442;
generate
genvar i446;

for(i446 = 1; i446<= 1; i446= i446 + 1) begin
always@(posedge clk) begin
tloop442delay[i446] <= tloop442delay[i446-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop445 = tloop442delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v447 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][123] = tloop442delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 11][/*idx22=*/ 7][0] = tloop442delay[0];
assign v1_wr_data_valid[/*idx412=*/ 11][/*idx22=*/ 7][0] = tloop442delay[0];
assign v1_wr_data_input[/*idx412=*/ 11][/*idx22=*/ 7][0] = v447;


//TerminatorOp

//} Unrolled body 11 of loop412.
//DEBUG: /*idx412=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop412.
//DEBUG: /*idx412=*/ 4'd12, expected 12
//printTimeOffset
reg tloop445delay[1:0] = '{default:0} ;
always@(*) tloop445delay[0] <= tloop445;
generate
genvar i449;

for(i449 = 1; i449<= 1; i449= i449 + 1) begin
always@(posedge clk) begin
tloop445delay[i449] <= tloop445delay[i449-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop448 = tloop445delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v450 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][124] = tloop445delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 12][/*idx22=*/ 7][0] = tloop445delay[0];
assign v1_wr_data_valid[/*idx412=*/ 12][/*idx22=*/ 7][0] = tloop445delay[0];
assign v1_wr_data_input[/*idx412=*/ 12][/*idx22=*/ 7][0] = v450;


//TerminatorOp

//} Unrolled body 12 of loop412.
//DEBUG: /*idx412=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop412.
//DEBUG: /*idx412=*/ 4'd13, expected 13
//printTimeOffset
reg tloop448delay[1:0] = '{default:0} ;
always@(*) tloop448delay[0] <= tloop448;
generate
genvar i452;

for(i452 = 1; i452<= 1; i452= i452 + 1) begin
always@(posedge clk) begin
tloop448delay[i452] <= tloop448delay[i452-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop451 = tloop448delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v453 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][125] = tloop448delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 13][/*idx22=*/ 7][0] = tloop448delay[0];
assign v1_wr_data_valid[/*idx412=*/ 13][/*idx22=*/ 7][0] = tloop448delay[0];
assign v1_wr_data_input[/*idx412=*/ 13][/*idx22=*/ 7][0] = v453;


//TerminatorOp

//} Unrolled body 13 of loop412.
//DEBUG: /*idx412=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop412.
//DEBUG: /*idx412=*/ 4'd14, expected 14
//printTimeOffset
reg tloop451delay[1:0] = '{default:0} ;
always@(*) tloop451delay[0] <= tloop451;
generate
genvar i455;

for(i455 = 1; i455<= 1; i455= i455 + 1) begin
always@(posedge clk) begin
tloop451delay[i455] <= tloop451delay[i455-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop454 = tloop451delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v456 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][126] = tloop451delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 14][/*idx22=*/ 7][0] = tloop451delay[0];
assign v1_wr_data_valid[/*idx412=*/ 14][/*idx22=*/ 7][0] = tloop451delay[0];
assign v1_wr_data_input[/*idx412=*/ 14][/*idx22=*/ 7][0] = v456;


//TerminatorOp

//} Unrolled body 14 of loop412.
//DEBUG: /*idx412=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop412.
//DEBUG: /*idx412=*/ 4'd15, expected 15
//printTimeOffset
reg tloop454delay[1:0] = '{default:0} ;
always@(*) tloop454delay[0] <= tloop454;
generate
genvar i458;

for(i458 = 1; i458<= 1; i458= i458 + 1) begin
always@(posedge clk) begin
tloop454delay[i458] <= tloop454delay[i458-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop457 = tloop454delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v459 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][127] = tloop454delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx412=*/ 15][/*idx22=*/ 7][0] = tloop454delay[0];
assign v1_wr_data_valid[/*idx412=*/ 15][/*idx22=*/ 7][0] = tloop454delay[0];
assign v1_wr_data_input[/*idx412=*/ 15][/*idx22=*/ 7][0] = v459;


//TerminatorOp

//} Unrolled body 15 of loop412.
//DEBUG: /*idx412=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t460;
assign t460 = tloop457;
//printTimeOffset
reg t460delay[1:0] = '{default:0} ;
always@(*) t460delay[0] <= t460;
generate
genvar i461;

for(i461 = 1; i461<= 1; i461= i461 + 1) begin
always@(posedge clk) begin
t460delay[i461] <= t460delay[i461-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop407 = t460delay[1];

//TerminatorOp

//} Unrolled body 7 of loop22.
//DEBUG: /*idx22=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop22.
//DEBUG: /*idx22=*/ 4'd8, expected 8
//printTimeOffset
reg tloop407delay[0:0] = '{default:0} ;
always@(*) tloop407delay[0] <= tloop407;
generate
genvar i463;

for(i463 = 1; i463<= 0; i463= i463 + 1) begin
always@(posedge clk) begin
tloop407delay[i463] <= tloop407delay[i463-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg465[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg465[0] <= tloop407;
always@(posedge clk) shiftreg465[/*v5=*/ 1:1] <= shiftreg465[/*v5=*/ 0:0];
wire v464 = shiftreg465[/*v5=*/ 1];
//printTimeOffset
reg v464delay[1:0] = '{default:0} ;
always@(*) v464delay[0] <= v464;
generate
genvar i466;

for(i466 = 1; i466<= 1; i466= i466 + 1) begin
always@(posedge clk) begin
v464delay[i466] <= v464delay[i466-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop467.
//DEBUG: /*idx467=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop468 = v464delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v469 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][128] = v464delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 0][/*idx22=*/ 8][0] = v464delay[0];
assign v1_wr_data_valid[/*idx467=*/ 0][/*idx22=*/ 8][0] = v464delay[0];
assign v1_wr_data_input[/*idx467=*/ 0][/*idx22=*/ 8][0] = v469;


//TerminatorOp

//} Unrolled body 0 of loop467.
//DEBUG: /*idx467=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop467.
//DEBUG: /*idx467=*/ 1'd1, expected 1
//printTimeOffset
reg tloop468delay[1:0] = '{default:0} ;
always@(*) tloop468delay[0] <= tloop468;
generate
genvar i471;

for(i471 = 1; i471<= 1; i471= i471 + 1) begin
always@(posedge clk) begin
tloop468delay[i471] <= tloop468delay[i471-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop470 = tloop468delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v472 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][129] = tloop468delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 1][/*idx22=*/ 8][0] = tloop468delay[0];
assign v1_wr_data_valid[/*idx467=*/ 1][/*idx22=*/ 8][0] = tloop468delay[0];
assign v1_wr_data_input[/*idx467=*/ 1][/*idx22=*/ 8][0] = v472;


//TerminatorOp

//} Unrolled body 1 of loop467.
//DEBUG: /*idx467=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop467.
//DEBUG: /*idx467=*/ 2'd2, expected 2
//printTimeOffset
reg tloop470delay[1:0] = '{default:0} ;
always@(*) tloop470delay[0] <= tloop470;
generate
genvar i474;

for(i474 = 1; i474<= 1; i474= i474 + 1) begin
always@(posedge clk) begin
tloop470delay[i474] <= tloop470delay[i474-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop473 = tloop470delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v475 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][130] = tloop470delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 2][/*idx22=*/ 8][0] = tloop470delay[0];
assign v1_wr_data_valid[/*idx467=*/ 2][/*idx22=*/ 8][0] = tloop470delay[0];
assign v1_wr_data_input[/*idx467=*/ 2][/*idx22=*/ 8][0] = v475;


//TerminatorOp

//} Unrolled body 2 of loop467.
//DEBUG: /*idx467=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop467.
//DEBUG: /*idx467=*/ 2'd3, expected 3
//printTimeOffset
reg tloop473delay[1:0] = '{default:0} ;
always@(*) tloop473delay[0] <= tloop473;
generate
genvar i477;

for(i477 = 1; i477<= 1; i477= i477 + 1) begin
always@(posedge clk) begin
tloop473delay[i477] <= tloop473delay[i477-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop476 = tloop473delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v478 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][131] = tloop473delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 3][/*idx22=*/ 8][0] = tloop473delay[0];
assign v1_wr_data_valid[/*idx467=*/ 3][/*idx22=*/ 8][0] = tloop473delay[0];
assign v1_wr_data_input[/*idx467=*/ 3][/*idx22=*/ 8][0] = v478;


//TerminatorOp

//} Unrolled body 3 of loop467.
//DEBUG: /*idx467=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop467.
//DEBUG: /*idx467=*/ 3'd4, expected 4
//printTimeOffset
reg tloop476delay[1:0] = '{default:0} ;
always@(*) tloop476delay[0] <= tloop476;
generate
genvar i480;

for(i480 = 1; i480<= 1; i480= i480 + 1) begin
always@(posedge clk) begin
tloop476delay[i480] <= tloop476delay[i480-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop479 = tloop476delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v481 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][132] = tloop476delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 4][/*idx22=*/ 8][0] = tloop476delay[0];
assign v1_wr_data_valid[/*idx467=*/ 4][/*idx22=*/ 8][0] = tloop476delay[0];
assign v1_wr_data_input[/*idx467=*/ 4][/*idx22=*/ 8][0] = v481;


//TerminatorOp

//} Unrolled body 4 of loop467.
//DEBUG: /*idx467=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop467.
//DEBUG: /*idx467=*/ 3'd5, expected 5
//printTimeOffset
reg tloop479delay[1:0] = '{default:0} ;
always@(*) tloop479delay[0] <= tloop479;
generate
genvar i483;

for(i483 = 1; i483<= 1; i483= i483 + 1) begin
always@(posedge clk) begin
tloop479delay[i483] <= tloop479delay[i483-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop482 = tloop479delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v484 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][133] = tloop479delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 5][/*idx22=*/ 8][0] = tloop479delay[0];
assign v1_wr_data_valid[/*idx467=*/ 5][/*idx22=*/ 8][0] = tloop479delay[0];
assign v1_wr_data_input[/*idx467=*/ 5][/*idx22=*/ 8][0] = v484;


//TerminatorOp

//} Unrolled body 5 of loop467.
//DEBUG: /*idx467=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop467.
//DEBUG: /*idx467=*/ 3'd6, expected 6
//printTimeOffset
reg tloop482delay[1:0] = '{default:0} ;
always@(*) tloop482delay[0] <= tloop482;
generate
genvar i486;

for(i486 = 1; i486<= 1; i486= i486 + 1) begin
always@(posedge clk) begin
tloop482delay[i486] <= tloop482delay[i486-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop485 = tloop482delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v487 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][134] = tloop482delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 6][/*idx22=*/ 8][0] = tloop482delay[0];
assign v1_wr_data_valid[/*idx467=*/ 6][/*idx22=*/ 8][0] = tloop482delay[0];
assign v1_wr_data_input[/*idx467=*/ 6][/*idx22=*/ 8][0] = v487;


//TerminatorOp

//} Unrolled body 6 of loop467.
//DEBUG: /*idx467=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop467.
//DEBUG: /*idx467=*/ 3'd7, expected 7
//printTimeOffset
reg tloop485delay[1:0] = '{default:0} ;
always@(*) tloop485delay[0] <= tloop485;
generate
genvar i489;

for(i489 = 1; i489<= 1; i489= i489 + 1) begin
always@(posedge clk) begin
tloop485delay[i489] <= tloop485delay[i489-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop488 = tloop485delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v490 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][135] = tloop485delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 7][/*idx22=*/ 8][0] = tloop485delay[0];
assign v1_wr_data_valid[/*idx467=*/ 7][/*idx22=*/ 8][0] = tloop485delay[0];
assign v1_wr_data_input[/*idx467=*/ 7][/*idx22=*/ 8][0] = v490;


//TerminatorOp

//} Unrolled body 7 of loop467.
//DEBUG: /*idx467=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop467.
//DEBUG: /*idx467=*/ 4'd8, expected 8
//printTimeOffset
reg tloop488delay[1:0] = '{default:0} ;
always@(*) tloop488delay[0] <= tloop488;
generate
genvar i492;

for(i492 = 1; i492<= 1; i492= i492 + 1) begin
always@(posedge clk) begin
tloop488delay[i492] <= tloop488delay[i492-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop491 = tloop488delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v493 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][136] = tloop488delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 8][/*idx22=*/ 8][0] = tloop488delay[0];
assign v1_wr_data_valid[/*idx467=*/ 8][/*idx22=*/ 8][0] = tloop488delay[0];
assign v1_wr_data_input[/*idx467=*/ 8][/*idx22=*/ 8][0] = v493;


//TerminatorOp

//} Unrolled body 8 of loop467.
//DEBUG: /*idx467=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop467.
//DEBUG: /*idx467=*/ 4'd9, expected 9
//printTimeOffset
reg tloop491delay[1:0] = '{default:0} ;
always@(*) tloop491delay[0] <= tloop491;
generate
genvar i495;

for(i495 = 1; i495<= 1; i495= i495 + 1) begin
always@(posedge clk) begin
tloop491delay[i495] <= tloop491delay[i495-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop494 = tloop491delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v496 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][137] = tloop491delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 9][/*idx22=*/ 8][0] = tloop491delay[0];
assign v1_wr_data_valid[/*idx467=*/ 9][/*idx22=*/ 8][0] = tloop491delay[0];
assign v1_wr_data_input[/*idx467=*/ 9][/*idx22=*/ 8][0] = v496;


//TerminatorOp

//} Unrolled body 9 of loop467.
//DEBUG: /*idx467=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop467.
//DEBUG: /*idx467=*/ 4'd10, expected 10
//printTimeOffset
reg tloop494delay[1:0] = '{default:0} ;
always@(*) tloop494delay[0] <= tloop494;
generate
genvar i498;

for(i498 = 1; i498<= 1; i498= i498 + 1) begin
always@(posedge clk) begin
tloop494delay[i498] <= tloop494delay[i498-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop497 = tloop494delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v499 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][138] = tloop494delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 10][/*idx22=*/ 8][0] = tloop494delay[0];
assign v1_wr_data_valid[/*idx467=*/ 10][/*idx22=*/ 8][0] = tloop494delay[0];
assign v1_wr_data_input[/*idx467=*/ 10][/*idx22=*/ 8][0] = v499;


//TerminatorOp

//} Unrolled body 10 of loop467.
//DEBUG: /*idx467=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop467.
//DEBUG: /*idx467=*/ 4'd11, expected 11
//printTimeOffset
reg tloop497delay[1:0] = '{default:0} ;
always@(*) tloop497delay[0] <= tloop497;
generate
genvar i501;

for(i501 = 1; i501<= 1; i501= i501 + 1) begin
always@(posedge clk) begin
tloop497delay[i501] <= tloop497delay[i501-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop500 = tloop497delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v502 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][139] = tloop497delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 11][/*idx22=*/ 8][0] = tloop497delay[0];
assign v1_wr_data_valid[/*idx467=*/ 11][/*idx22=*/ 8][0] = tloop497delay[0];
assign v1_wr_data_input[/*idx467=*/ 11][/*idx22=*/ 8][0] = v502;


//TerminatorOp

//} Unrolled body 11 of loop467.
//DEBUG: /*idx467=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop467.
//DEBUG: /*idx467=*/ 4'd12, expected 12
//printTimeOffset
reg tloop500delay[1:0] = '{default:0} ;
always@(*) tloop500delay[0] <= tloop500;
generate
genvar i504;

for(i504 = 1; i504<= 1; i504= i504 + 1) begin
always@(posedge clk) begin
tloop500delay[i504] <= tloop500delay[i504-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop503 = tloop500delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v505 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][140] = tloop500delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 12][/*idx22=*/ 8][0] = tloop500delay[0];
assign v1_wr_data_valid[/*idx467=*/ 12][/*idx22=*/ 8][0] = tloop500delay[0];
assign v1_wr_data_input[/*idx467=*/ 12][/*idx22=*/ 8][0] = v505;


//TerminatorOp

//} Unrolled body 12 of loop467.
//DEBUG: /*idx467=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop467.
//DEBUG: /*idx467=*/ 4'd13, expected 13
//printTimeOffset
reg tloop503delay[1:0] = '{default:0} ;
always@(*) tloop503delay[0] <= tloop503;
generate
genvar i507;

for(i507 = 1; i507<= 1; i507= i507 + 1) begin
always@(posedge clk) begin
tloop503delay[i507] <= tloop503delay[i507-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop506 = tloop503delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v508 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][141] = tloop503delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 13][/*idx22=*/ 8][0] = tloop503delay[0];
assign v1_wr_data_valid[/*idx467=*/ 13][/*idx22=*/ 8][0] = tloop503delay[0];
assign v1_wr_data_input[/*idx467=*/ 13][/*idx22=*/ 8][0] = v508;


//TerminatorOp

//} Unrolled body 13 of loop467.
//DEBUG: /*idx467=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop467.
//DEBUG: /*idx467=*/ 4'd14, expected 14
//printTimeOffset
reg tloop506delay[1:0] = '{default:0} ;
always@(*) tloop506delay[0] <= tloop506;
generate
genvar i510;

for(i510 = 1; i510<= 1; i510= i510 + 1) begin
always@(posedge clk) begin
tloop506delay[i510] <= tloop506delay[i510-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop509 = tloop506delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v511 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][142] = tloop506delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 14][/*idx22=*/ 8][0] = tloop506delay[0];
assign v1_wr_data_valid[/*idx467=*/ 14][/*idx22=*/ 8][0] = tloop506delay[0];
assign v1_wr_data_input[/*idx467=*/ 14][/*idx22=*/ 8][0] = v511;


//TerminatorOp

//} Unrolled body 14 of loop467.
//DEBUG: /*idx467=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop467.
//DEBUG: /*idx467=*/ 4'd15, expected 15
//printTimeOffset
reg tloop509delay[1:0] = '{default:0} ;
always@(*) tloop509delay[0] <= tloop509;
generate
genvar i513;

for(i513 = 1; i513<= 1; i513= i513 + 1) begin
always@(posedge clk) begin
tloop509delay[i513] <= tloop509delay[i513-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop512 = tloop509delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v514 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][143] = tloop509delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx467=*/ 15][/*idx22=*/ 8][0] = tloop509delay[0];
assign v1_wr_data_valid[/*idx467=*/ 15][/*idx22=*/ 8][0] = tloop509delay[0];
assign v1_wr_data_input[/*idx467=*/ 15][/*idx22=*/ 8][0] = v514;


//TerminatorOp

//} Unrolled body 15 of loop467.
//DEBUG: /*idx467=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t515;
assign t515 = tloop512;
//printTimeOffset
reg t515delay[1:0] = '{default:0} ;
always@(*) t515delay[0] <= t515;
generate
genvar i516;

for(i516 = 1; i516<= 1; i516= i516 + 1) begin
always@(posedge clk) begin
t515delay[i516] <= t515delay[i516-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop462 = t515delay[1];

//TerminatorOp

//} Unrolled body 8 of loop22.
//DEBUG: /*idx22=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop22.
//DEBUG: /*idx22=*/ 4'd9, expected 9
//printTimeOffset
reg tloop462delay[0:0] = '{default:0} ;
always@(*) tloop462delay[0] <= tloop462;
generate
genvar i518;

for(i518 = 1; i518<= 0; i518= i518 + 1) begin
always@(posedge clk) begin
tloop462delay[i518] <= tloop462delay[i518-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg520[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg520[0] <= tloop462;
always@(posedge clk) shiftreg520[/*v5=*/ 1:1] <= shiftreg520[/*v5=*/ 0:0];
wire v519 = shiftreg520[/*v5=*/ 1];
//printTimeOffset
reg v519delay[1:0] = '{default:0} ;
always@(*) v519delay[0] <= v519;
generate
genvar i521;

for(i521 = 1; i521<= 1; i521= i521 + 1) begin
always@(posedge clk) begin
v519delay[i521] <= v519delay[i521-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop522.
//DEBUG: /*idx522=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop523 = v519delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v524 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][144] = v519delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 0][/*idx22=*/ 9][0] = v519delay[0];
assign v1_wr_data_valid[/*idx522=*/ 0][/*idx22=*/ 9][0] = v519delay[0];
assign v1_wr_data_input[/*idx522=*/ 0][/*idx22=*/ 9][0] = v524;


//TerminatorOp

//} Unrolled body 0 of loop522.
//DEBUG: /*idx522=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop522.
//DEBUG: /*idx522=*/ 1'd1, expected 1
//printTimeOffset
reg tloop523delay[1:0] = '{default:0} ;
always@(*) tloop523delay[0] <= tloop523;
generate
genvar i526;

for(i526 = 1; i526<= 1; i526= i526 + 1) begin
always@(posedge clk) begin
tloop523delay[i526] <= tloop523delay[i526-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop525 = tloop523delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v527 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][145] = tloop523delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 1][/*idx22=*/ 9][0] = tloop523delay[0];
assign v1_wr_data_valid[/*idx522=*/ 1][/*idx22=*/ 9][0] = tloop523delay[0];
assign v1_wr_data_input[/*idx522=*/ 1][/*idx22=*/ 9][0] = v527;


//TerminatorOp

//} Unrolled body 1 of loop522.
//DEBUG: /*idx522=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop522.
//DEBUG: /*idx522=*/ 2'd2, expected 2
//printTimeOffset
reg tloop525delay[1:0] = '{default:0} ;
always@(*) tloop525delay[0] <= tloop525;
generate
genvar i529;

for(i529 = 1; i529<= 1; i529= i529 + 1) begin
always@(posedge clk) begin
tloop525delay[i529] <= tloop525delay[i529-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop528 = tloop525delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v530 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][146] = tloop525delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 2][/*idx22=*/ 9][0] = tloop525delay[0];
assign v1_wr_data_valid[/*idx522=*/ 2][/*idx22=*/ 9][0] = tloop525delay[0];
assign v1_wr_data_input[/*idx522=*/ 2][/*idx22=*/ 9][0] = v530;


//TerminatorOp

//} Unrolled body 2 of loop522.
//DEBUG: /*idx522=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop522.
//DEBUG: /*idx522=*/ 2'd3, expected 3
//printTimeOffset
reg tloop528delay[1:0] = '{default:0} ;
always@(*) tloop528delay[0] <= tloop528;
generate
genvar i532;

for(i532 = 1; i532<= 1; i532= i532 + 1) begin
always@(posedge clk) begin
tloop528delay[i532] <= tloop528delay[i532-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop531 = tloop528delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v533 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][147] = tloop528delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 3][/*idx22=*/ 9][0] = tloop528delay[0];
assign v1_wr_data_valid[/*idx522=*/ 3][/*idx22=*/ 9][0] = tloop528delay[0];
assign v1_wr_data_input[/*idx522=*/ 3][/*idx22=*/ 9][0] = v533;


//TerminatorOp

//} Unrolled body 3 of loop522.
//DEBUG: /*idx522=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop522.
//DEBUG: /*idx522=*/ 3'd4, expected 4
//printTimeOffset
reg tloop531delay[1:0] = '{default:0} ;
always@(*) tloop531delay[0] <= tloop531;
generate
genvar i535;

for(i535 = 1; i535<= 1; i535= i535 + 1) begin
always@(posedge clk) begin
tloop531delay[i535] <= tloop531delay[i535-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop534 = tloop531delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v536 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][148] = tloop531delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 4][/*idx22=*/ 9][0] = tloop531delay[0];
assign v1_wr_data_valid[/*idx522=*/ 4][/*idx22=*/ 9][0] = tloop531delay[0];
assign v1_wr_data_input[/*idx522=*/ 4][/*idx22=*/ 9][0] = v536;


//TerminatorOp

//} Unrolled body 4 of loop522.
//DEBUG: /*idx522=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop522.
//DEBUG: /*idx522=*/ 3'd5, expected 5
//printTimeOffset
reg tloop534delay[1:0] = '{default:0} ;
always@(*) tloop534delay[0] <= tloop534;
generate
genvar i538;

for(i538 = 1; i538<= 1; i538= i538 + 1) begin
always@(posedge clk) begin
tloop534delay[i538] <= tloop534delay[i538-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop537 = tloop534delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v539 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][149] = tloop534delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 5][/*idx22=*/ 9][0] = tloop534delay[0];
assign v1_wr_data_valid[/*idx522=*/ 5][/*idx22=*/ 9][0] = tloop534delay[0];
assign v1_wr_data_input[/*idx522=*/ 5][/*idx22=*/ 9][0] = v539;


//TerminatorOp

//} Unrolled body 5 of loop522.
//DEBUG: /*idx522=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop522.
//DEBUG: /*idx522=*/ 3'd6, expected 6
//printTimeOffset
reg tloop537delay[1:0] = '{default:0} ;
always@(*) tloop537delay[0] <= tloop537;
generate
genvar i541;

for(i541 = 1; i541<= 1; i541= i541 + 1) begin
always@(posedge clk) begin
tloop537delay[i541] <= tloop537delay[i541-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop540 = tloop537delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v542 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][150] = tloop537delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 6][/*idx22=*/ 9][0] = tloop537delay[0];
assign v1_wr_data_valid[/*idx522=*/ 6][/*idx22=*/ 9][0] = tloop537delay[0];
assign v1_wr_data_input[/*idx522=*/ 6][/*idx22=*/ 9][0] = v542;


//TerminatorOp

//} Unrolled body 6 of loop522.
//DEBUG: /*idx522=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop522.
//DEBUG: /*idx522=*/ 3'd7, expected 7
//printTimeOffset
reg tloop540delay[1:0] = '{default:0} ;
always@(*) tloop540delay[0] <= tloop540;
generate
genvar i544;

for(i544 = 1; i544<= 1; i544= i544 + 1) begin
always@(posedge clk) begin
tloop540delay[i544] <= tloop540delay[i544-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop543 = tloop540delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v545 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][151] = tloop540delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 7][/*idx22=*/ 9][0] = tloop540delay[0];
assign v1_wr_data_valid[/*idx522=*/ 7][/*idx22=*/ 9][0] = tloop540delay[0];
assign v1_wr_data_input[/*idx522=*/ 7][/*idx22=*/ 9][0] = v545;


//TerminatorOp

//} Unrolled body 7 of loop522.
//DEBUG: /*idx522=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop522.
//DEBUG: /*idx522=*/ 4'd8, expected 8
//printTimeOffset
reg tloop543delay[1:0] = '{default:0} ;
always@(*) tloop543delay[0] <= tloop543;
generate
genvar i547;

for(i547 = 1; i547<= 1; i547= i547 + 1) begin
always@(posedge clk) begin
tloop543delay[i547] <= tloop543delay[i547-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop546 = tloop543delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v548 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][152] = tloop543delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 8][/*idx22=*/ 9][0] = tloop543delay[0];
assign v1_wr_data_valid[/*idx522=*/ 8][/*idx22=*/ 9][0] = tloop543delay[0];
assign v1_wr_data_input[/*idx522=*/ 8][/*idx22=*/ 9][0] = v548;


//TerminatorOp

//} Unrolled body 8 of loop522.
//DEBUG: /*idx522=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop522.
//DEBUG: /*idx522=*/ 4'd9, expected 9
//printTimeOffset
reg tloop546delay[1:0] = '{default:0} ;
always@(*) tloop546delay[0] <= tloop546;
generate
genvar i550;

for(i550 = 1; i550<= 1; i550= i550 + 1) begin
always@(posedge clk) begin
tloop546delay[i550] <= tloop546delay[i550-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop549 = tloop546delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v551 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][153] = tloop546delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 9][/*idx22=*/ 9][0] = tloop546delay[0];
assign v1_wr_data_valid[/*idx522=*/ 9][/*idx22=*/ 9][0] = tloop546delay[0];
assign v1_wr_data_input[/*idx522=*/ 9][/*idx22=*/ 9][0] = v551;


//TerminatorOp

//} Unrolled body 9 of loop522.
//DEBUG: /*idx522=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop522.
//DEBUG: /*idx522=*/ 4'd10, expected 10
//printTimeOffset
reg tloop549delay[1:0] = '{default:0} ;
always@(*) tloop549delay[0] <= tloop549;
generate
genvar i553;

for(i553 = 1; i553<= 1; i553= i553 + 1) begin
always@(posedge clk) begin
tloop549delay[i553] <= tloop549delay[i553-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop552 = tloop549delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v554 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][154] = tloop549delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 10][/*idx22=*/ 9][0] = tloop549delay[0];
assign v1_wr_data_valid[/*idx522=*/ 10][/*idx22=*/ 9][0] = tloop549delay[0];
assign v1_wr_data_input[/*idx522=*/ 10][/*idx22=*/ 9][0] = v554;


//TerminatorOp

//} Unrolled body 10 of loop522.
//DEBUG: /*idx522=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop522.
//DEBUG: /*idx522=*/ 4'd11, expected 11
//printTimeOffset
reg tloop552delay[1:0] = '{default:0} ;
always@(*) tloop552delay[0] <= tloop552;
generate
genvar i556;

for(i556 = 1; i556<= 1; i556= i556 + 1) begin
always@(posedge clk) begin
tloop552delay[i556] <= tloop552delay[i556-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop555 = tloop552delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v557 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][155] = tloop552delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 11][/*idx22=*/ 9][0] = tloop552delay[0];
assign v1_wr_data_valid[/*idx522=*/ 11][/*idx22=*/ 9][0] = tloop552delay[0];
assign v1_wr_data_input[/*idx522=*/ 11][/*idx22=*/ 9][0] = v557;


//TerminatorOp

//} Unrolled body 11 of loop522.
//DEBUG: /*idx522=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop522.
//DEBUG: /*idx522=*/ 4'd12, expected 12
//printTimeOffset
reg tloop555delay[1:0] = '{default:0} ;
always@(*) tloop555delay[0] <= tloop555;
generate
genvar i559;

for(i559 = 1; i559<= 1; i559= i559 + 1) begin
always@(posedge clk) begin
tloop555delay[i559] <= tloop555delay[i559-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop558 = tloop555delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v560 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][156] = tloop555delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 12][/*idx22=*/ 9][0] = tloop555delay[0];
assign v1_wr_data_valid[/*idx522=*/ 12][/*idx22=*/ 9][0] = tloop555delay[0];
assign v1_wr_data_input[/*idx522=*/ 12][/*idx22=*/ 9][0] = v560;


//TerminatorOp

//} Unrolled body 12 of loop522.
//DEBUG: /*idx522=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop522.
//DEBUG: /*idx522=*/ 4'd13, expected 13
//printTimeOffset
reg tloop558delay[1:0] = '{default:0} ;
always@(*) tloop558delay[0] <= tloop558;
generate
genvar i562;

for(i562 = 1; i562<= 1; i562= i562 + 1) begin
always@(posedge clk) begin
tloop558delay[i562] <= tloop558delay[i562-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop561 = tloop558delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v563 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][157] = tloop558delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 13][/*idx22=*/ 9][0] = tloop558delay[0];
assign v1_wr_data_valid[/*idx522=*/ 13][/*idx22=*/ 9][0] = tloop558delay[0];
assign v1_wr_data_input[/*idx522=*/ 13][/*idx22=*/ 9][0] = v563;


//TerminatorOp

//} Unrolled body 13 of loop522.
//DEBUG: /*idx522=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop522.
//DEBUG: /*idx522=*/ 4'd14, expected 14
//printTimeOffset
reg tloop561delay[1:0] = '{default:0} ;
always@(*) tloop561delay[0] <= tloop561;
generate
genvar i565;

for(i565 = 1; i565<= 1; i565= i565 + 1) begin
always@(posedge clk) begin
tloop561delay[i565] <= tloop561delay[i565-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop564 = tloop561delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v566 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][158] = tloop561delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 14][/*idx22=*/ 9][0] = tloop561delay[0];
assign v1_wr_data_valid[/*idx522=*/ 14][/*idx22=*/ 9][0] = tloop561delay[0];
assign v1_wr_data_input[/*idx522=*/ 14][/*idx22=*/ 9][0] = v566;


//TerminatorOp

//} Unrolled body 14 of loop522.
//DEBUG: /*idx522=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop522.
//DEBUG: /*idx522=*/ 4'd15, expected 15
//printTimeOffset
reg tloop564delay[1:0] = '{default:0} ;
always@(*) tloop564delay[0] <= tloop564;
generate
genvar i568;

for(i568 = 1; i568<= 1; i568= i568 + 1) begin
always@(posedge clk) begin
tloop564delay[i568] <= tloop564delay[i568-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop567 = tloop564delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v569 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][159] = tloop564delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx522=*/ 15][/*idx22=*/ 9][0] = tloop564delay[0];
assign v1_wr_data_valid[/*idx522=*/ 15][/*idx22=*/ 9][0] = tloop564delay[0];
assign v1_wr_data_input[/*idx522=*/ 15][/*idx22=*/ 9][0] = v569;


//TerminatorOp

//} Unrolled body 15 of loop522.
//DEBUG: /*idx522=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t570;
assign t570 = tloop567;
//printTimeOffset
reg t570delay[1:0] = '{default:0} ;
always@(*) t570delay[0] <= t570;
generate
genvar i571;

for(i571 = 1; i571<= 1; i571= i571 + 1) begin
always@(posedge clk) begin
t570delay[i571] <= t570delay[i571-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop517 = t570delay[1];

//TerminatorOp

//} Unrolled body 9 of loop22.
//DEBUG: /*idx22=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop22.
//DEBUG: /*idx22=*/ 4'd10, expected 10
//printTimeOffset
reg tloop517delay[0:0] = '{default:0} ;
always@(*) tloop517delay[0] <= tloop517;
generate
genvar i573;

for(i573 = 1; i573<= 0; i573= i573 + 1) begin
always@(posedge clk) begin
tloop517delay[i573] <= tloop517delay[i573-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg575[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg575[0] <= tloop517;
always@(posedge clk) shiftreg575[/*v5=*/ 1:1] <= shiftreg575[/*v5=*/ 0:0];
wire v574 = shiftreg575[/*v5=*/ 1];
//printTimeOffset
reg v574delay[1:0] = '{default:0} ;
always@(*) v574delay[0] <= v574;
generate
genvar i576;

for(i576 = 1; i576<= 1; i576= i576 + 1) begin
always@(posedge clk) begin
v574delay[i576] <= v574delay[i576-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop577.
//DEBUG: /*idx577=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop578 = v574delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v579 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][160] = v574delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 0][/*idx22=*/ 10][0] = v574delay[0];
assign v1_wr_data_valid[/*idx577=*/ 0][/*idx22=*/ 10][0] = v574delay[0];
assign v1_wr_data_input[/*idx577=*/ 0][/*idx22=*/ 10][0] = v579;


//TerminatorOp

//} Unrolled body 0 of loop577.
//DEBUG: /*idx577=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop577.
//DEBUG: /*idx577=*/ 1'd1, expected 1
//printTimeOffset
reg tloop578delay[1:0] = '{default:0} ;
always@(*) tloop578delay[0] <= tloop578;
generate
genvar i581;

for(i581 = 1; i581<= 1; i581= i581 + 1) begin
always@(posedge clk) begin
tloop578delay[i581] <= tloop578delay[i581-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop580 = tloop578delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v582 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][161] = tloop578delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 1][/*idx22=*/ 10][0] = tloop578delay[0];
assign v1_wr_data_valid[/*idx577=*/ 1][/*idx22=*/ 10][0] = tloop578delay[0];
assign v1_wr_data_input[/*idx577=*/ 1][/*idx22=*/ 10][0] = v582;


//TerminatorOp

//} Unrolled body 1 of loop577.
//DEBUG: /*idx577=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop577.
//DEBUG: /*idx577=*/ 2'd2, expected 2
//printTimeOffset
reg tloop580delay[1:0] = '{default:0} ;
always@(*) tloop580delay[0] <= tloop580;
generate
genvar i584;

for(i584 = 1; i584<= 1; i584= i584 + 1) begin
always@(posedge clk) begin
tloop580delay[i584] <= tloop580delay[i584-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop583 = tloop580delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v585 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][162] = tloop580delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 2][/*idx22=*/ 10][0] = tloop580delay[0];
assign v1_wr_data_valid[/*idx577=*/ 2][/*idx22=*/ 10][0] = tloop580delay[0];
assign v1_wr_data_input[/*idx577=*/ 2][/*idx22=*/ 10][0] = v585;


//TerminatorOp

//} Unrolled body 2 of loop577.
//DEBUG: /*idx577=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop577.
//DEBUG: /*idx577=*/ 2'd3, expected 3
//printTimeOffset
reg tloop583delay[1:0] = '{default:0} ;
always@(*) tloop583delay[0] <= tloop583;
generate
genvar i587;

for(i587 = 1; i587<= 1; i587= i587 + 1) begin
always@(posedge clk) begin
tloop583delay[i587] <= tloop583delay[i587-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop586 = tloop583delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v588 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][163] = tloop583delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 3][/*idx22=*/ 10][0] = tloop583delay[0];
assign v1_wr_data_valid[/*idx577=*/ 3][/*idx22=*/ 10][0] = tloop583delay[0];
assign v1_wr_data_input[/*idx577=*/ 3][/*idx22=*/ 10][0] = v588;


//TerminatorOp

//} Unrolled body 3 of loop577.
//DEBUG: /*idx577=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop577.
//DEBUG: /*idx577=*/ 3'd4, expected 4
//printTimeOffset
reg tloop586delay[1:0] = '{default:0} ;
always@(*) tloop586delay[0] <= tloop586;
generate
genvar i590;

for(i590 = 1; i590<= 1; i590= i590 + 1) begin
always@(posedge clk) begin
tloop586delay[i590] <= tloop586delay[i590-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop589 = tloop586delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v591 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][164] = tloop586delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 4][/*idx22=*/ 10][0] = tloop586delay[0];
assign v1_wr_data_valid[/*idx577=*/ 4][/*idx22=*/ 10][0] = tloop586delay[0];
assign v1_wr_data_input[/*idx577=*/ 4][/*idx22=*/ 10][0] = v591;


//TerminatorOp

//} Unrolled body 4 of loop577.
//DEBUG: /*idx577=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop577.
//DEBUG: /*idx577=*/ 3'd5, expected 5
//printTimeOffset
reg tloop589delay[1:0] = '{default:0} ;
always@(*) tloop589delay[0] <= tloop589;
generate
genvar i593;

for(i593 = 1; i593<= 1; i593= i593 + 1) begin
always@(posedge clk) begin
tloop589delay[i593] <= tloop589delay[i593-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop592 = tloop589delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v594 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][165] = tloop589delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 5][/*idx22=*/ 10][0] = tloop589delay[0];
assign v1_wr_data_valid[/*idx577=*/ 5][/*idx22=*/ 10][0] = tloop589delay[0];
assign v1_wr_data_input[/*idx577=*/ 5][/*idx22=*/ 10][0] = v594;


//TerminatorOp

//} Unrolled body 5 of loop577.
//DEBUG: /*idx577=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop577.
//DEBUG: /*idx577=*/ 3'd6, expected 6
//printTimeOffset
reg tloop592delay[1:0] = '{default:0} ;
always@(*) tloop592delay[0] <= tloop592;
generate
genvar i596;

for(i596 = 1; i596<= 1; i596= i596 + 1) begin
always@(posedge clk) begin
tloop592delay[i596] <= tloop592delay[i596-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop595 = tloop592delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v597 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][166] = tloop592delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 6][/*idx22=*/ 10][0] = tloop592delay[0];
assign v1_wr_data_valid[/*idx577=*/ 6][/*idx22=*/ 10][0] = tloop592delay[0];
assign v1_wr_data_input[/*idx577=*/ 6][/*idx22=*/ 10][0] = v597;


//TerminatorOp

//} Unrolled body 6 of loop577.
//DEBUG: /*idx577=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop577.
//DEBUG: /*idx577=*/ 3'd7, expected 7
//printTimeOffset
reg tloop595delay[1:0] = '{default:0} ;
always@(*) tloop595delay[0] <= tloop595;
generate
genvar i599;

for(i599 = 1; i599<= 1; i599= i599 + 1) begin
always@(posedge clk) begin
tloop595delay[i599] <= tloop595delay[i599-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop598 = tloop595delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v600 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][167] = tloop595delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 7][/*idx22=*/ 10][0] = tloop595delay[0];
assign v1_wr_data_valid[/*idx577=*/ 7][/*idx22=*/ 10][0] = tloop595delay[0];
assign v1_wr_data_input[/*idx577=*/ 7][/*idx22=*/ 10][0] = v600;


//TerminatorOp

//} Unrolled body 7 of loop577.
//DEBUG: /*idx577=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop577.
//DEBUG: /*idx577=*/ 4'd8, expected 8
//printTimeOffset
reg tloop598delay[1:0] = '{default:0} ;
always@(*) tloop598delay[0] <= tloop598;
generate
genvar i602;

for(i602 = 1; i602<= 1; i602= i602 + 1) begin
always@(posedge clk) begin
tloop598delay[i602] <= tloop598delay[i602-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop601 = tloop598delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v603 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][168] = tloop598delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 8][/*idx22=*/ 10][0] = tloop598delay[0];
assign v1_wr_data_valid[/*idx577=*/ 8][/*idx22=*/ 10][0] = tloop598delay[0];
assign v1_wr_data_input[/*idx577=*/ 8][/*idx22=*/ 10][0] = v603;


//TerminatorOp

//} Unrolled body 8 of loop577.
//DEBUG: /*idx577=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop577.
//DEBUG: /*idx577=*/ 4'd9, expected 9
//printTimeOffset
reg tloop601delay[1:0] = '{default:0} ;
always@(*) tloop601delay[0] <= tloop601;
generate
genvar i605;

for(i605 = 1; i605<= 1; i605= i605 + 1) begin
always@(posedge clk) begin
tloop601delay[i605] <= tloop601delay[i605-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop604 = tloop601delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v606 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][169] = tloop601delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 9][/*idx22=*/ 10][0] = tloop601delay[0];
assign v1_wr_data_valid[/*idx577=*/ 9][/*idx22=*/ 10][0] = tloop601delay[0];
assign v1_wr_data_input[/*idx577=*/ 9][/*idx22=*/ 10][0] = v606;


//TerminatorOp

//} Unrolled body 9 of loop577.
//DEBUG: /*idx577=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop577.
//DEBUG: /*idx577=*/ 4'd10, expected 10
//printTimeOffset
reg tloop604delay[1:0] = '{default:0} ;
always@(*) tloop604delay[0] <= tloop604;
generate
genvar i608;

for(i608 = 1; i608<= 1; i608= i608 + 1) begin
always@(posedge clk) begin
tloop604delay[i608] <= tloop604delay[i608-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop607 = tloop604delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v609 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][170] = tloop604delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 10][/*idx22=*/ 10][0] = tloop604delay[0];
assign v1_wr_data_valid[/*idx577=*/ 10][/*idx22=*/ 10][0] = tloop604delay[0];
assign v1_wr_data_input[/*idx577=*/ 10][/*idx22=*/ 10][0] = v609;


//TerminatorOp

//} Unrolled body 10 of loop577.
//DEBUG: /*idx577=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop577.
//DEBUG: /*idx577=*/ 4'd11, expected 11
//printTimeOffset
reg tloop607delay[1:0] = '{default:0} ;
always@(*) tloop607delay[0] <= tloop607;
generate
genvar i611;

for(i611 = 1; i611<= 1; i611= i611 + 1) begin
always@(posedge clk) begin
tloop607delay[i611] <= tloop607delay[i611-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop610 = tloop607delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v612 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][171] = tloop607delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 11][/*idx22=*/ 10][0] = tloop607delay[0];
assign v1_wr_data_valid[/*idx577=*/ 11][/*idx22=*/ 10][0] = tloop607delay[0];
assign v1_wr_data_input[/*idx577=*/ 11][/*idx22=*/ 10][0] = v612;


//TerminatorOp

//} Unrolled body 11 of loop577.
//DEBUG: /*idx577=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop577.
//DEBUG: /*idx577=*/ 4'd12, expected 12
//printTimeOffset
reg tloop610delay[1:0] = '{default:0} ;
always@(*) tloop610delay[0] <= tloop610;
generate
genvar i614;

for(i614 = 1; i614<= 1; i614= i614 + 1) begin
always@(posedge clk) begin
tloop610delay[i614] <= tloop610delay[i614-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop613 = tloop610delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v615 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][172] = tloop610delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 12][/*idx22=*/ 10][0] = tloop610delay[0];
assign v1_wr_data_valid[/*idx577=*/ 12][/*idx22=*/ 10][0] = tloop610delay[0];
assign v1_wr_data_input[/*idx577=*/ 12][/*idx22=*/ 10][0] = v615;


//TerminatorOp

//} Unrolled body 12 of loop577.
//DEBUG: /*idx577=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop577.
//DEBUG: /*idx577=*/ 4'd13, expected 13
//printTimeOffset
reg tloop613delay[1:0] = '{default:0} ;
always@(*) tloop613delay[0] <= tloop613;
generate
genvar i617;

for(i617 = 1; i617<= 1; i617= i617 + 1) begin
always@(posedge clk) begin
tloop613delay[i617] <= tloop613delay[i617-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop616 = tloop613delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v618 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][173] = tloop613delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 13][/*idx22=*/ 10][0] = tloop613delay[0];
assign v1_wr_data_valid[/*idx577=*/ 13][/*idx22=*/ 10][0] = tloop613delay[0];
assign v1_wr_data_input[/*idx577=*/ 13][/*idx22=*/ 10][0] = v618;


//TerminatorOp

//} Unrolled body 13 of loop577.
//DEBUG: /*idx577=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop577.
//DEBUG: /*idx577=*/ 4'd14, expected 14
//printTimeOffset
reg tloop616delay[1:0] = '{default:0} ;
always@(*) tloop616delay[0] <= tloop616;
generate
genvar i620;

for(i620 = 1; i620<= 1; i620= i620 + 1) begin
always@(posedge clk) begin
tloop616delay[i620] <= tloop616delay[i620-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop619 = tloop616delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v621 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][174] = tloop616delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 14][/*idx22=*/ 10][0] = tloop616delay[0];
assign v1_wr_data_valid[/*idx577=*/ 14][/*idx22=*/ 10][0] = tloop616delay[0];
assign v1_wr_data_input[/*idx577=*/ 14][/*idx22=*/ 10][0] = v621;


//TerminatorOp

//} Unrolled body 14 of loop577.
//DEBUG: /*idx577=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop577.
//DEBUG: /*idx577=*/ 4'd15, expected 15
//printTimeOffset
reg tloop619delay[1:0] = '{default:0} ;
always@(*) tloop619delay[0] <= tloop619;
generate
genvar i623;

for(i623 = 1; i623<= 1; i623= i623 + 1) begin
always@(posedge clk) begin
tloop619delay[i623] <= tloop619delay[i623-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop622 = tloop619delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v624 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][175] = tloop619delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx577=*/ 15][/*idx22=*/ 10][0] = tloop619delay[0];
assign v1_wr_data_valid[/*idx577=*/ 15][/*idx22=*/ 10][0] = tloop619delay[0];
assign v1_wr_data_input[/*idx577=*/ 15][/*idx22=*/ 10][0] = v624;


//TerminatorOp

//} Unrolled body 15 of loop577.
//DEBUG: /*idx577=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t625;
assign t625 = tloop622;
//printTimeOffset
reg t625delay[1:0] = '{default:0} ;
always@(*) t625delay[0] <= t625;
generate
genvar i626;

for(i626 = 1; i626<= 1; i626= i626 + 1) begin
always@(posedge clk) begin
t625delay[i626] <= t625delay[i626-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop572 = t625delay[1];

//TerminatorOp

//} Unrolled body 10 of loop22.
//DEBUG: /*idx22=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop22.
//DEBUG: /*idx22=*/ 4'd11, expected 11
//printTimeOffset
reg tloop572delay[0:0] = '{default:0} ;
always@(*) tloop572delay[0] <= tloop572;
generate
genvar i628;

for(i628 = 1; i628<= 0; i628= i628 + 1) begin
always@(posedge clk) begin
tloop572delay[i628] <= tloop572delay[i628-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg630[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg630[0] <= tloop572;
always@(posedge clk) shiftreg630[/*v5=*/ 1:1] <= shiftreg630[/*v5=*/ 0:0];
wire v629 = shiftreg630[/*v5=*/ 1];
//printTimeOffset
reg v629delay[1:0] = '{default:0} ;
always@(*) v629delay[0] <= v629;
generate
genvar i631;

for(i631 = 1; i631<= 1; i631= i631 + 1) begin
always@(posedge clk) begin
v629delay[i631] <= v629delay[i631-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop632.
//DEBUG: /*idx632=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop633 = v629delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v634 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][176] = v629delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 0][/*idx22=*/ 11][0] = v629delay[0];
assign v1_wr_data_valid[/*idx632=*/ 0][/*idx22=*/ 11][0] = v629delay[0];
assign v1_wr_data_input[/*idx632=*/ 0][/*idx22=*/ 11][0] = v634;


//TerminatorOp

//} Unrolled body 0 of loop632.
//DEBUG: /*idx632=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop632.
//DEBUG: /*idx632=*/ 1'd1, expected 1
//printTimeOffset
reg tloop633delay[1:0] = '{default:0} ;
always@(*) tloop633delay[0] <= tloop633;
generate
genvar i636;

for(i636 = 1; i636<= 1; i636= i636 + 1) begin
always@(posedge clk) begin
tloop633delay[i636] <= tloop633delay[i636-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop635 = tloop633delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v637 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][177] = tloop633delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 1][/*idx22=*/ 11][0] = tloop633delay[0];
assign v1_wr_data_valid[/*idx632=*/ 1][/*idx22=*/ 11][0] = tloop633delay[0];
assign v1_wr_data_input[/*idx632=*/ 1][/*idx22=*/ 11][0] = v637;


//TerminatorOp

//} Unrolled body 1 of loop632.
//DEBUG: /*idx632=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop632.
//DEBUG: /*idx632=*/ 2'd2, expected 2
//printTimeOffset
reg tloop635delay[1:0] = '{default:0} ;
always@(*) tloop635delay[0] <= tloop635;
generate
genvar i639;

for(i639 = 1; i639<= 1; i639= i639 + 1) begin
always@(posedge clk) begin
tloop635delay[i639] <= tloop635delay[i639-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop638 = tloop635delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v640 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][178] = tloop635delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 2][/*idx22=*/ 11][0] = tloop635delay[0];
assign v1_wr_data_valid[/*idx632=*/ 2][/*idx22=*/ 11][0] = tloop635delay[0];
assign v1_wr_data_input[/*idx632=*/ 2][/*idx22=*/ 11][0] = v640;


//TerminatorOp

//} Unrolled body 2 of loop632.
//DEBUG: /*idx632=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop632.
//DEBUG: /*idx632=*/ 2'd3, expected 3
//printTimeOffset
reg tloop638delay[1:0] = '{default:0} ;
always@(*) tloop638delay[0] <= tloop638;
generate
genvar i642;

for(i642 = 1; i642<= 1; i642= i642 + 1) begin
always@(posedge clk) begin
tloop638delay[i642] <= tloop638delay[i642-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop641 = tloop638delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v643 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][179] = tloop638delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 3][/*idx22=*/ 11][0] = tloop638delay[0];
assign v1_wr_data_valid[/*idx632=*/ 3][/*idx22=*/ 11][0] = tloop638delay[0];
assign v1_wr_data_input[/*idx632=*/ 3][/*idx22=*/ 11][0] = v643;


//TerminatorOp

//} Unrolled body 3 of loop632.
//DEBUG: /*idx632=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop632.
//DEBUG: /*idx632=*/ 3'd4, expected 4
//printTimeOffset
reg tloop641delay[1:0] = '{default:0} ;
always@(*) tloop641delay[0] <= tloop641;
generate
genvar i645;

for(i645 = 1; i645<= 1; i645= i645 + 1) begin
always@(posedge clk) begin
tloop641delay[i645] <= tloop641delay[i645-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop644 = tloop641delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v646 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][180] = tloop641delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 4][/*idx22=*/ 11][0] = tloop641delay[0];
assign v1_wr_data_valid[/*idx632=*/ 4][/*idx22=*/ 11][0] = tloop641delay[0];
assign v1_wr_data_input[/*idx632=*/ 4][/*idx22=*/ 11][0] = v646;


//TerminatorOp

//} Unrolled body 4 of loop632.
//DEBUG: /*idx632=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop632.
//DEBUG: /*idx632=*/ 3'd5, expected 5
//printTimeOffset
reg tloop644delay[1:0] = '{default:0} ;
always@(*) tloop644delay[0] <= tloop644;
generate
genvar i648;

for(i648 = 1; i648<= 1; i648= i648 + 1) begin
always@(posedge clk) begin
tloop644delay[i648] <= tloop644delay[i648-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop647 = tloop644delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v649 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][181] = tloop644delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 5][/*idx22=*/ 11][0] = tloop644delay[0];
assign v1_wr_data_valid[/*idx632=*/ 5][/*idx22=*/ 11][0] = tloop644delay[0];
assign v1_wr_data_input[/*idx632=*/ 5][/*idx22=*/ 11][0] = v649;


//TerminatorOp

//} Unrolled body 5 of loop632.
//DEBUG: /*idx632=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop632.
//DEBUG: /*idx632=*/ 3'd6, expected 6
//printTimeOffset
reg tloop647delay[1:0] = '{default:0} ;
always@(*) tloop647delay[0] <= tloop647;
generate
genvar i651;

for(i651 = 1; i651<= 1; i651= i651 + 1) begin
always@(posedge clk) begin
tloop647delay[i651] <= tloop647delay[i651-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop650 = tloop647delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v652 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][182] = tloop647delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 6][/*idx22=*/ 11][0] = tloop647delay[0];
assign v1_wr_data_valid[/*idx632=*/ 6][/*idx22=*/ 11][0] = tloop647delay[0];
assign v1_wr_data_input[/*idx632=*/ 6][/*idx22=*/ 11][0] = v652;


//TerminatorOp

//} Unrolled body 6 of loop632.
//DEBUG: /*idx632=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop632.
//DEBUG: /*idx632=*/ 3'd7, expected 7
//printTimeOffset
reg tloop650delay[1:0] = '{default:0} ;
always@(*) tloop650delay[0] <= tloop650;
generate
genvar i654;

for(i654 = 1; i654<= 1; i654= i654 + 1) begin
always@(posedge clk) begin
tloop650delay[i654] <= tloop650delay[i654-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop653 = tloop650delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v655 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][183] = tloop650delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 7][/*idx22=*/ 11][0] = tloop650delay[0];
assign v1_wr_data_valid[/*idx632=*/ 7][/*idx22=*/ 11][0] = tloop650delay[0];
assign v1_wr_data_input[/*idx632=*/ 7][/*idx22=*/ 11][0] = v655;


//TerminatorOp

//} Unrolled body 7 of loop632.
//DEBUG: /*idx632=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop632.
//DEBUG: /*idx632=*/ 4'd8, expected 8
//printTimeOffset
reg tloop653delay[1:0] = '{default:0} ;
always@(*) tloop653delay[0] <= tloop653;
generate
genvar i657;

for(i657 = 1; i657<= 1; i657= i657 + 1) begin
always@(posedge clk) begin
tloop653delay[i657] <= tloop653delay[i657-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop656 = tloop653delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v658 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][184] = tloop653delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 8][/*idx22=*/ 11][0] = tloop653delay[0];
assign v1_wr_data_valid[/*idx632=*/ 8][/*idx22=*/ 11][0] = tloop653delay[0];
assign v1_wr_data_input[/*idx632=*/ 8][/*idx22=*/ 11][0] = v658;


//TerminatorOp

//} Unrolled body 8 of loop632.
//DEBUG: /*idx632=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop632.
//DEBUG: /*idx632=*/ 4'd9, expected 9
//printTimeOffset
reg tloop656delay[1:0] = '{default:0} ;
always@(*) tloop656delay[0] <= tloop656;
generate
genvar i660;

for(i660 = 1; i660<= 1; i660= i660 + 1) begin
always@(posedge clk) begin
tloop656delay[i660] <= tloop656delay[i660-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop659 = tloop656delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v661 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][185] = tloop656delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 9][/*idx22=*/ 11][0] = tloop656delay[0];
assign v1_wr_data_valid[/*idx632=*/ 9][/*idx22=*/ 11][0] = tloop656delay[0];
assign v1_wr_data_input[/*idx632=*/ 9][/*idx22=*/ 11][0] = v661;


//TerminatorOp

//} Unrolled body 9 of loop632.
//DEBUG: /*idx632=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop632.
//DEBUG: /*idx632=*/ 4'd10, expected 10
//printTimeOffset
reg tloop659delay[1:0] = '{default:0} ;
always@(*) tloop659delay[0] <= tloop659;
generate
genvar i663;

for(i663 = 1; i663<= 1; i663= i663 + 1) begin
always@(posedge clk) begin
tloop659delay[i663] <= tloop659delay[i663-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop662 = tloop659delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v664 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][186] = tloop659delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 10][/*idx22=*/ 11][0] = tloop659delay[0];
assign v1_wr_data_valid[/*idx632=*/ 10][/*idx22=*/ 11][0] = tloop659delay[0];
assign v1_wr_data_input[/*idx632=*/ 10][/*idx22=*/ 11][0] = v664;


//TerminatorOp

//} Unrolled body 10 of loop632.
//DEBUG: /*idx632=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop632.
//DEBUG: /*idx632=*/ 4'd11, expected 11
//printTimeOffset
reg tloop662delay[1:0] = '{default:0} ;
always@(*) tloop662delay[0] <= tloop662;
generate
genvar i666;

for(i666 = 1; i666<= 1; i666= i666 + 1) begin
always@(posedge clk) begin
tloop662delay[i666] <= tloop662delay[i666-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop665 = tloop662delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v667 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][187] = tloop662delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 11][/*idx22=*/ 11][0] = tloop662delay[0];
assign v1_wr_data_valid[/*idx632=*/ 11][/*idx22=*/ 11][0] = tloop662delay[0];
assign v1_wr_data_input[/*idx632=*/ 11][/*idx22=*/ 11][0] = v667;


//TerminatorOp

//} Unrolled body 11 of loop632.
//DEBUG: /*idx632=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop632.
//DEBUG: /*idx632=*/ 4'd12, expected 12
//printTimeOffset
reg tloop665delay[1:0] = '{default:0} ;
always@(*) tloop665delay[0] <= tloop665;
generate
genvar i669;

for(i669 = 1; i669<= 1; i669= i669 + 1) begin
always@(posedge clk) begin
tloop665delay[i669] <= tloop665delay[i669-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop668 = tloop665delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v670 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][188] = tloop665delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 12][/*idx22=*/ 11][0] = tloop665delay[0];
assign v1_wr_data_valid[/*idx632=*/ 12][/*idx22=*/ 11][0] = tloop665delay[0];
assign v1_wr_data_input[/*idx632=*/ 12][/*idx22=*/ 11][0] = v670;


//TerminatorOp

//} Unrolled body 12 of loop632.
//DEBUG: /*idx632=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop632.
//DEBUG: /*idx632=*/ 4'd13, expected 13
//printTimeOffset
reg tloop668delay[1:0] = '{default:0} ;
always@(*) tloop668delay[0] <= tloop668;
generate
genvar i672;

for(i672 = 1; i672<= 1; i672= i672 + 1) begin
always@(posedge clk) begin
tloop668delay[i672] <= tloop668delay[i672-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop671 = tloop668delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v673 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][189] = tloop668delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 13][/*idx22=*/ 11][0] = tloop668delay[0];
assign v1_wr_data_valid[/*idx632=*/ 13][/*idx22=*/ 11][0] = tloop668delay[0];
assign v1_wr_data_input[/*idx632=*/ 13][/*idx22=*/ 11][0] = v673;


//TerminatorOp

//} Unrolled body 13 of loop632.
//DEBUG: /*idx632=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop632.
//DEBUG: /*idx632=*/ 4'd14, expected 14
//printTimeOffset
reg tloop671delay[1:0] = '{default:0} ;
always@(*) tloop671delay[0] <= tloop671;
generate
genvar i675;

for(i675 = 1; i675<= 1; i675= i675 + 1) begin
always@(posedge clk) begin
tloop671delay[i675] <= tloop671delay[i675-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop674 = tloop671delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v676 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][190] = tloop671delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 14][/*idx22=*/ 11][0] = tloop671delay[0];
assign v1_wr_data_valid[/*idx632=*/ 14][/*idx22=*/ 11][0] = tloop671delay[0];
assign v1_wr_data_input[/*idx632=*/ 14][/*idx22=*/ 11][0] = v676;


//TerminatorOp

//} Unrolled body 14 of loop632.
//DEBUG: /*idx632=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop632.
//DEBUG: /*idx632=*/ 4'd15, expected 15
//printTimeOffset
reg tloop674delay[1:0] = '{default:0} ;
always@(*) tloop674delay[0] <= tloop674;
generate
genvar i678;

for(i678 = 1; i678<= 1; i678= i678 + 1) begin
always@(posedge clk) begin
tloop674delay[i678] <= tloop674delay[i678-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop677 = tloop674delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v679 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][191] = tloop674delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx632=*/ 15][/*idx22=*/ 11][0] = tloop674delay[0];
assign v1_wr_data_valid[/*idx632=*/ 15][/*idx22=*/ 11][0] = tloop674delay[0];
assign v1_wr_data_input[/*idx632=*/ 15][/*idx22=*/ 11][0] = v679;


//TerminatorOp

//} Unrolled body 15 of loop632.
//DEBUG: /*idx632=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t680;
assign t680 = tloop677;
//printTimeOffset
reg t680delay[1:0] = '{default:0} ;
always@(*) t680delay[0] <= t680;
generate
genvar i681;

for(i681 = 1; i681<= 1; i681= i681 + 1) begin
always@(posedge clk) begin
t680delay[i681] <= t680delay[i681-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop627 = t680delay[1];

//TerminatorOp

//} Unrolled body 11 of loop22.
//DEBUG: /*idx22=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop22.
//DEBUG: /*idx22=*/ 4'd12, expected 12
//printTimeOffset
reg tloop627delay[0:0] = '{default:0} ;
always@(*) tloop627delay[0] <= tloop627;
generate
genvar i683;

for(i683 = 1; i683<= 0; i683= i683 + 1) begin
always@(posedge clk) begin
tloop627delay[i683] <= tloop627delay[i683-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg685[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg685[0] <= tloop627;
always@(posedge clk) shiftreg685[/*v5=*/ 1:1] <= shiftreg685[/*v5=*/ 0:0];
wire v684 = shiftreg685[/*v5=*/ 1];
//printTimeOffset
reg v684delay[1:0] = '{default:0} ;
always@(*) v684delay[0] <= v684;
generate
genvar i686;

for(i686 = 1; i686<= 1; i686= i686 + 1) begin
always@(posedge clk) begin
v684delay[i686] <= v684delay[i686-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop687.
//DEBUG: /*idx687=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop688 = v684delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v689 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][192] = v684delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 0][/*idx22=*/ 12][0] = v684delay[0];
assign v1_wr_data_valid[/*idx687=*/ 0][/*idx22=*/ 12][0] = v684delay[0];
assign v1_wr_data_input[/*idx687=*/ 0][/*idx22=*/ 12][0] = v689;


//TerminatorOp

//} Unrolled body 0 of loop687.
//DEBUG: /*idx687=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop687.
//DEBUG: /*idx687=*/ 1'd1, expected 1
//printTimeOffset
reg tloop688delay[1:0] = '{default:0} ;
always@(*) tloop688delay[0] <= tloop688;
generate
genvar i691;

for(i691 = 1; i691<= 1; i691= i691 + 1) begin
always@(posedge clk) begin
tloop688delay[i691] <= tloop688delay[i691-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop690 = tloop688delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v692 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][193] = tloop688delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 1][/*idx22=*/ 12][0] = tloop688delay[0];
assign v1_wr_data_valid[/*idx687=*/ 1][/*idx22=*/ 12][0] = tloop688delay[0];
assign v1_wr_data_input[/*idx687=*/ 1][/*idx22=*/ 12][0] = v692;


//TerminatorOp

//} Unrolled body 1 of loop687.
//DEBUG: /*idx687=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop687.
//DEBUG: /*idx687=*/ 2'd2, expected 2
//printTimeOffset
reg tloop690delay[1:0] = '{default:0} ;
always@(*) tloop690delay[0] <= tloop690;
generate
genvar i694;

for(i694 = 1; i694<= 1; i694= i694 + 1) begin
always@(posedge clk) begin
tloop690delay[i694] <= tloop690delay[i694-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop693 = tloop690delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v695 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][194] = tloop690delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 2][/*idx22=*/ 12][0] = tloop690delay[0];
assign v1_wr_data_valid[/*idx687=*/ 2][/*idx22=*/ 12][0] = tloop690delay[0];
assign v1_wr_data_input[/*idx687=*/ 2][/*idx22=*/ 12][0] = v695;


//TerminatorOp

//} Unrolled body 2 of loop687.
//DEBUG: /*idx687=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop687.
//DEBUG: /*idx687=*/ 2'd3, expected 3
//printTimeOffset
reg tloop693delay[1:0] = '{default:0} ;
always@(*) tloop693delay[0] <= tloop693;
generate
genvar i697;

for(i697 = 1; i697<= 1; i697= i697 + 1) begin
always@(posedge clk) begin
tloop693delay[i697] <= tloop693delay[i697-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop696 = tloop693delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v698 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][195] = tloop693delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 3][/*idx22=*/ 12][0] = tloop693delay[0];
assign v1_wr_data_valid[/*idx687=*/ 3][/*idx22=*/ 12][0] = tloop693delay[0];
assign v1_wr_data_input[/*idx687=*/ 3][/*idx22=*/ 12][0] = v698;


//TerminatorOp

//} Unrolled body 3 of loop687.
//DEBUG: /*idx687=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop687.
//DEBUG: /*idx687=*/ 3'd4, expected 4
//printTimeOffset
reg tloop696delay[1:0] = '{default:0} ;
always@(*) tloop696delay[0] <= tloop696;
generate
genvar i700;

for(i700 = 1; i700<= 1; i700= i700 + 1) begin
always@(posedge clk) begin
tloop696delay[i700] <= tloop696delay[i700-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop699 = tloop696delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v701 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][196] = tloop696delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 4][/*idx22=*/ 12][0] = tloop696delay[0];
assign v1_wr_data_valid[/*idx687=*/ 4][/*idx22=*/ 12][0] = tloop696delay[0];
assign v1_wr_data_input[/*idx687=*/ 4][/*idx22=*/ 12][0] = v701;


//TerminatorOp

//} Unrolled body 4 of loop687.
//DEBUG: /*idx687=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop687.
//DEBUG: /*idx687=*/ 3'd5, expected 5
//printTimeOffset
reg tloop699delay[1:0] = '{default:0} ;
always@(*) tloop699delay[0] <= tloop699;
generate
genvar i703;

for(i703 = 1; i703<= 1; i703= i703 + 1) begin
always@(posedge clk) begin
tloop699delay[i703] <= tloop699delay[i703-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop702 = tloop699delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v704 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][197] = tloop699delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 5][/*idx22=*/ 12][0] = tloop699delay[0];
assign v1_wr_data_valid[/*idx687=*/ 5][/*idx22=*/ 12][0] = tloop699delay[0];
assign v1_wr_data_input[/*idx687=*/ 5][/*idx22=*/ 12][0] = v704;


//TerminatorOp

//} Unrolled body 5 of loop687.
//DEBUG: /*idx687=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop687.
//DEBUG: /*idx687=*/ 3'd6, expected 6
//printTimeOffset
reg tloop702delay[1:0] = '{default:0} ;
always@(*) tloop702delay[0] <= tloop702;
generate
genvar i706;

for(i706 = 1; i706<= 1; i706= i706 + 1) begin
always@(posedge clk) begin
tloop702delay[i706] <= tloop702delay[i706-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop705 = tloop702delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v707 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][198] = tloop702delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 6][/*idx22=*/ 12][0] = tloop702delay[0];
assign v1_wr_data_valid[/*idx687=*/ 6][/*idx22=*/ 12][0] = tloop702delay[0];
assign v1_wr_data_input[/*idx687=*/ 6][/*idx22=*/ 12][0] = v707;


//TerminatorOp

//} Unrolled body 6 of loop687.
//DEBUG: /*idx687=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop687.
//DEBUG: /*idx687=*/ 3'd7, expected 7
//printTimeOffset
reg tloop705delay[1:0] = '{default:0} ;
always@(*) tloop705delay[0] <= tloop705;
generate
genvar i709;

for(i709 = 1; i709<= 1; i709= i709 + 1) begin
always@(posedge clk) begin
tloop705delay[i709] <= tloop705delay[i709-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop708 = tloop705delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v710 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][199] = tloop705delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 7][/*idx22=*/ 12][0] = tloop705delay[0];
assign v1_wr_data_valid[/*idx687=*/ 7][/*idx22=*/ 12][0] = tloop705delay[0];
assign v1_wr_data_input[/*idx687=*/ 7][/*idx22=*/ 12][0] = v710;


//TerminatorOp

//} Unrolled body 7 of loop687.
//DEBUG: /*idx687=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop687.
//DEBUG: /*idx687=*/ 4'd8, expected 8
//printTimeOffset
reg tloop708delay[1:0] = '{default:0} ;
always@(*) tloop708delay[0] <= tloop708;
generate
genvar i712;

for(i712 = 1; i712<= 1; i712= i712 + 1) begin
always@(posedge clk) begin
tloop708delay[i712] <= tloop708delay[i712-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop711 = tloop708delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v713 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][200] = tloop708delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 8][/*idx22=*/ 12][0] = tloop708delay[0];
assign v1_wr_data_valid[/*idx687=*/ 8][/*idx22=*/ 12][0] = tloop708delay[0];
assign v1_wr_data_input[/*idx687=*/ 8][/*idx22=*/ 12][0] = v713;


//TerminatorOp

//} Unrolled body 8 of loop687.
//DEBUG: /*idx687=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop687.
//DEBUG: /*idx687=*/ 4'd9, expected 9
//printTimeOffset
reg tloop711delay[1:0] = '{default:0} ;
always@(*) tloop711delay[0] <= tloop711;
generate
genvar i715;

for(i715 = 1; i715<= 1; i715= i715 + 1) begin
always@(posedge clk) begin
tloop711delay[i715] <= tloop711delay[i715-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop714 = tloop711delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v716 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][201] = tloop711delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 9][/*idx22=*/ 12][0] = tloop711delay[0];
assign v1_wr_data_valid[/*idx687=*/ 9][/*idx22=*/ 12][0] = tloop711delay[0];
assign v1_wr_data_input[/*idx687=*/ 9][/*idx22=*/ 12][0] = v716;


//TerminatorOp

//} Unrolled body 9 of loop687.
//DEBUG: /*idx687=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop687.
//DEBUG: /*idx687=*/ 4'd10, expected 10
//printTimeOffset
reg tloop714delay[1:0] = '{default:0} ;
always@(*) tloop714delay[0] <= tloop714;
generate
genvar i718;

for(i718 = 1; i718<= 1; i718= i718 + 1) begin
always@(posedge clk) begin
tloop714delay[i718] <= tloop714delay[i718-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop717 = tloop714delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v719 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][202] = tloop714delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 10][/*idx22=*/ 12][0] = tloop714delay[0];
assign v1_wr_data_valid[/*idx687=*/ 10][/*idx22=*/ 12][0] = tloop714delay[0];
assign v1_wr_data_input[/*idx687=*/ 10][/*idx22=*/ 12][0] = v719;


//TerminatorOp

//} Unrolled body 10 of loop687.
//DEBUG: /*idx687=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop687.
//DEBUG: /*idx687=*/ 4'd11, expected 11
//printTimeOffset
reg tloop717delay[1:0] = '{default:0} ;
always@(*) tloop717delay[0] <= tloop717;
generate
genvar i721;

for(i721 = 1; i721<= 1; i721= i721 + 1) begin
always@(posedge clk) begin
tloop717delay[i721] <= tloop717delay[i721-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop720 = tloop717delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v722 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][203] = tloop717delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 11][/*idx22=*/ 12][0] = tloop717delay[0];
assign v1_wr_data_valid[/*idx687=*/ 11][/*idx22=*/ 12][0] = tloop717delay[0];
assign v1_wr_data_input[/*idx687=*/ 11][/*idx22=*/ 12][0] = v722;


//TerminatorOp

//} Unrolled body 11 of loop687.
//DEBUG: /*idx687=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop687.
//DEBUG: /*idx687=*/ 4'd12, expected 12
//printTimeOffset
reg tloop720delay[1:0] = '{default:0} ;
always@(*) tloop720delay[0] <= tloop720;
generate
genvar i724;

for(i724 = 1; i724<= 1; i724= i724 + 1) begin
always@(posedge clk) begin
tloop720delay[i724] <= tloop720delay[i724-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop723 = tloop720delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v725 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][204] = tloop720delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 12][/*idx22=*/ 12][0] = tloop720delay[0];
assign v1_wr_data_valid[/*idx687=*/ 12][/*idx22=*/ 12][0] = tloop720delay[0];
assign v1_wr_data_input[/*idx687=*/ 12][/*idx22=*/ 12][0] = v725;


//TerminatorOp

//} Unrolled body 12 of loop687.
//DEBUG: /*idx687=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop687.
//DEBUG: /*idx687=*/ 4'd13, expected 13
//printTimeOffset
reg tloop723delay[1:0] = '{default:0} ;
always@(*) tloop723delay[0] <= tloop723;
generate
genvar i727;

for(i727 = 1; i727<= 1; i727= i727 + 1) begin
always@(posedge clk) begin
tloop723delay[i727] <= tloop723delay[i727-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop726 = tloop723delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v728 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][205] = tloop723delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 13][/*idx22=*/ 12][0] = tloop723delay[0];
assign v1_wr_data_valid[/*idx687=*/ 13][/*idx22=*/ 12][0] = tloop723delay[0];
assign v1_wr_data_input[/*idx687=*/ 13][/*idx22=*/ 12][0] = v728;


//TerminatorOp

//} Unrolled body 13 of loop687.
//DEBUG: /*idx687=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop687.
//DEBUG: /*idx687=*/ 4'd14, expected 14
//printTimeOffset
reg tloop726delay[1:0] = '{default:0} ;
always@(*) tloop726delay[0] <= tloop726;
generate
genvar i730;

for(i730 = 1; i730<= 1; i730= i730 + 1) begin
always@(posedge clk) begin
tloop726delay[i730] <= tloop726delay[i730-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop729 = tloop726delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v731 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][206] = tloop726delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 14][/*idx22=*/ 12][0] = tloop726delay[0];
assign v1_wr_data_valid[/*idx687=*/ 14][/*idx22=*/ 12][0] = tloop726delay[0];
assign v1_wr_data_input[/*idx687=*/ 14][/*idx22=*/ 12][0] = v731;


//TerminatorOp

//} Unrolled body 14 of loop687.
//DEBUG: /*idx687=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop687.
//DEBUG: /*idx687=*/ 4'd15, expected 15
//printTimeOffset
reg tloop729delay[1:0] = '{default:0} ;
always@(*) tloop729delay[0] <= tloop729;
generate
genvar i733;

for(i733 = 1; i733<= 1; i733= i733 + 1) begin
always@(posedge clk) begin
tloop729delay[i733] <= tloop729delay[i733-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop732 = tloop729delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v734 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][207] = tloop729delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx687=*/ 15][/*idx22=*/ 12][0] = tloop729delay[0];
assign v1_wr_data_valid[/*idx687=*/ 15][/*idx22=*/ 12][0] = tloop729delay[0];
assign v1_wr_data_input[/*idx687=*/ 15][/*idx22=*/ 12][0] = v734;


//TerminatorOp

//} Unrolled body 15 of loop687.
//DEBUG: /*idx687=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t735;
assign t735 = tloop732;
//printTimeOffset
reg t735delay[1:0] = '{default:0} ;
always@(*) t735delay[0] <= t735;
generate
genvar i736;

for(i736 = 1; i736<= 1; i736= i736 + 1) begin
always@(posedge clk) begin
t735delay[i736] <= t735delay[i736-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop682 = t735delay[1];

//TerminatorOp

//} Unrolled body 12 of loop22.
//DEBUG: /*idx22=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop22.
//DEBUG: /*idx22=*/ 4'd13, expected 13
//printTimeOffset
reg tloop682delay[0:0] = '{default:0} ;
always@(*) tloop682delay[0] <= tloop682;
generate
genvar i738;

for(i738 = 1; i738<= 0; i738= i738 + 1) begin
always@(posedge clk) begin
tloop682delay[i738] <= tloop682delay[i738-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg740[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg740[0] <= tloop682;
always@(posedge clk) shiftreg740[/*v5=*/ 1:1] <= shiftreg740[/*v5=*/ 0:0];
wire v739 = shiftreg740[/*v5=*/ 1];
//printTimeOffset
reg v739delay[1:0] = '{default:0} ;
always@(*) v739delay[0] <= v739;
generate
genvar i741;

for(i741 = 1; i741<= 1; i741= i741 + 1) begin
always@(posedge clk) begin
v739delay[i741] <= v739delay[i741-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop742.
//DEBUG: /*idx742=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop743 = v739delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v744 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][208] = v739delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 0][/*idx22=*/ 13][0] = v739delay[0];
assign v1_wr_data_valid[/*idx742=*/ 0][/*idx22=*/ 13][0] = v739delay[0];
assign v1_wr_data_input[/*idx742=*/ 0][/*idx22=*/ 13][0] = v744;


//TerminatorOp

//} Unrolled body 0 of loop742.
//DEBUG: /*idx742=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop742.
//DEBUG: /*idx742=*/ 1'd1, expected 1
//printTimeOffset
reg tloop743delay[1:0] = '{default:0} ;
always@(*) tloop743delay[0] <= tloop743;
generate
genvar i746;

for(i746 = 1; i746<= 1; i746= i746 + 1) begin
always@(posedge clk) begin
tloop743delay[i746] <= tloop743delay[i746-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop745 = tloop743delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v747 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][209] = tloop743delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 1][/*idx22=*/ 13][0] = tloop743delay[0];
assign v1_wr_data_valid[/*idx742=*/ 1][/*idx22=*/ 13][0] = tloop743delay[0];
assign v1_wr_data_input[/*idx742=*/ 1][/*idx22=*/ 13][0] = v747;


//TerminatorOp

//} Unrolled body 1 of loop742.
//DEBUG: /*idx742=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop742.
//DEBUG: /*idx742=*/ 2'd2, expected 2
//printTimeOffset
reg tloop745delay[1:0] = '{default:0} ;
always@(*) tloop745delay[0] <= tloop745;
generate
genvar i749;

for(i749 = 1; i749<= 1; i749= i749 + 1) begin
always@(posedge clk) begin
tloop745delay[i749] <= tloop745delay[i749-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop748 = tloop745delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v750 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][210] = tloop745delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 2][/*idx22=*/ 13][0] = tloop745delay[0];
assign v1_wr_data_valid[/*idx742=*/ 2][/*idx22=*/ 13][0] = tloop745delay[0];
assign v1_wr_data_input[/*idx742=*/ 2][/*idx22=*/ 13][0] = v750;


//TerminatorOp

//} Unrolled body 2 of loop742.
//DEBUG: /*idx742=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop742.
//DEBUG: /*idx742=*/ 2'd3, expected 3
//printTimeOffset
reg tloop748delay[1:0] = '{default:0} ;
always@(*) tloop748delay[0] <= tloop748;
generate
genvar i752;

for(i752 = 1; i752<= 1; i752= i752 + 1) begin
always@(posedge clk) begin
tloop748delay[i752] <= tloop748delay[i752-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop751 = tloop748delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v753 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][211] = tloop748delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 3][/*idx22=*/ 13][0] = tloop748delay[0];
assign v1_wr_data_valid[/*idx742=*/ 3][/*idx22=*/ 13][0] = tloop748delay[0];
assign v1_wr_data_input[/*idx742=*/ 3][/*idx22=*/ 13][0] = v753;


//TerminatorOp

//} Unrolled body 3 of loop742.
//DEBUG: /*idx742=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop742.
//DEBUG: /*idx742=*/ 3'd4, expected 4
//printTimeOffset
reg tloop751delay[1:0] = '{default:0} ;
always@(*) tloop751delay[0] <= tloop751;
generate
genvar i755;

for(i755 = 1; i755<= 1; i755= i755 + 1) begin
always@(posedge clk) begin
tloop751delay[i755] <= tloop751delay[i755-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop754 = tloop751delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v756 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][212] = tloop751delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 4][/*idx22=*/ 13][0] = tloop751delay[0];
assign v1_wr_data_valid[/*idx742=*/ 4][/*idx22=*/ 13][0] = tloop751delay[0];
assign v1_wr_data_input[/*idx742=*/ 4][/*idx22=*/ 13][0] = v756;


//TerminatorOp

//} Unrolled body 4 of loop742.
//DEBUG: /*idx742=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop742.
//DEBUG: /*idx742=*/ 3'd5, expected 5
//printTimeOffset
reg tloop754delay[1:0] = '{default:0} ;
always@(*) tloop754delay[0] <= tloop754;
generate
genvar i758;

for(i758 = 1; i758<= 1; i758= i758 + 1) begin
always@(posedge clk) begin
tloop754delay[i758] <= tloop754delay[i758-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop757 = tloop754delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v759 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][213] = tloop754delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 5][/*idx22=*/ 13][0] = tloop754delay[0];
assign v1_wr_data_valid[/*idx742=*/ 5][/*idx22=*/ 13][0] = tloop754delay[0];
assign v1_wr_data_input[/*idx742=*/ 5][/*idx22=*/ 13][0] = v759;


//TerminatorOp

//} Unrolled body 5 of loop742.
//DEBUG: /*idx742=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop742.
//DEBUG: /*idx742=*/ 3'd6, expected 6
//printTimeOffset
reg tloop757delay[1:0] = '{default:0} ;
always@(*) tloop757delay[0] <= tloop757;
generate
genvar i761;

for(i761 = 1; i761<= 1; i761= i761 + 1) begin
always@(posedge clk) begin
tloop757delay[i761] <= tloop757delay[i761-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop760 = tloop757delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v762 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][214] = tloop757delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 6][/*idx22=*/ 13][0] = tloop757delay[0];
assign v1_wr_data_valid[/*idx742=*/ 6][/*idx22=*/ 13][0] = tloop757delay[0];
assign v1_wr_data_input[/*idx742=*/ 6][/*idx22=*/ 13][0] = v762;


//TerminatorOp

//} Unrolled body 6 of loop742.
//DEBUG: /*idx742=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop742.
//DEBUG: /*idx742=*/ 3'd7, expected 7
//printTimeOffset
reg tloop760delay[1:0] = '{default:0} ;
always@(*) tloop760delay[0] <= tloop760;
generate
genvar i764;

for(i764 = 1; i764<= 1; i764= i764 + 1) begin
always@(posedge clk) begin
tloop760delay[i764] <= tloop760delay[i764-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop763 = tloop760delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v765 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][215] = tloop760delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 7][/*idx22=*/ 13][0] = tloop760delay[0];
assign v1_wr_data_valid[/*idx742=*/ 7][/*idx22=*/ 13][0] = tloop760delay[0];
assign v1_wr_data_input[/*idx742=*/ 7][/*idx22=*/ 13][0] = v765;


//TerminatorOp

//} Unrolled body 7 of loop742.
//DEBUG: /*idx742=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop742.
//DEBUG: /*idx742=*/ 4'd8, expected 8
//printTimeOffset
reg tloop763delay[1:0] = '{default:0} ;
always@(*) tloop763delay[0] <= tloop763;
generate
genvar i767;

for(i767 = 1; i767<= 1; i767= i767 + 1) begin
always@(posedge clk) begin
tloop763delay[i767] <= tloop763delay[i767-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop766 = tloop763delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v768 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][216] = tloop763delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 8][/*idx22=*/ 13][0] = tloop763delay[0];
assign v1_wr_data_valid[/*idx742=*/ 8][/*idx22=*/ 13][0] = tloop763delay[0];
assign v1_wr_data_input[/*idx742=*/ 8][/*idx22=*/ 13][0] = v768;


//TerminatorOp

//} Unrolled body 8 of loop742.
//DEBUG: /*idx742=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop742.
//DEBUG: /*idx742=*/ 4'd9, expected 9
//printTimeOffset
reg tloop766delay[1:0] = '{default:0} ;
always@(*) tloop766delay[0] <= tloop766;
generate
genvar i770;

for(i770 = 1; i770<= 1; i770= i770 + 1) begin
always@(posedge clk) begin
tloop766delay[i770] <= tloop766delay[i770-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop769 = tloop766delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v771 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][217] = tloop766delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 9][/*idx22=*/ 13][0] = tloop766delay[0];
assign v1_wr_data_valid[/*idx742=*/ 9][/*idx22=*/ 13][0] = tloop766delay[0];
assign v1_wr_data_input[/*idx742=*/ 9][/*idx22=*/ 13][0] = v771;


//TerminatorOp

//} Unrolled body 9 of loop742.
//DEBUG: /*idx742=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop742.
//DEBUG: /*idx742=*/ 4'd10, expected 10
//printTimeOffset
reg tloop769delay[1:0] = '{default:0} ;
always@(*) tloop769delay[0] <= tloop769;
generate
genvar i773;

for(i773 = 1; i773<= 1; i773= i773 + 1) begin
always@(posedge clk) begin
tloop769delay[i773] <= tloop769delay[i773-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop772 = tloop769delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v774 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][218] = tloop769delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 10][/*idx22=*/ 13][0] = tloop769delay[0];
assign v1_wr_data_valid[/*idx742=*/ 10][/*idx22=*/ 13][0] = tloop769delay[0];
assign v1_wr_data_input[/*idx742=*/ 10][/*idx22=*/ 13][0] = v774;


//TerminatorOp

//} Unrolled body 10 of loop742.
//DEBUG: /*idx742=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop742.
//DEBUG: /*idx742=*/ 4'd11, expected 11
//printTimeOffset
reg tloop772delay[1:0] = '{default:0} ;
always@(*) tloop772delay[0] <= tloop772;
generate
genvar i776;

for(i776 = 1; i776<= 1; i776= i776 + 1) begin
always@(posedge clk) begin
tloop772delay[i776] <= tloop772delay[i776-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop775 = tloop772delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v777 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][219] = tloop772delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 11][/*idx22=*/ 13][0] = tloop772delay[0];
assign v1_wr_data_valid[/*idx742=*/ 11][/*idx22=*/ 13][0] = tloop772delay[0];
assign v1_wr_data_input[/*idx742=*/ 11][/*idx22=*/ 13][0] = v777;


//TerminatorOp

//} Unrolled body 11 of loop742.
//DEBUG: /*idx742=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop742.
//DEBUG: /*idx742=*/ 4'd12, expected 12
//printTimeOffset
reg tloop775delay[1:0] = '{default:0} ;
always@(*) tloop775delay[0] <= tloop775;
generate
genvar i779;

for(i779 = 1; i779<= 1; i779= i779 + 1) begin
always@(posedge clk) begin
tloop775delay[i779] <= tloop775delay[i779-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop778 = tloop775delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v780 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][220] = tloop775delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 12][/*idx22=*/ 13][0] = tloop775delay[0];
assign v1_wr_data_valid[/*idx742=*/ 12][/*idx22=*/ 13][0] = tloop775delay[0];
assign v1_wr_data_input[/*idx742=*/ 12][/*idx22=*/ 13][0] = v780;


//TerminatorOp

//} Unrolled body 12 of loop742.
//DEBUG: /*idx742=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop742.
//DEBUG: /*idx742=*/ 4'd13, expected 13
//printTimeOffset
reg tloop778delay[1:0] = '{default:0} ;
always@(*) tloop778delay[0] <= tloop778;
generate
genvar i782;

for(i782 = 1; i782<= 1; i782= i782 + 1) begin
always@(posedge clk) begin
tloop778delay[i782] <= tloop778delay[i782-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop781 = tloop778delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v783 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][221] = tloop778delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 13][/*idx22=*/ 13][0] = tloop778delay[0];
assign v1_wr_data_valid[/*idx742=*/ 13][/*idx22=*/ 13][0] = tloop778delay[0];
assign v1_wr_data_input[/*idx742=*/ 13][/*idx22=*/ 13][0] = v783;


//TerminatorOp

//} Unrolled body 13 of loop742.
//DEBUG: /*idx742=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop742.
//DEBUG: /*idx742=*/ 4'd14, expected 14
//printTimeOffset
reg tloop781delay[1:0] = '{default:0} ;
always@(*) tloop781delay[0] <= tloop781;
generate
genvar i785;

for(i785 = 1; i785<= 1; i785= i785 + 1) begin
always@(posedge clk) begin
tloop781delay[i785] <= tloop781delay[i785-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop784 = tloop781delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v786 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][222] = tloop781delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 14][/*idx22=*/ 13][0] = tloop781delay[0];
assign v1_wr_data_valid[/*idx742=*/ 14][/*idx22=*/ 13][0] = tloop781delay[0];
assign v1_wr_data_input[/*idx742=*/ 14][/*idx22=*/ 13][0] = v786;


//TerminatorOp

//} Unrolled body 14 of loop742.
//DEBUG: /*idx742=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop742.
//DEBUG: /*idx742=*/ 4'd15, expected 15
//printTimeOffset
reg tloop784delay[1:0] = '{default:0} ;
always@(*) tloop784delay[0] <= tloop784;
generate
genvar i788;

for(i788 = 1; i788<= 1; i788= i788 + 1) begin
always@(posedge clk) begin
tloop784delay[i788] <= tloop784delay[i788-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop787 = tloop784delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v789 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][223] = tloop784delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx742=*/ 15][/*idx22=*/ 13][0] = tloop784delay[0];
assign v1_wr_data_valid[/*idx742=*/ 15][/*idx22=*/ 13][0] = tloop784delay[0];
assign v1_wr_data_input[/*idx742=*/ 15][/*idx22=*/ 13][0] = v789;


//TerminatorOp

//} Unrolled body 15 of loop742.
//DEBUG: /*idx742=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t790;
assign t790 = tloop787;
//printTimeOffset
reg t790delay[1:0] = '{default:0} ;
always@(*) t790delay[0] <= t790;
generate
genvar i791;

for(i791 = 1; i791<= 1; i791= i791 + 1) begin
always@(posedge clk) begin
t790delay[i791] <= t790delay[i791-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop737 = t790delay[1];

//TerminatorOp

//} Unrolled body 13 of loop22.
//DEBUG: /*idx22=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop22.
//DEBUG: /*idx22=*/ 4'd14, expected 14
//printTimeOffset
reg tloop737delay[0:0] = '{default:0} ;
always@(*) tloop737delay[0] <= tloop737;
generate
genvar i793;

for(i793 = 1; i793<= 0; i793= i793 + 1) begin
always@(posedge clk) begin
tloop737delay[i793] <= tloop737delay[i793-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg795[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg795[0] <= tloop737;
always@(posedge clk) shiftreg795[/*v5=*/ 1:1] <= shiftreg795[/*v5=*/ 0:0];
wire v794 = shiftreg795[/*v5=*/ 1];
//printTimeOffset
reg v794delay[1:0] = '{default:0} ;
always@(*) v794delay[0] <= v794;
generate
genvar i796;

for(i796 = 1; i796<= 1; i796= i796 + 1) begin
always@(posedge clk) begin
v794delay[i796] <= v794delay[i796-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop797.
//DEBUG: /*idx797=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop798 = v794delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v799 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][224] = v794delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 0][/*idx22=*/ 14][0] = v794delay[0];
assign v1_wr_data_valid[/*idx797=*/ 0][/*idx22=*/ 14][0] = v794delay[0];
assign v1_wr_data_input[/*idx797=*/ 0][/*idx22=*/ 14][0] = v799;


//TerminatorOp

//} Unrolled body 0 of loop797.
//DEBUG: /*idx797=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop797.
//DEBUG: /*idx797=*/ 1'd1, expected 1
//printTimeOffset
reg tloop798delay[1:0] = '{default:0} ;
always@(*) tloop798delay[0] <= tloop798;
generate
genvar i801;

for(i801 = 1; i801<= 1; i801= i801 + 1) begin
always@(posedge clk) begin
tloop798delay[i801] <= tloop798delay[i801-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop800 = tloop798delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v802 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][225] = tloop798delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 1][/*idx22=*/ 14][0] = tloop798delay[0];
assign v1_wr_data_valid[/*idx797=*/ 1][/*idx22=*/ 14][0] = tloop798delay[0];
assign v1_wr_data_input[/*idx797=*/ 1][/*idx22=*/ 14][0] = v802;


//TerminatorOp

//} Unrolled body 1 of loop797.
//DEBUG: /*idx797=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop797.
//DEBUG: /*idx797=*/ 2'd2, expected 2
//printTimeOffset
reg tloop800delay[1:0] = '{default:0} ;
always@(*) tloop800delay[0] <= tloop800;
generate
genvar i804;

for(i804 = 1; i804<= 1; i804= i804 + 1) begin
always@(posedge clk) begin
tloop800delay[i804] <= tloop800delay[i804-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop803 = tloop800delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v805 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][226] = tloop800delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 2][/*idx22=*/ 14][0] = tloop800delay[0];
assign v1_wr_data_valid[/*idx797=*/ 2][/*idx22=*/ 14][0] = tloop800delay[0];
assign v1_wr_data_input[/*idx797=*/ 2][/*idx22=*/ 14][0] = v805;


//TerminatorOp

//} Unrolled body 2 of loop797.
//DEBUG: /*idx797=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop797.
//DEBUG: /*idx797=*/ 2'd3, expected 3
//printTimeOffset
reg tloop803delay[1:0] = '{default:0} ;
always@(*) tloop803delay[0] <= tloop803;
generate
genvar i807;

for(i807 = 1; i807<= 1; i807= i807 + 1) begin
always@(posedge clk) begin
tloop803delay[i807] <= tloop803delay[i807-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop806 = tloop803delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v808 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][227] = tloop803delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 3][/*idx22=*/ 14][0] = tloop803delay[0];
assign v1_wr_data_valid[/*idx797=*/ 3][/*idx22=*/ 14][0] = tloop803delay[0];
assign v1_wr_data_input[/*idx797=*/ 3][/*idx22=*/ 14][0] = v808;


//TerminatorOp

//} Unrolled body 3 of loop797.
//DEBUG: /*idx797=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop797.
//DEBUG: /*idx797=*/ 3'd4, expected 4
//printTimeOffset
reg tloop806delay[1:0] = '{default:0} ;
always@(*) tloop806delay[0] <= tloop806;
generate
genvar i810;

for(i810 = 1; i810<= 1; i810= i810 + 1) begin
always@(posedge clk) begin
tloop806delay[i810] <= tloop806delay[i810-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop809 = tloop806delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v811 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][228] = tloop806delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 4][/*idx22=*/ 14][0] = tloop806delay[0];
assign v1_wr_data_valid[/*idx797=*/ 4][/*idx22=*/ 14][0] = tloop806delay[0];
assign v1_wr_data_input[/*idx797=*/ 4][/*idx22=*/ 14][0] = v811;


//TerminatorOp

//} Unrolled body 4 of loop797.
//DEBUG: /*idx797=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop797.
//DEBUG: /*idx797=*/ 3'd5, expected 5
//printTimeOffset
reg tloop809delay[1:0] = '{default:0} ;
always@(*) tloop809delay[0] <= tloop809;
generate
genvar i813;

for(i813 = 1; i813<= 1; i813= i813 + 1) begin
always@(posedge clk) begin
tloop809delay[i813] <= tloop809delay[i813-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop812 = tloop809delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v814 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][229] = tloop809delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 5][/*idx22=*/ 14][0] = tloop809delay[0];
assign v1_wr_data_valid[/*idx797=*/ 5][/*idx22=*/ 14][0] = tloop809delay[0];
assign v1_wr_data_input[/*idx797=*/ 5][/*idx22=*/ 14][0] = v814;


//TerminatorOp

//} Unrolled body 5 of loop797.
//DEBUG: /*idx797=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop797.
//DEBUG: /*idx797=*/ 3'd6, expected 6
//printTimeOffset
reg tloop812delay[1:0] = '{default:0} ;
always@(*) tloop812delay[0] <= tloop812;
generate
genvar i816;

for(i816 = 1; i816<= 1; i816= i816 + 1) begin
always@(posedge clk) begin
tloop812delay[i816] <= tloop812delay[i816-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop815 = tloop812delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v817 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][230] = tloop812delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 6][/*idx22=*/ 14][0] = tloop812delay[0];
assign v1_wr_data_valid[/*idx797=*/ 6][/*idx22=*/ 14][0] = tloop812delay[0];
assign v1_wr_data_input[/*idx797=*/ 6][/*idx22=*/ 14][0] = v817;


//TerminatorOp

//} Unrolled body 6 of loop797.
//DEBUG: /*idx797=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop797.
//DEBUG: /*idx797=*/ 3'd7, expected 7
//printTimeOffset
reg tloop815delay[1:0] = '{default:0} ;
always@(*) tloop815delay[0] <= tloop815;
generate
genvar i819;

for(i819 = 1; i819<= 1; i819= i819 + 1) begin
always@(posedge clk) begin
tloop815delay[i819] <= tloop815delay[i819-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop818 = tloop815delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v820 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][231] = tloop815delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 7][/*idx22=*/ 14][0] = tloop815delay[0];
assign v1_wr_data_valid[/*idx797=*/ 7][/*idx22=*/ 14][0] = tloop815delay[0];
assign v1_wr_data_input[/*idx797=*/ 7][/*idx22=*/ 14][0] = v820;


//TerminatorOp

//} Unrolled body 7 of loop797.
//DEBUG: /*idx797=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop797.
//DEBUG: /*idx797=*/ 4'd8, expected 8
//printTimeOffset
reg tloop818delay[1:0] = '{default:0} ;
always@(*) tloop818delay[0] <= tloop818;
generate
genvar i822;

for(i822 = 1; i822<= 1; i822= i822 + 1) begin
always@(posedge clk) begin
tloop818delay[i822] <= tloop818delay[i822-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop821 = tloop818delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v823 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][232] = tloop818delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 8][/*idx22=*/ 14][0] = tloop818delay[0];
assign v1_wr_data_valid[/*idx797=*/ 8][/*idx22=*/ 14][0] = tloop818delay[0];
assign v1_wr_data_input[/*idx797=*/ 8][/*idx22=*/ 14][0] = v823;


//TerminatorOp

//} Unrolled body 8 of loop797.
//DEBUG: /*idx797=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop797.
//DEBUG: /*idx797=*/ 4'd9, expected 9
//printTimeOffset
reg tloop821delay[1:0] = '{default:0} ;
always@(*) tloop821delay[0] <= tloop821;
generate
genvar i825;

for(i825 = 1; i825<= 1; i825= i825 + 1) begin
always@(posedge clk) begin
tloop821delay[i825] <= tloop821delay[i825-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop824 = tloop821delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v826 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][233] = tloop821delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 9][/*idx22=*/ 14][0] = tloop821delay[0];
assign v1_wr_data_valid[/*idx797=*/ 9][/*idx22=*/ 14][0] = tloop821delay[0];
assign v1_wr_data_input[/*idx797=*/ 9][/*idx22=*/ 14][0] = v826;


//TerminatorOp

//} Unrolled body 9 of loop797.
//DEBUG: /*idx797=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop797.
//DEBUG: /*idx797=*/ 4'd10, expected 10
//printTimeOffset
reg tloop824delay[1:0] = '{default:0} ;
always@(*) tloop824delay[0] <= tloop824;
generate
genvar i828;

for(i828 = 1; i828<= 1; i828= i828 + 1) begin
always@(posedge clk) begin
tloop824delay[i828] <= tloop824delay[i828-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop827 = tloop824delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v829 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][234] = tloop824delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 10][/*idx22=*/ 14][0] = tloop824delay[0];
assign v1_wr_data_valid[/*idx797=*/ 10][/*idx22=*/ 14][0] = tloop824delay[0];
assign v1_wr_data_input[/*idx797=*/ 10][/*idx22=*/ 14][0] = v829;


//TerminatorOp

//} Unrolled body 10 of loop797.
//DEBUG: /*idx797=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop797.
//DEBUG: /*idx797=*/ 4'd11, expected 11
//printTimeOffset
reg tloop827delay[1:0] = '{default:0} ;
always@(*) tloop827delay[0] <= tloop827;
generate
genvar i831;

for(i831 = 1; i831<= 1; i831= i831 + 1) begin
always@(posedge clk) begin
tloop827delay[i831] <= tloop827delay[i831-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop830 = tloop827delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v832 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][235] = tloop827delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 11][/*idx22=*/ 14][0] = tloop827delay[0];
assign v1_wr_data_valid[/*idx797=*/ 11][/*idx22=*/ 14][0] = tloop827delay[0];
assign v1_wr_data_input[/*idx797=*/ 11][/*idx22=*/ 14][0] = v832;


//TerminatorOp

//} Unrolled body 11 of loop797.
//DEBUG: /*idx797=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop797.
//DEBUG: /*idx797=*/ 4'd12, expected 12
//printTimeOffset
reg tloop830delay[1:0] = '{default:0} ;
always@(*) tloop830delay[0] <= tloop830;
generate
genvar i834;

for(i834 = 1; i834<= 1; i834= i834 + 1) begin
always@(posedge clk) begin
tloop830delay[i834] <= tloop830delay[i834-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop833 = tloop830delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v835 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][236] = tloop830delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 12][/*idx22=*/ 14][0] = tloop830delay[0];
assign v1_wr_data_valid[/*idx797=*/ 12][/*idx22=*/ 14][0] = tloop830delay[0];
assign v1_wr_data_input[/*idx797=*/ 12][/*idx22=*/ 14][0] = v835;


//TerminatorOp

//} Unrolled body 12 of loop797.
//DEBUG: /*idx797=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop797.
//DEBUG: /*idx797=*/ 4'd13, expected 13
//printTimeOffset
reg tloop833delay[1:0] = '{default:0} ;
always@(*) tloop833delay[0] <= tloop833;
generate
genvar i837;

for(i837 = 1; i837<= 1; i837= i837 + 1) begin
always@(posedge clk) begin
tloop833delay[i837] <= tloop833delay[i837-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop836 = tloop833delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v838 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][237] = tloop833delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 13][/*idx22=*/ 14][0] = tloop833delay[0];
assign v1_wr_data_valid[/*idx797=*/ 13][/*idx22=*/ 14][0] = tloop833delay[0];
assign v1_wr_data_input[/*idx797=*/ 13][/*idx22=*/ 14][0] = v838;


//TerminatorOp

//} Unrolled body 13 of loop797.
//DEBUG: /*idx797=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop797.
//DEBUG: /*idx797=*/ 4'd14, expected 14
//printTimeOffset
reg tloop836delay[1:0] = '{default:0} ;
always@(*) tloop836delay[0] <= tloop836;
generate
genvar i840;

for(i840 = 1; i840<= 1; i840= i840 + 1) begin
always@(posedge clk) begin
tloop836delay[i840] <= tloop836delay[i840-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop839 = tloop836delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v841 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][238] = tloop836delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 14][/*idx22=*/ 14][0] = tloop836delay[0];
assign v1_wr_data_valid[/*idx797=*/ 14][/*idx22=*/ 14][0] = tloop836delay[0];
assign v1_wr_data_input[/*idx797=*/ 14][/*idx22=*/ 14][0] = v841;


//TerminatorOp

//} Unrolled body 14 of loop797.
//DEBUG: /*idx797=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop797.
//DEBUG: /*idx797=*/ 4'd15, expected 15
//printTimeOffset
reg tloop839delay[1:0] = '{default:0} ;
always@(*) tloop839delay[0] <= tloop839;
generate
genvar i843;

for(i843 = 1; i843<= 1; i843= i843 + 1) begin
always@(posedge clk) begin
tloop839delay[i843] <= tloop839delay[i843-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop842 = tloop839delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v844 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][239] = tloop839delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx797=*/ 15][/*idx22=*/ 14][0] = tloop839delay[0];
assign v1_wr_data_valid[/*idx797=*/ 15][/*idx22=*/ 14][0] = tloop839delay[0];
assign v1_wr_data_input[/*idx797=*/ 15][/*idx22=*/ 14][0] = v844;


//TerminatorOp

//} Unrolled body 15 of loop797.
//DEBUG: /*idx797=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t845;
assign t845 = tloop842;
//printTimeOffset
reg t845delay[1:0] = '{default:0} ;
always@(*) t845delay[0] <= t845;
generate
genvar i846;

for(i846 = 1; i846<= 1; i846= i846 + 1) begin
always@(posedge clk) begin
t845delay[i846] <= t845delay[i846-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop792 = t845delay[1];

//TerminatorOp

//} Unrolled body 14 of loop22.
//DEBUG: /*idx22=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop22.
//DEBUG: /*idx22=*/ 4'd15, expected 15
//printTimeOffset
reg tloop792delay[0:0] = '{default:0} ;
always@(*) tloop792delay[0] <= tloop792;
generate
genvar i848;

for(i848 = 1; i848<= 0; i848= i848 + 1) begin
always@(posedge clk) begin
tloop792delay[i848] <= tloop792delay[i848-1];
end
end
endgenerate


//DelayOp at loc("test/HIR/matmul.mlir":50:12)
reg[0:0]shiftreg850[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg850[0] <= tloop792;
always@(posedge clk) shiftreg850[/*v5=*/ 1:1] <= shiftreg850[/*v5=*/ 0:0];
wire v849 = shiftreg850[/*v5=*/ 1];
//printTimeOffset
reg v849delay[1:0] = '{default:0} ;
always@(*) v849delay[0] <= v849;
generate
genvar i851;

for(i851 = 1; i851<= 1; i851= i851 + 1) begin
always@(posedge clk) begin
v849delay[i851] <= v849delay[i851-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":51:13)

//{ Unrolled body 0 of loop852.
//DEBUG: /*idx852=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop853 = v849delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v854 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][240] = v849delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 0][/*idx22=*/ 15][0] = v849delay[0];
assign v1_wr_data_valid[/*idx852=*/ 0][/*idx22=*/ 15][0] = v849delay[0];
assign v1_wr_data_input[/*idx852=*/ 0][/*idx22=*/ 15][0] = v854;


//TerminatorOp

//} Unrolled body 0 of loop852.
//DEBUG: /*idx852=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop852.
//DEBUG: /*idx852=*/ 1'd1, expected 1
//printTimeOffset
reg tloop853delay[1:0] = '{default:0} ;
always@(*) tloop853delay[0] <= tloop853;
generate
genvar i856;

for(i856 = 1; i856<= 1; i856= i856 + 1) begin
always@(posedge clk) begin
tloop853delay[i856] <= tloop853delay[i856-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop855 = tloop853delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v857 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][241] = tloop853delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 1][/*idx22=*/ 15][0] = tloop853delay[0];
assign v1_wr_data_valid[/*idx852=*/ 1][/*idx22=*/ 15][0] = tloop853delay[0];
assign v1_wr_data_input[/*idx852=*/ 1][/*idx22=*/ 15][0] = v857;


//TerminatorOp

//} Unrolled body 1 of loop852.
//DEBUG: /*idx852=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop852.
//DEBUG: /*idx852=*/ 2'd2, expected 2
//printTimeOffset
reg tloop855delay[1:0] = '{default:0} ;
always@(*) tloop855delay[0] <= tloop855;
generate
genvar i859;

for(i859 = 1; i859<= 1; i859= i859 + 1) begin
always@(posedge clk) begin
tloop855delay[i859] <= tloop855delay[i859-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop858 = tloop855delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v860 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][242] = tloop855delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 2][/*idx22=*/ 15][0] = tloop855delay[0];
assign v1_wr_data_valid[/*idx852=*/ 2][/*idx22=*/ 15][0] = tloop855delay[0];
assign v1_wr_data_input[/*idx852=*/ 2][/*idx22=*/ 15][0] = v860;


//TerminatorOp

//} Unrolled body 2 of loop852.
//DEBUG: /*idx852=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop852.
//DEBUG: /*idx852=*/ 2'd3, expected 3
//printTimeOffset
reg tloop858delay[1:0] = '{default:0} ;
always@(*) tloop858delay[0] <= tloop858;
generate
genvar i862;

for(i862 = 1; i862<= 1; i862= i862 + 1) begin
always@(posedge clk) begin
tloop858delay[i862] <= tloop858delay[i862-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop861 = tloop858delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v863 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][243] = tloop858delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 3][/*idx22=*/ 15][0] = tloop858delay[0];
assign v1_wr_data_valid[/*idx852=*/ 3][/*idx22=*/ 15][0] = tloop858delay[0];
assign v1_wr_data_input[/*idx852=*/ 3][/*idx22=*/ 15][0] = v863;


//TerminatorOp

//} Unrolled body 3 of loop852.
//DEBUG: /*idx852=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop852.
//DEBUG: /*idx852=*/ 3'd4, expected 4
//printTimeOffset
reg tloop861delay[1:0] = '{default:0} ;
always@(*) tloop861delay[0] <= tloop861;
generate
genvar i865;

for(i865 = 1; i865<= 1; i865= i865 + 1) begin
always@(posedge clk) begin
tloop861delay[i865] <= tloop861delay[i865-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop864 = tloop861delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v866 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][244] = tloop861delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 4][/*idx22=*/ 15][0] = tloop861delay[0];
assign v1_wr_data_valid[/*idx852=*/ 4][/*idx22=*/ 15][0] = tloop861delay[0];
assign v1_wr_data_input[/*idx852=*/ 4][/*idx22=*/ 15][0] = v866;


//TerminatorOp

//} Unrolled body 4 of loop852.
//DEBUG: /*idx852=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop852.
//DEBUG: /*idx852=*/ 3'd5, expected 5
//printTimeOffset
reg tloop864delay[1:0] = '{default:0} ;
always@(*) tloop864delay[0] <= tloop864;
generate
genvar i868;

for(i868 = 1; i868<= 1; i868= i868 + 1) begin
always@(posedge clk) begin
tloop864delay[i868] <= tloop864delay[i868-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop867 = tloop864delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v869 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][245] = tloop864delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 5][/*idx22=*/ 15][0] = tloop864delay[0];
assign v1_wr_data_valid[/*idx852=*/ 5][/*idx22=*/ 15][0] = tloop864delay[0];
assign v1_wr_data_input[/*idx852=*/ 5][/*idx22=*/ 15][0] = v869;


//TerminatorOp

//} Unrolled body 5 of loop852.
//DEBUG: /*idx852=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop852.
//DEBUG: /*idx852=*/ 3'd6, expected 6
//printTimeOffset
reg tloop867delay[1:0] = '{default:0} ;
always@(*) tloop867delay[0] <= tloop867;
generate
genvar i871;

for(i871 = 1; i871<= 1; i871= i871 + 1) begin
always@(posedge clk) begin
tloop867delay[i871] <= tloop867delay[i871-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop870 = tloop867delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v872 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][246] = tloop867delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 6][/*idx22=*/ 15][0] = tloop867delay[0];
assign v1_wr_data_valid[/*idx852=*/ 6][/*idx22=*/ 15][0] = tloop867delay[0];
assign v1_wr_data_input[/*idx852=*/ 6][/*idx22=*/ 15][0] = v872;


//TerminatorOp

//} Unrolled body 6 of loop852.
//DEBUG: /*idx852=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop852.
//DEBUG: /*idx852=*/ 3'd7, expected 7
//printTimeOffset
reg tloop870delay[1:0] = '{default:0} ;
always@(*) tloop870delay[0] <= tloop870;
generate
genvar i874;

for(i874 = 1; i874<= 1; i874= i874 + 1) begin
always@(posedge clk) begin
tloop870delay[i874] <= tloop870delay[i874-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop873 = tloop870delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v875 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][247] = tloop870delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 7][/*idx22=*/ 15][0] = tloop870delay[0];
assign v1_wr_data_valid[/*idx852=*/ 7][/*idx22=*/ 15][0] = tloop870delay[0];
assign v1_wr_data_input[/*idx852=*/ 7][/*idx22=*/ 15][0] = v875;


//TerminatorOp

//} Unrolled body 7 of loop852.
//DEBUG: /*idx852=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop852.
//DEBUG: /*idx852=*/ 4'd8, expected 8
//printTimeOffset
reg tloop873delay[1:0] = '{default:0} ;
always@(*) tloop873delay[0] <= tloop873;
generate
genvar i877;

for(i877 = 1; i877<= 1; i877= i877 + 1) begin
always@(posedge clk) begin
tloop873delay[i877] <= tloop873delay[i877-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop876 = tloop873delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v878 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][248] = tloop873delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 8][/*idx22=*/ 15][0] = tloop873delay[0];
assign v1_wr_data_valid[/*idx852=*/ 8][/*idx22=*/ 15][0] = tloop873delay[0];
assign v1_wr_data_input[/*idx852=*/ 8][/*idx22=*/ 15][0] = v878;


//TerminatorOp

//} Unrolled body 8 of loop852.
//DEBUG: /*idx852=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop852.
//DEBUG: /*idx852=*/ 4'd9, expected 9
//printTimeOffset
reg tloop876delay[1:0] = '{default:0} ;
always@(*) tloop876delay[0] <= tloop876;
generate
genvar i880;

for(i880 = 1; i880<= 1; i880= i880 + 1) begin
always@(posedge clk) begin
tloop876delay[i880] <= tloop876delay[i880-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop879 = tloop876delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v881 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][249] = tloop876delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 9][/*idx22=*/ 15][0] = tloop876delay[0];
assign v1_wr_data_valid[/*idx852=*/ 9][/*idx22=*/ 15][0] = tloop876delay[0];
assign v1_wr_data_input[/*idx852=*/ 9][/*idx22=*/ 15][0] = v881;


//TerminatorOp

//} Unrolled body 9 of loop852.
//DEBUG: /*idx852=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop852.
//DEBUG: /*idx852=*/ 4'd10, expected 10
//printTimeOffset
reg tloop879delay[1:0] = '{default:0} ;
always@(*) tloop879delay[0] <= tloop879;
generate
genvar i883;

for(i883 = 1; i883<= 1; i883= i883 + 1) begin
always@(posedge clk) begin
tloop879delay[i883] <= tloop879delay[i883-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop882 = tloop879delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v884 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][250] = tloop879delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 10][/*idx22=*/ 15][0] = tloop879delay[0];
assign v1_wr_data_valid[/*idx852=*/ 10][/*idx22=*/ 15][0] = tloop879delay[0];
assign v1_wr_data_input[/*idx852=*/ 10][/*idx22=*/ 15][0] = v884;


//TerminatorOp

//} Unrolled body 10 of loop852.
//DEBUG: /*idx852=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop852.
//DEBUG: /*idx852=*/ 4'd11, expected 11
//printTimeOffset
reg tloop882delay[1:0] = '{default:0} ;
always@(*) tloop882delay[0] <= tloop882;
generate
genvar i886;

for(i886 = 1; i886<= 1; i886= i886 + 1) begin
always@(posedge clk) begin
tloop882delay[i886] <= tloop882delay[i886-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop885 = tloop882delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v887 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][251] = tloop882delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 11][/*idx22=*/ 15][0] = tloop882delay[0];
assign v1_wr_data_valid[/*idx852=*/ 11][/*idx22=*/ 15][0] = tloop882delay[0];
assign v1_wr_data_input[/*idx852=*/ 11][/*idx22=*/ 15][0] = v887;


//TerminatorOp

//} Unrolled body 11 of loop852.
//DEBUG: /*idx852=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop852.
//DEBUG: /*idx852=*/ 4'd12, expected 12
//printTimeOffset
reg tloop885delay[1:0] = '{default:0} ;
always@(*) tloop885delay[0] <= tloop885;
generate
genvar i889;

for(i889 = 1; i889<= 1; i889= i889 + 1) begin
always@(posedge clk) begin
tloop885delay[i889] <= tloop885delay[i889-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop888 = tloop885delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v890 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][252] = tloop885delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 12][/*idx22=*/ 15][0] = tloop885delay[0];
assign v1_wr_data_valid[/*idx852=*/ 12][/*idx22=*/ 15][0] = tloop885delay[0];
assign v1_wr_data_input[/*idx852=*/ 12][/*idx22=*/ 15][0] = v890;


//TerminatorOp

//} Unrolled body 12 of loop852.
//DEBUG: /*idx852=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop852.
//DEBUG: /*idx852=*/ 4'd13, expected 13
//printTimeOffset
reg tloop888delay[1:0] = '{default:0} ;
always@(*) tloop888delay[0] <= tloop888;
generate
genvar i892;

for(i892 = 1; i892<= 1; i892= i892 + 1) begin
always@(posedge clk) begin
tloop888delay[i892] <= tloop888delay[i892-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop891 = tloop888delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v893 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][253] = tloop888delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 13][/*idx22=*/ 15][0] = tloop888delay[0];
assign v1_wr_data_valid[/*idx852=*/ 13][/*idx22=*/ 15][0] = tloop888delay[0];
assign v1_wr_data_input[/*idx852=*/ 13][/*idx22=*/ 15][0] = v893;


//TerminatorOp

//} Unrolled body 13 of loop852.
//DEBUG: /*idx852=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop852.
//DEBUG: /*idx852=*/ 4'd14, expected 14
//printTimeOffset
reg tloop891delay[1:0] = '{default:0} ;
always@(*) tloop891delay[0] <= tloop891;
generate
genvar i895;

for(i895 = 1; i895<= 1; i895= i895 + 1) begin
always@(posedge clk) begin
tloop891delay[i895] <= tloop891delay[i895-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop894 = tloop891delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v896 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][254] = tloop891delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 14][/*idx22=*/ 15][0] = tloop891delay[0];
assign v1_wr_data_valid[/*idx852=*/ 14][/*idx22=*/ 15][0] = tloop891delay[0];
assign v1_wr_data_input[/*idx852=*/ 14][/*idx22=*/ 15][0] = v896;


//TerminatorOp

//} Unrolled body 14 of loop852.
//DEBUG: /*idx852=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop852.
//DEBUG: /*idx852=*/ 4'd15, expected 15
//printTimeOffset
reg tloop894delay[1:0] = '{default:0} ;
always@(*) tloop894delay[0] <= tloop894;
generate
genvar i898;

for(i898 = 1; i898<= 1; i898= i898 + 1) begin
always@(posedge clk) begin
tloop894delay[i898] <= tloop894delay[i898-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":52:9)
wire tloop897 = tloop894delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":53:15)
wire[31:0] v899 = v10_rd_data[/*v4=*/ 0];
assign v10_rd_en_input[/*v4=*/ 0][255] = tloop894delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":54:9)
assign v1_wr_en_input[/*idx852=*/ 15][/*idx22=*/ 15][0] = tloop894delay[0];
assign v1_wr_data_valid[/*idx852=*/ 15][/*idx22=*/ 15][0] = tloop894delay[0];
assign v1_wr_data_input[/*idx852=*/ 15][/*idx22=*/ 15][0] = v899;


//TerminatorOp

//} Unrolled body 15 of loop852.
//DEBUG: /*idx852=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t900;
assign t900 = tloop897;
//printTimeOffset
reg t900delay[1:0] = '{default:0} ;
always@(*) t900delay[0] <= t900;
generate
genvar i901;

for(i901 = 1; i901<= 1; i901= i901 + 1) begin
always@(posedge clk) begin
t900delay[i901] <= t900delay[i901-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":56:5)
wire tloop847 = t900delay[1];

//TerminatorOp

//} Unrolled body 15 of loop22.
//DEBUG: /*idx22=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t902;
assign t902 = tloop847;
//printTimeOffset
reg t902delay[0:0] = '{default:0} ;
always@(*) t902delay[0] <= t902;
generate
genvar i903;

for(i903 = 1; i903<= 0; i903= i903 + 1) begin
always@(posedge clk) begin
t902delay[i903] <= t902delay[i903-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//ReturnOp at loc("test/HIR/matmul.mlir":58:3)
assign out0 = t902;
endmodule
module kernel(
//Outputs.

//Inputs.

//MemrefType : port = r.
output reg[3:0] v0_addr[15:0],
output wire v0_rd_en[15:0],
input wire[31:0] v0_rd_data[15:0],
//MemrefType : port = r.
output wire v1_rd_en[15:0][15:0],
input wire[31:0] v1_rd_data[15:0][15:0],
//MemrefType : port = w.
output reg[3:0] v2_addr[15:0],
output wire v2_wr_en[15:0],
output reg[31:0] v2_wr_data[15:0],
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

wire v0_addr_valid [15:0] [15:0] ;
wire [3:0] v0_addr_input [15:0] [15:0];
 generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
always@(*) begin
if(v0_addr_valid[i0][0] )
v0_addr[i0] = v0_addr_input[i0][0];
else if (v0_addr_valid[i0][1])
v0_addr[i0] = v0_addr_input[i0][1];
else if (v0_addr_valid[i0][2])
v0_addr[i0] = v0_addr_input[i0][2];
else if (v0_addr_valid[i0][3])
v0_addr[i0] = v0_addr_input[i0][3];
else if (v0_addr_valid[i0][4])
v0_addr[i0] = v0_addr_input[i0][4];
else if (v0_addr_valid[i0][5])
v0_addr[i0] = v0_addr_input[i0][5];
else if (v0_addr_valid[i0][6])
v0_addr[i0] = v0_addr_input[i0][6];
else if (v0_addr_valid[i0][7])
v0_addr[i0] = v0_addr_input[i0][7];
else if (v0_addr_valid[i0][8])
v0_addr[i0] = v0_addr_input[i0][8];
else if (v0_addr_valid[i0][9])
v0_addr[i0] = v0_addr_input[i0][9];
else if (v0_addr_valid[i0][10])
v0_addr[i0] = v0_addr_input[i0][10];
else if (v0_addr_valid[i0][11])
v0_addr[i0] = v0_addr_input[i0][11];
else if (v0_addr_valid[i0][12])
v0_addr[i0] = v0_addr_input[i0][12];
else if (v0_addr_valid[i0][13])
v0_addr[i0] = v0_addr_input[i0][13];
else if (v0_addr_valid[i0][14])
v0_addr[i0] = v0_addr_input[i0][14];
else if (v0_addr_valid[i0][15])
v0_addr[i0] = v0_addr_input[i0][15];
else
 v0_addr[i0] = 'x;
end
end
endgenerate

wire [15:0] v0_rd_en_input [15:0];
generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
assign v0_rd_en [i0] =| v0_rd_en_input [i0];
end
endgenerate


wire [0:0] v1_rd_en_input [15:0][15:0];
generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
for(genvar i1 = 0; i1 < 16;i1=i1 + 1) begin
assign v1_rd_en [i0][i1] =| v1_rd_en_input [i0][i1];
end
end
endgenerate


wire v2_addr_valid [15:0] [0:0] ;
wire [3:0] v2_addr_input [15:0] [0:0];
 generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
always@(*) begin
if(v2_addr_valid[i0][0] )
v2_addr[i0] = v2_addr_input[i0][0];
else
 v2_addr[i0] = 'x;
end
end
endgenerate

wire [0:0] v2_wr_en_input [15:0];
generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
assign v2_wr_en [i0] =| v2_wr_en_input [i0];
end
endgenerate
wire v2_wr_data_valid [15:0] [0:0] ;
wire [31:0] v2_wr_data_input [15:0] [0:0];
 generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
always@(*) begin
if(v2_wr_data_valid[i0][0] )
v2_wr_data[i0] = v2_wr_data_input[i0][0];
else
 v2_wr_data[i0] = 'x;
end
end
endgenerate


//printTimeOffset
reg tstartdelay[0:0] = '{default:0} ;
always@(*) tstartdelay[0] <= tstart;
generate
genvar i4;

for(i4 = 1; i4<= 0; i4= i4 + 1) begin
always@(posedge clk) begin
tstartdelay[i4] <= tstartdelay[i4-1];
end
end
endgenerate


//ConstantOp at loc("test/HIR/matmul.mlir":66:8)
//constant v5 = 1'd0;

//ConstantOp at loc("test/HIR/matmul.mlir":67:8)
//constant v6 = 1'd1;

//ConstantOp at loc("test/HIR/matmul.mlir":68:8)
//constant [1:0] v7 = 2'd2;

//ConstantOp at loc("test/HIR/matmul.mlir":69:8)
//constant [1:0] v8 = 2'd3;

//ConstantOp at loc("test/HIR/matmul.mlir":70:8)
//constant [2:0] v9 = 3'd4;

//ConstantOp at loc("test/HIR/matmul.mlir":71:9)
//constant [4:0] v10 = 5'd16;

//ForOp at loc("test/HIR/matmul.mlir":73:4)

//{ Loop11

reg[31:0] idx11 ;
reg[4:0] ub11 ;
reg[0:0] step11 ;
wire tloop_in11;
reg tloop11;
reg tfinish11;
always@(posedge clk) begin
 if(/*tstart=*/ tstartdelay[0]) begin
   idx11 <= /*v5=*/ 1'd0; //lower bound.
   step11 <= /*v6=*/ 1'd1;
   ub11 <= /*v10=*/ 5'd16;
   tloop11 <= (/*v10=*/ 5'd16 > /*v5=*/ 1'd0);
   tfinish11 <=!(/*v10=*/ 5'd16 > /*v5=*/ 1'd0);
 end
 else if (tloop_in11) begin
   idx11 <= idx11 + step11; //increment
   tloop11 <= (idx11 + step11) < ub11;
   tfinish11 <= !((idx11 + step11) < ub11);
 end
 else begin
   tloop11 <= 1'b0;
   tfinish11 <= 1'b0;
 end
end
//Loop11 body
//printTimeOffset
reg tloop11delay[15:0] = '{default:0} ;
always@(*) tloop11delay[0] <= tloop11;
generate
genvar i12;

for(i12 = 1; i12<= 15; i12= i12 + 1) begin
always@(posedge clk) begin
tloop11delay[i12] <= tloop11delay[i12-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":74:5)
assign tloop_in11 = tloop11delay[0];

//UnrollForOp at loc("test/HIR/matmul.mlir":75:5)

//{ Unrolled body 0 of loop13.
//DEBUG: /*idx13=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop14 = tloop11delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v15[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v15[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop16.
//DEBUG: /*idx16=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop17 = tloop11delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg19[/*idx16=*/ 0:0] = '{default:0};
always@(*) shiftreg19[0] <= idx11;
wire [31:0] v18 = shiftreg19[/*idx16=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 0][0] = tloop11delay[0];
assign v0_addr_input[/*idx16=*/ 0][0] = {v18[3:0]};
wire[31:0] v20 = v0_rd_data[/*idx16=*/ 0];
assign v0_rd_en_input[/*idx16=*/ 0][0] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v21 = /*idx16=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg23[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg23[0] <= v20;
wire [31:0] v22 = shiftreg23[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v24 = v1_rd_data[/*idx16=*/ 0][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 0][/*idx13=*/ 0][0] = tloop11delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v25;
mult mult26(v25,
v22,
v24,
tloop11delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v27 = v15[/*idx16=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v28;
add add29(v28,
v25,
v27,
tloop11delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[1] = v28;

//TerminatorOp

//} Unrolled body 0 of loop16.
//DEBUG: /*idx16=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop16.
//DEBUG: /*idx16=*/ 1'd1, expected 1
//printTimeOffset
reg tloop17delay[3:0] = '{default:0} ;
always@(*) tloop17delay[0] <= tloop17;
generate
genvar i31;

for(i31 = 1; i31<= 3; i31= i31 + 1) begin
always@(posedge clk) begin
tloop17delay[i31] <= tloop17delay[i31-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop30 = tloop17delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg33[/*idx16=*/ 1:0] = '{default:0};
always@(*) shiftreg33[0] <= idx11;
always@(posedge clk) shiftreg33[/*idx16=*/ 1:1] <= shiftreg33[/*idx16=*/ 0:0];
wire [31:0] v32 = shiftreg33[/*idx16=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 1][0] = tloop11delay[1];
assign v0_addr_input[/*idx16=*/ 1][0] = {v32[3:0]};
wire[31:0] v34 = v0_rd_data[/*idx16=*/ 1];
assign v0_rd_en_input[/*idx16=*/ 1][0] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v35 = /*idx16=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg37[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg37[0] <= v34;
wire [31:0] v36 = shiftreg37[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v38 = v1_rd_data[/*idx16=*/ 1][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 1][/*idx13=*/ 0][0] = tloop17delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v39;
mult mult40(v39,
v36,
v38,
tloop17delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v41 = v15[/*idx16=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v42;
add add43(v42,
v39,
v41,
tloop17delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[2] = v42;

//TerminatorOp

//} Unrolled body 1 of loop16.
//DEBUG: /*idx16=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop16.
//DEBUG: /*idx16=*/ 2'd2, expected 2
//printTimeOffset
reg tloop30delay[3:0] = '{default:0} ;
always@(*) tloop30delay[0] <= tloop30;
generate
genvar i45;

for(i45 = 1; i45<= 3; i45= i45 + 1) begin
always@(posedge clk) begin
tloop30delay[i45] <= tloop30delay[i45-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop44 = tloop30delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg47[/*idx16=*/ 2:0] = '{default:0};
always@(*) shiftreg47[0] <= idx11;
always@(posedge clk) shiftreg47[/*idx16=*/ 2:1] <= shiftreg47[/*idx16=*/ 1:0];
wire [31:0] v46 = shiftreg47[/*idx16=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 2][0] = tloop11delay[2];
assign v0_addr_input[/*idx16=*/ 2][0] = {v46[3:0]};
wire[31:0] v48 = v0_rd_data[/*idx16=*/ 2];
assign v0_rd_en_input[/*idx16=*/ 2][0] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v49 = /*idx16=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg51[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg51[0] <= v48;
wire [31:0] v50 = shiftreg51[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v52 = v1_rd_data[/*idx16=*/ 2][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 2][/*idx13=*/ 0][0] = tloop30delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v53;
mult mult54(v53,
v50,
v52,
tloop30delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v55 = v15[/*idx16=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v56;
add add57(v56,
v53,
v55,
tloop30delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[3] = v56;

//TerminatorOp

//} Unrolled body 2 of loop16.
//DEBUG: /*idx16=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop16.
//DEBUG: /*idx16=*/ 2'd3, expected 3
//printTimeOffset
reg tloop44delay[3:0] = '{default:0} ;
always@(*) tloop44delay[0] <= tloop44;
generate
genvar i59;

for(i59 = 1; i59<= 3; i59= i59 + 1) begin
always@(posedge clk) begin
tloop44delay[i59] <= tloop44delay[i59-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop58 = tloop44delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg61[/*idx16=*/ 3:0] = '{default:0};
always@(*) shiftreg61[0] <= idx11;
always@(posedge clk) shiftreg61[/*idx16=*/ 3:1] <= shiftreg61[/*idx16=*/ 2:0];
wire [31:0] v60 = shiftreg61[/*idx16=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 3][0] = tloop11delay[3];
assign v0_addr_input[/*idx16=*/ 3][0] = {v60[3:0]};
wire[31:0] v62 = v0_rd_data[/*idx16=*/ 3];
assign v0_rd_en_input[/*idx16=*/ 3][0] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v63 = /*idx16=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg65[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg65[0] <= v62;
wire [31:0] v64 = shiftreg65[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v66 = v1_rd_data[/*idx16=*/ 3][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 3][/*idx13=*/ 0][0] = tloop44delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v67;
mult mult68(v67,
v64,
v66,
tloop44delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v69 = v15[/*idx16=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v70;
add add71(v70,
v67,
v69,
tloop44delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[4] = v70;

//TerminatorOp

//} Unrolled body 3 of loop16.
//DEBUG: /*idx16=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop16.
//DEBUG: /*idx16=*/ 3'd4, expected 4
//printTimeOffset
reg tloop58delay[3:0] = '{default:0} ;
always@(*) tloop58delay[0] <= tloop58;
generate
genvar i73;

for(i73 = 1; i73<= 3; i73= i73 + 1) begin
always@(posedge clk) begin
tloop58delay[i73] <= tloop58delay[i73-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop72 = tloop58delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg75[/*idx16=*/ 4:0] = '{default:0};
always@(*) shiftreg75[0] <= idx11;
always@(posedge clk) shiftreg75[/*idx16=*/ 4:1] <= shiftreg75[/*idx16=*/ 3:0];
wire [31:0] v74 = shiftreg75[/*idx16=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 4][0] = tloop11delay[4];
assign v0_addr_input[/*idx16=*/ 4][0] = {v74[3:0]};
wire[31:0] v76 = v0_rd_data[/*idx16=*/ 4];
assign v0_rd_en_input[/*idx16=*/ 4][0] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v77 = /*idx16=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg79[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg79[0] <= v76;
wire [31:0] v78 = shiftreg79[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v80 = v1_rd_data[/*idx16=*/ 4][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 4][/*idx13=*/ 0][0] = tloop58delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v81;
mult mult82(v81,
v78,
v80,
tloop58delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v83 = v15[/*idx16=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v84;
add add85(v84,
v81,
v83,
tloop58delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[5] = v84;

//TerminatorOp

//} Unrolled body 4 of loop16.
//DEBUG: /*idx16=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop16.
//DEBUG: /*idx16=*/ 3'd5, expected 5
//printTimeOffset
reg tloop72delay[3:0] = '{default:0} ;
always@(*) tloop72delay[0] <= tloop72;
generate
genvar i87;

for(i87 = 1; i87<= 3; i87= i87 + 1) begin
always@(posedge clk) begin
tloop72delay[i87] <= tloop72delay[i87-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop86 = tloop72delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg89[/*idx16=*/ 5:0] = '{default:0};
always@(*) shiftreg89[0] <= idx11;
always@(posedge clk) shiftreg89[/*idx16=*/ 5:1] <= shiftreg89[/*idx16=*/ 4:0];
wire [31:0] v88 = shiftreg89[/*idx16=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 5][0] = tloop11delay[5];
assign v0_addr_input[/*idx16=*/ 5][0] = {v88[3:0]};
wire[31:0] v90 = v0_rd_data[/*idx16=*/ 5];
assign v0_rd_en_input[/*idx16=*/ 5][0] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v91 = /*idx16=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg93[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg93[0] <= v90;
wire [31:0] v92 = shiftreg93[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v94 = v1_rd_data[/*idx16=*/ 5][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 5][/*idx13=*/ 0][0] = tloop72delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v95;
mult mult96(v95,
v92,
v94,
tloop72delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v97 = v15[/*idx16=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v98;
add add99(v98,
v95,
v97,
tloop72delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[6] = v98;

//TerminatorOp

//} Unrolled body 5 of loop16.
//DEBUG: /*idx16=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop16.
//DEBUG: /*idx16=*/ 3'd6, expected 6
//printTimeOffset
reg tloop86delay[3:0] = '{default:0} ;
always@(*) tloop86delay[0] <= tloop86;
generate
genvar i101;

for(i101 = 1; i101<= 3; i101= i101 + 1) begin
always@(posedge clk) begin
tloop86delay[i101] <= tloop86delay[i101-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop100 = tloop86delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg103[/*idx16=*/ 6:0] = '{default:0};
always@(*) shiftreg103[0] <= idx11;
always@(posedge clk) shiftreg103[/*idx16=*/ 6:1] <= shiftreg103[/*idx16=*/ 5:0];
wire [31:0] v102 = shiftreg103[/*idx16=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 6][0] = tloop11delay[6];
assign v0_addr_input[/*idx16=*/ 6][0] = {v102[3:0]};
wire[31:0] v104 = v0_rd_data[/*idx16=*/ 6];
assign v0_rd_en_input[/*idx16=*/ 6][0] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v105 = /*idx16=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg107[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg107[0] <= v104;
wire [31:0] v106 = shiftreg107[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v108 = v1_rd_data[/*idx16=*/ 6][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 6][/*idx13=*/ 0][0] = tloop86delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v109;
mult mult110(v109,
v106,
v108,
tloop86delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v111 = v15[/*idx16=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v112;
add add113(v112,
v109,
v111,
tloop86delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[7] = v112;

//TerminatorOp

//} Unrolled body 6 of loop16.
//DEBUG: /*idx16=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop16.
//DEBUG: /*idx16=*/ 3'd7, expected 7
//printTimeOffset
reg tloop100delay[3:0] = '{default:0} ;
always@(*) tloop100delay[0] <= tloop100;
generate
genvar i115;

for(i115 = 1; i115<= 3; i115= i115 + 1) begin
always@(posedge clk) begin
tloop100delay[i115] <= tloop100delay[i115-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop114 = tloop100delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg117[/*idx16=*/ 7:0] = '{default:0};
always@(*) shiftreg117[0] <= idx11;
always@(posedge clk) shiftreg117[/*idx16=*/ 7:1] <= shiftreg117[/*idx16=*/ 6:0];
wire [31:0] v116 = shiftreg117[/*idx16=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 7][0] = tloop11delay[7];
assign v0_addr_input[/*idx16=*/ 7][0] = {v116[3:0]};
wire[31:0] v118 = v0_rd_data[/*idx16=*/ 7];
assign v0_rd_en_input[/*idx16=*/ 7][0] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v119 = /*idx16=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg121[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg121[0] <= v118;
wire [31:0] v120 = shiftreg121[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v122 = v1_rd_data[/*idx16=*/ 7][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 7][/*idx13=*/ 0][0] = tloop100delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v123;
mult mult124(v123,
v120,
v122,
tloop100delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v125 = v15[/*idx16=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v126;
add add127(v126,
v123,
v125,
tloop100delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[8] = v126;

//TerminatorOp

//} Unrolled body 7 of loop16.
//DEBUG: /*idx16=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop16.
//DEBUG: /*idx16=*/ 4'd8, expected 8
//printTimeOffset
reg tloop114delay[3:0] = '{default:0} ;
always@(*) tloop114delay[0] <= tloop114;
generate
genvar i129;

for(i129 = 1; i129<= 3; i129= i129 + 1) begin
always@(posedge clk) begin
tloop114delay[i129] <= tloop114delay[i129-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop128 = tloop114delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg131[/*idx16=*/ 8:0] = '{default:0};
always@(*) shiftreg131[0] <= idx11;
always@(posedge clk) shiftreg131[/*idx16=*/ 8:1] <= shiftreg131[/*idx16=*/ 7:0];
wire [31:0] v130 = shiftreg131[/*idx16=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 8][0] = tloop11delay[8];
assign v0_addr_input[/*idx16=*/ 8][0] = {v130[3:0]};
wire[31:0] v132 = v0_rd_data[/*idx16=*/ 8];
assign v0_rd_en_input[/*idx16=*/ 8][0] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v133 = /*idx16=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg135[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg135[0] <= v132;
wire [31:0] v134 = shiftreg135[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v136 = v1_rd_data[/*idx16=*/ 8][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 8][/*idx13=*/ 0][0] = tloop114delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v137;
mult mult138(v137,
v134,
v136,
tloop114delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v139 = v15[/*idx16=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v140;
add add141(v140,
v137,
v139,
tloop114delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[9] = v140;

//TerminatorOp

//} Unrolled body 8 of loop16.
//DEBUG: /*idx16=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop16.
//DEBUG: /*idx16=*/ 4'd9, expected 9
//printTimeOffset
reg tloop128delay[3:0] = '{default:0} ;
always@(*) tloop128delay[0] <= tloop128;
generate
genvar i143;

for(i143 = 1; i143<= 3; i143= i143 + 1) begin
always@(posedge clk) begin
tloop128delay[i143] <= tloop128delay[i143-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop142 = tloop128delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg145[/*idx16=*/ 9:0] = '{default:0};
always@(*) shiftreg145[0] <= idx11;
always@(posedge clk) shiftreg145[/*idx16=*/ 9:1] <= shiftreg145[/*idx16=*/ 8:0];
wire [31:0] v144 = shiftreg145[/*idx16=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 9][0] = tloop11delay[9];
assign v0_addr_input[/*idx16=*/ 9][0] = {v144[3:0]};
wire[31:0] v146 = v0_rd_data[/*idx16=*/ 9];
assign v0_rd_en_input[/*idx16=*/ 9][0] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v147 = /*idx16=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg149[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg149[0] <= v146;
wire [31:0] v148 = shiftreg149[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v150 = v1_rd_data[/*idx16=*/ 9][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 9][/*idx13=*/ 0][0] = tloop128delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v151;
mult mult152(v151,
v148,
v150,
tloop128delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v153 = v15[/*idx16=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v154;
add add155(v154,
v151,
v153,
tloop128delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[10] = v154;

//TerminatorOp

//} Unrolled body 9 of loop16.
//DEBUG: /*idx16=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop16.
//DEBUG: /*idx16=*/ 4'd10, expected 10
//printTimeOffset
reg tloop142delay[3:0] = '{default:0} ;
always@(*) tloop142delay[0] <= tloop142;
generate
genvar i157;

for(i157 = 1; i157<= 3; i157= i157 + 1) begin
always@(posedge clk) begin
tloop142delay[i157] <= tloop142delay[i157-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop156 = tloop142delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg159[/*idx16=*/ 10:0] = '{default:0};
always@(*) shiftreg159[0] <= idx11;
always@(posedge clk) shiftreg159[/*idx16=*/ 10:1] <= shiftreg159[/*idx16=*/ 9:0];
wire [31:0] v158 = shiftreg159[/*idx16=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 10][0] = tloop11delay[10];
assign v0_addr_input[/*idx16=*/ 10][0] = {v158[3:0]};
wire[31:0] v160 = v0_rd_data[/*idx16=*/ 10];
assign v0_rd_en_input[/*idx16=*/ 10][0] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v161 = /*idx16=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg163[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg163[0] <= v160;
wire [31:0] v162 = shiftreg163[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v164 = v1_rd_data[/*idx16=*/ 10][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 10][/*idx13=*/ 0][0] = tloop142delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v165;
mult mult166(v165,
v162,
v164,
tloop142delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v167 = v15[/*idx16=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v168;
add add169(v168,
v165,
v167,
tloop142delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[11] = v168;

//TerminatorOp

//} Unrolled body 10 of loop16.
//DEBUG: /*idx16=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop16.
//DEBUG: /*idx16=*/ 4'd11, expected 11
//printTimeOffset
reg tloop156delay[3:0] = '{default:0} ;
always@(*) tloop156delay[0] <= tloop156;
generate
genvar i171;

for(i171 = 1; i171<= 3; i171= i171 + 1) begin
always@(posedge clk) begin
tloop156delay[i171] <= tloop156delay[i171-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop170 = tloop156delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg173[/*idx16=*/ 11:0] = '{default:0};
always@(*) shiftreg173[0] <= idx11;
always@(posedge clk) shiftreg173[/*idx16=*/ 11:1] <= shiftreg173[/*idx16=*/ 10:0];
wire [31:0] v172 = shiftreg173[/*idx16=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 11][0] = tloop11delay[11];
assign v0_addr_input[/*idx16=*/ 11][0] = {v172[3:0]};
wire[31:0] v174 = v0_rd_data[/*idx16=*/ 11];
assign v0_rd_en_input[/*idx16=*/ 11][0] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v175 = /*idx16=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg177[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg177[0] <= v174;
wire [31:0] v176 = shiftreg177[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v178 = v1_rd_data[/*idx16=*/ 11][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 11][/*idx13=*/ 0][0] = tloop156delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v179;
mult mult180(v179,
v176,
v178,
tloop156delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v181 = v15[/*idx16=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v182;
add add183(v182,
v179,
v181,
tloop156delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[12] = v182;

//TerminatorOp

//} Unrolled body 11 of loop16.
//DEBUG: /*idx16=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop16.
//DEBUG: /*idx16=*/ 4'd12, expected 12
//printTimeOffset
reg tloop170delay[3:0] = '{default:0} ;
always@(*) tloop170delay[0] <= tloop170;
generate
genvar i185;

for(i185 = 1; i185<= 3; i185= i185 + 1) begin
always@(posedge clk) begin
tloop170delay[i185] <= tloop170delay[i185-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop184 = tloop170delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg187[/*idx16=*/ 12:0] = '{default:0};
always@(*) shiftreg187[0] <= idx11;
always@(posedge clk) shiftreg187[/*idx16=*/ 12:1] <= shiftreg187[/*idx16=*/ 11:0];
wire [31:0] v186 = shiftreg187[/*idx16=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 12][0] = tloop11delay[12];
assign v0_addr_input[/*idx16=*/ 12][0] = {v186[3:0]};
wire[31:0] v188 = v0_rd_data[/*idx16=*/ 12];
assign v0_rd_en_input[/*idx16=*/ 12][0] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v189 = /*idx16=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg191[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg191[0] <= v188;
wire [31:0] v190 = shiftreg191[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v192 = v1_rd_data[/*idx16=*/ 12][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 12][/*idx13=*/ 0][0] = tloop170delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v193;
mult mult194(v193,
v190,
v192,
tloop170delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v195 = v15[/*idx16=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v196;
add add197(v196,
v193,
v195,
tloop170delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[13] = v196;

//TerminatorOp

//} Unrolled body 12 of loop16.
//DEBUG: /*idx16=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop16.
//DEBUG: /*idx16=*/ 4'd13, expected 13
//printTimeOffset
reg tloop184delay[3:0] = '{default:0} ;
always@(*) tloop184delay[0] <= tloop184;
generate
genvar i199;

for(i199 = 1; i199<= 3; i199= i199 + 1) begin
always@(posedge clk) begin
tloop184delay[i199] <= tloop184delay[i199-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop198 = tloop184delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg201[/*idx16=*/ 13:0] = '{default:0};
always@(*) shiftreg201[0] <= idx11;
always@(posedge clk) shiftreg201[/*idx16=*/ 13:1] <= shiftreg201[/*idx16=*/ 12:0];
wire [31:0] v200 = shiftreg201[/*idx16=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 13][0] = tloop11delay[13];
assign v0_addr_input[/*idx16=*/ 13][0] = {v200[3:0]};
wire[31:0] v202 = v0_rd_data[/*idx16=*/ 13];
assign v0_rd_en_input[/*idx16=*/ 13][0] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v203 = /*idx16=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg205[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg205[0] <= v202;
wire [31:0] v204 = shiftreg205[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v206 = v1_rd_data[/*idx16=*/ 13][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 13][/*idx13=*/ 0][0] = tloop184delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v207;
mult mult208(v207,
v204,
v206,
tloop184delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v209 = v15[/*idx16=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v210;
add add211(v210,
v207,
v209,
tloop184delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[14] = v210;

//TerminatorOp

//} Unrolled body 13 of loop16.
//DEBUG: /*idx16=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop16.
//DEBUG: /*idx16=*/ 4'd14, expected 14
//printTimeOffset
reg tloop198delay[3:0] = '{default:0} ;
always@(*) tloop198delay[0] <= tloop198;
generate
genvar i213;

for(i213 = 1; i213<= 3; i213= i213 + 1) begin
always@(posedge clk) begin
tloop198delay[i213] <= tloop198delay[i213-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop212 = tloop198delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg215[/*idx16=*/ 14:0] = '{default:0};
always@(*) shiftreg215[0] <= idx11;
always@(posedge clk) shiftreg215[/*idx16=*/ 14:1] <= shiftreg215[/*idx16=*/ 13:0];
wire [31:0] v214 = shiftreg215[/*idx16=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 14][0] = tloop11delay[14];
assign v0_addr_input[/*idx16=*/ 14][0] = {v214[3:0]};
wire[31:0] v216 = v0_rd_data[/*idx16=*/ 14];
assign v0_rd_en_input[/*idx16=*/ 14][0] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v217 = /*idx16=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg219[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg219[0] <= v216;
wire [31:0] v218 = shiftreg219[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v220 = v1_rd_data[/*idx16=*/ 14][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 14][/*idx13=*/ 0][0] = tloop198delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v221;
mult mult222(v221,
v218,
v220,
tloop198delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v223 = v15[/*idx16=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v224;
add add225(v224,
v221,
v223,
tloop198delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[15] = v224;

//TerminatorOp

//} Unrolled body 14 of loop16.
//DEBUG: /*idx16=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop16.
//DEBUG: /*idx16=*/ 4'd15, expected 15
//printTimeOffset
reg tloop212delay[3:0] = '{default:0} ;
always@(*) tloop212delay[0] <= tloop212;
generate
genvar i227;

for(i227 = 1; i227<= 3; i227= i227 + 1) begin
always@(posedge clk) begin
tloop212delay[i227] <= tloop212delay[i227-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop226 = tloop212delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg229[/*idx16=*/ 15:0] = '{default:0};
always@(*) shiftreg229[0] <= idx11;
always@(posedge clk) shiftreg229[/*idx16=*/ 15:1] <= shiftreg229[/*idx16=*/ 14:0];
wire [31:0] v228 = shiftreg229[/*idx16=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx16=*/ 15][0] = tloop11delay[15];
assign v0_addr_input[/*idx16=*/ 15][0] = {v228[3:0]};
wire[31:0] v230 = v0_rd_data[/*idx16=*/ 15];
assign v0_rd_en_input[/*idx16=*/ 15][0] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v231 = /*idx16=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg233[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg233[0] <= v230;
wire [31:0] v232 = shiftreg233[/*idx13=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v234 = v1_rd_data[/*idx16=*/ 15][/*idx13=*/ 0];
assign v1_rd_en_input[/*idx16=*/ 15][/*idx13=*/ 0][0] = tloop212delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v235;
mult mult236(v235,
v232,
v234,
tloop212delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v237 = v15[/*idx16=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v238;
add add239(v238,
v235,
v237,
tloop212delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v15[16] = v238;

//TerminatorOp

//} Unrolled body 15 of loop16.
//DEBUG: /*idx16=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t240;
assign t240 = tloop226;
//printTimeOffset
reg t240delay[3:0] = '{default:0} ;
always@(*) t240delay[0] <= t240;
generate
genvar i241;

for(i241 = 1; i241<= 3; i241= i241 + 1) begin
always@(posedge clk) begin
t240delay[i241] <= t240delay[i241-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v242 = v15[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg244[/*idx13=*/ 0:0] = '{default:0};
always@(*) shiftreg244[0] <= idx11;
wire [31:0] v243 = shiftreg244[/*idx13=*/ 0];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg246[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg246[0] <= v243;
always@(posedge clk) shiftreg246[/*v10=*/ 16:1] <= shiftreg246[/*v10=*/ 15:0];
wire [31:0] v245 = shiftreg246[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg248[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg248[0] <= v245;
always@(posedge clk) shiftreg248[/*v8=*/ 3:1] <= shiftreg248[/*v8=*/ 2:0];
wire [31:0] v247 = shiftreg248[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 0][0] = t240delay[3];
assign v2_addr_input[/*idx13=*/ 0][0] = {v247[3:0]};
assign v2_wr_en_input[/*idx13=*/ 0][0] = t240delay[3];
assign v2_wr_data_valid[/*idx13=*/ 0][0] = t240delay[3];
assign v2_wr_data_input[/*idx13=*/ 0][0] = v242;


//TerminatorOp

//} Unrolled body 0 of loop13.
//DEBUG: /*idx13=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop13.
//DEBUG: /*idx13=*/ 1'd1, expected 1
//printTimeOffset
reg tloop14delay[3:0] = '{default:0} ;
always@(*) tloop14delay[0] <= tloop14;
generate
genvar i250;

for(i250 = 1; i250<= 3; i250= i250 + 1) begin
always@(posedge clk) begin
tloop14delay[i250] <= tloop14delay[i250-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop249 = tloop14delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v251[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v251[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop252.
//DEBUG: /*idx252=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop253 = tloop14delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg255[/*idx252=*/ 0:0] = '{default:0};
always@(*) shiftreg255[0] <= idx11;
wire [31:0] v254 = shiftreg255[/*idx252=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 0][1] = tloop11delay[0];
assign v0_addr_input[/*idx252=*/ 0][1] = {v254[3:0]};
wire[31:0] v256 = v0_rd_data[/*idx252=*/ 0];
assign v0_rd_en_input[/*idx252=*/ 0][1] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v257 = /*idx252=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg259[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg259[0] <= v256;
always@(posedge clk) shiftreg259[/*idx13=*/ 1:1] <= shiftreg259[/*idx13=*/ 0:0];
wire [31:0] v258 = shiftreg259[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v260 = v1_rd_data[/*idx252=*/ 0][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 0][/*idx13=*/ 1][0] = tloop14delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v261;
mult mult262(v261,
v258,
v260,
tloop14delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v263 = v251[/*idx252=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v264;
add add265(v264,
v261,
v263,
tloop14delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[1] = v264;

//TerminatorOp

//} Unrolled body 0 of loop252.
//DEBUG: /*idx252=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop252.
//DEBUG: /*idx252=*/ 1'd1, expected 1
//printTimeOffset
reg tloop253delay[3:0] = '{default:0} ;
always@(*) tloop253delay[0] <= tloop253;
generate
genvar i267;

for(i267 = 1; i267<= 3; i267= i267 + 1) begin
always@(posedge clk) begin
tloop253delay[i267] <= tloop253delay[i267-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop266 = tloop253delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg269[/*idx252=*/ 1:0] = '{default:0};
always@(*) shiftreg269[0] <= idx11;
always@(posedge clk) shiftreg269[/*idx252=*/ 1:1] <= shiftreg269[/*idx252=*/ 0:0];
wire [31:0] v268 = shiftreg269[/*idx252=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 1][1] = tloop11delay[1];
assign v0_addr_input[/*idx252=*/ 1][1] = {v268[3:0]};
wire[31:0] v270 = v0_rd_data[/*idx252=*/ 1];
assign v0_rd_en_input[/*idx252=*/ 1][1] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v271 = /*idx252=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg273[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg273[0] <= v270;
always@(posedge clk) shiftreg273[/*idx13=*/ 1:1] <= shiftreg273[/*idx13=*/ 0:0];
wire [31:0] v272 = shiftreg273[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v274 = v1_rd_data[/*idx252=*/ 1][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 1][/*idx13=*/ 1][0] = tloop253delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v275;
mult mult276(v275,
v272,
v274,
tloop253delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v277 = v251[/*idx252=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v278;
add add279(v278,
v275,
v277,
tloop253delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[2] = v278;

//TerminatorOp

//} Unrolled body 1 of loop252.
//DEBUG: /*idx252=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop252.
//DEBUG: /*idx252=*/ 2'd2, expected 2
//printTimeOffset
reg tloop266delay[3:0] = '{default:0} ;
always@(*) tloop266delay[0] <= tloop266;
generate
genvar i281;

for(i281 = 1; i281<= 3; i281= i281 + 1) begin
always@(posedge clk) begin
tloop266delay[i281] <= tloop266delay[i281-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop280 = tloop266delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg283[/*idx252=*/ 2:0] = '{default:0};
always@(*) shiftreg283[0] <= idx11;
always@(posedge clk) shiftreg283[/*idx252=*/ 2:1] <= shiftreg283[/*idx252=*/ 1:0];
wire [31:0] v282 = shiftreg283[/*idx252=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 2][1] = tloop11delay[2];
assign v0_addr_input[/*idx252=*/ 2][1] = {v282[3:0]};
wire[31:0] v284 = v0_rd_data[/*idx252=*/ 2];
assign v0_rd_en_input[/*idx252=*/ 2][1] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v285 = /*idx252=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg287[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg287[0] <= v284;
always@(posedge clk) shiftreg287[/*idx13=*/ 1:1] <= shiftreg287[/*idx13=*/ 0:0];
wire [31:0] v286 = shiftreg287[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v288 = v1_rd_data[/*idx252=*/ 2][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 2][/*idx13=*/ 1][0] = tloop266delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v289;
mult mult290(v289,
v286,
v288,
tloop266delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v291 = v251[/*idx252=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v292;
add add293(v292,
v289,
v291,
tloop266delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[3] = v292;

//TerminatorOp

//} Unrolled body 2 of loop252.
//DEBUG: /*idx252=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop252.
//DEBUG: /*idx252=*/ 2'd3, expected 3
//printTimeOffset
reg tloop280delay[3:0] = '{default:0} ;
always@(*) tloop280delay[0] <= tloop280;
generate
genvar i295;

for(i295 = 1; i295<= 3; i295= i295 + 1) begin
always@(posedge clk) begin
tloop280delay[i295] <= tloop280delay[i295-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop294 = tloop280delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg297[/*idx252=*/ 3:0] = '{default:0};
always@(*) shiftreg297[0] <= idx11;
always@(posedge clk) shiftreg297[/*idx252=*/ 3:1] <= shiftreg297[/*idx252=*/ 2:0];
wire [31:0] v296 = shiftreg297[/*idx252=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 3][1] = tloop11delay[3];
assign v0_addr_input[/*idx252=*/ 3][1] = {v296[3:0]};
wire[31:0] v298 = v0_rd_data[/*idx252=*/ 3];
assign v0_rd_en_input[/*idx252=*/ 3][1] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v299 = /*idx252=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg301[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg301[0] <= v298;
always@(posedge clk) shiftreg301[/*idx13=*/ 1:1] <= shiftreg301[/*idx13=*/ 0:0];
wire [31:0] v300 = shiftreg301[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v302 = v1_rd_data[/*idx252=*/ 3][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 3][/*idx13=*/ 1][0] = tloop280delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v303;
mult mult304(v303,
v300,
v302,
tloop280delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v305 = v251[/*idx252=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v306;
add add307(v306,
v303,
v305,
tloop280delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[4] = v306;

//TerminatorOp

//} Unrolled body 3 of loop252.
//DEBUG: /*idx252=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop252.
//DEBUG: /*idx252=*/ 3'd4, expected 4
//printTimeOffset
reg tloop294delay[3:0] = '{default:0} ;
always@(*) tloop294delay[0] <= tloop294;
generate
genvar i309;

for(i309 = 1; i309<= 3; i309= i309 + 1) begin
always@(posedge clk) begin
tloop294delay[i309] <= tloop294delay[i309-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop308 = tloop294delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg311[/*idx252=*/ 4:0] = '{default:0};
always@(*) shiftreg311[0] <= idx11;
always@(posedge clk) shiftreg311[/*idx252=*/ 4:1] <= shiftreg311[/*idx252=*/ 3:0];
wire [31:0] v310 = shiftreg311[/*idx252=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 4][1] = tloop11delay[4];
assign v0_addr_input[/*idx252=*/ 4][1] = {v310[3:0]};
wire[31:0] v312 = v0_rd_data[/*idx252=*/ 4];
assign v0_rd_en_input[/*idx252=*/ 4][1] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v313 = /*idx252=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg315[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg315[0] <= v312;
always@(posedge clk) shiftreg315[/*idx13=*/ 1:1] <= shiftreg315[/*idx13=*/ 0:0];
wire [31:0] v314 = shiftreg315[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v316 = v1_rd_data[/*idx252=*/ 4][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 4][/*idx13=*/ 1][0] = tloop294delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v317;
mult mult318(v317,
v314,
v316,
tloop294delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v319 = v251[/*idx252=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v320;
add add321(v320,
v317,
v319,
tloop294delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[5] = v320;

//TerminatorOp

//} Unrolled body 4 of loop252.
//DEBUG: /*idx252=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop252.
//DEBUG: /*idx252=*/ 3'd5, expected 5
//printTimeOffset
reg tloop308delay[3:0] = '{default:0} ;
always@(*) tloop308delay[0] <= tloop308;
generate
genvar i323;

for(i323 = 1; i323<= 3; i323= i323 + 1) begin
always@(posedge clk) begin
tloop308delay[i323] <= tloop308delay[i323-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop322 = tloop308delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg325[/*idx252=*/ 5:0] = '{default:0};
always@(*) shiftreg325[0] <= idx11;
always@(posedge clk) shiftreg325[/*idx252=*/ 5:1] <= shiftreg325[/*idx252=*/ 4:0];
wire [31:0] v324 = shiftreg325[/*idx252=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 5][1] = tloop11delay[5];
assign v0_addr_input[/*idx252=*/ 5][1] = {v324[3:0]};
wire[31:0] v326 = v0_rd_data[/*idx252=*/ 5];
assign v0_rd_en_input[/*idx252=*/ 5][1] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v327 = /*idx252=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg329[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg329[0] <= v326;
always@(posedge clk) shiftreg329[/*idx13=*/ 1:1] <= shiftreg329[/*idx13=*/ 0:0];
wire [31:0] v328 = shiftreg329[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v330 = v1_rd_data[/*idx252=*/ 5][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 5][/*idx13=*/ 1][0] = tloop308delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v331;
mult mult332(v331,
v328,
v330,
tloop308delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v333 = v251[/*idx252=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v334;
add add335(v334,
v331,
v333,
tloop308delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[6] = v334;

//TerminatorOp

//} Unrolled body 5 of loop252.
//DEBUG: /*idx252=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop252.
//DEBUG: /*idx252=*/ 3'd6, expected 6
//printTimeOffset
reg tloop322delay[3:0] = '{default:0} ;
always@(*) tloop322delay[0] <= tloop322;
generate
genvar i337;

for(i337 = 1; i337<= 3; i337= i337 + 1) begin
always@(posedge clk) begin
tloop322delay[i337] <= tloop322delay[i337-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop336 = tloop322delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg339[/*idx252=*/ 6:0] = '{default:0};
always@(*) shiftreg339[0] <= idx11;
always@(posedge clk) shiftreg339[/*idx252=*/ 6:1] <= shiftreg339[/*idx252=*/ 5:0];
wire [31:0] v338 = shiftreg339[/*idx252=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 6][1] = tloop11delay[6];
assign v0_addr_input[/*idx252=*/ 6][1] = {v338[3:0]};
wire[31:0] v340 = v0_rd_data[/*idx252=*/ 6];
assign v0_rd_en_input[/*idx252=*/ 6][1] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v341 = /*idx252=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg343[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg343[0] <= v340;
always@(posedge clk) shiftreg343[/*idx13=*/ 1:1] <= shiftreg343[/*idx13=*/ 0:0];
wire [31:0] v342 = shiftreg343[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v344 = v1_rd_data[/*idx252=*/ 6][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 6][/*idx13=*/ 1][0] = tloop322delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v345;
mult mult346(v345,
v342,
v344,
tloop322delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v347 = v251[/*idx252=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v348;
add add349(v348,
v345,
v347,
tloop322delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[7] = v348;

//TerminatorOp

//} Unrolled body 6 of loop252.
//DEBUG: /*idx252=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop252.
//DEBUG: /*idx252=*/ 3'd7, expected 7
//printTimeOffset
reg tloop336delay[3:0] = '{default:0} ;
always@(*) tloop336delay[0] <= tloop336;
generate
genvar i351;

for(i351 = 1; i351<= 3; i351= i351 + 1) begin
always@(posedge clk) begin
tloop336delay[i351] <= tloop336delay[i351-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop350 = tloop336delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg353[/*idx252=*/ 7:0] = '{default:0};
always@(*) shiftreg353[0] <= idx11;
always@(posedge clk) shiftreg353[/*idx252=*/ 7:1] <= shiftreg353[/*idx252=*/ 6:0];
wire [31:0] v352 = shiftreg353[/*idx252=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 7][1] = tloop11delay[7];
assign v0_addr_input[/*idx252=*/ 7][1] = {v352[3:0]};
wire[31:0] v354 = v0_rd_data[/*idx252=*/ 7];
assign v0_rd_en_input[/*idx252=*/ 7][1] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v355 = /*idx252=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg357[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg357[0] <= v354;
always@(posedge clk) shiftreg357[/*idx13=*/ 1:1] <= shiftreg357[/*idx13=*/ 0:0];
wire [31:0] v356 = shiftreg357[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v358 = v1_rd_data[/*idx252=*/ 7][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 7][/*idx13=*/ 1][0] = tloop336delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v359;
mult mult360(v359,
v356,
v358,
tloop336delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v361 = v251[/*idx252=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v362;
add add363(v362,
v359,
v361,
tloop336delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[8] = v362;

//TerminatorOp

//} Unrolled body 7 of loop252.
//DEBUG: /*idx252=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop252.
//DEBUG: /*idx252=*/ 4'd8, expected 8
//printTimeOffset
reg tloop350delay[3:0] = '{default:0} ;
always@(*) tloop350delay[0] <= tloop350;
generate
genvar i365;

for(i365 = 1; i365<= 3; i365= i365 + 1) begin
always@(posedge clk) begin
tloop350delay[i365] <= tloop350delay[i365-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop364 = tloop350delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg367[/*idx252=*/ 8:0] = '{default:0};
always@(*) shiftreg367[0] <= idx11;
always@(posedge clk) shiftreg367[/*idx252=*/ 8:1] <= shiftreg367[/*idx252=*/ 7:0];
wire [31:0] v366 = shiftreg367[/*idx252=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 8][1] = tloop11delay[8];
assign v0_addr_input[/*idx252=*/ 8][1] = {v366[3:0]};
wire[31:0] v368 = v0_rd_data[/*idx252=*/ 8];
assign v0_rd_en_input[/*idx252=*/ 8][1] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v369 = /*idx252=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg371[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg371[0] <= v368;
always@(posedge clk) shiftreg371[/*idx13=*/ 1:1] <= shiftreg371[/*idx13=*/ 0:0];
wire [31:0] v370 = shiftreg371[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v372 = v1_rd_data[/*idx252=*/ 8][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 8][/*idx13=*/ 1][0] = tloop350delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v373;
mult mult374(v373,
v370,
v372,
tloop350delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v375 = v251[/*idx252=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v376;
add add377(v376,
v373,
v375,
tloop350delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[9] = v376;

//TerminatorOp

//} Unrolled body 8 of loop252.
//DEBUG: /*idx252=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop252.
//DEBUG: /*idx252=*/ 4'd9, expected 9
//printTimeOffset
reg tloop364delay[3:0] = '{default:0} ;
always@(*) tloop364delay[0] <= tloop364;
generate
genvar i379;

for(i379 = 1; i379<= 3; i379= i379 + 1) begin
always@(posedge clk) begin
tloop364delay[i379] <= tloop364delay[i379-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop378 = tloop364delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg381[/*idx252=*/ 9:0] = '{default:0};
always@(*) shiftreg381[0] <= idx11;
always@(posedge clk) shiftreg381[/*idx252=*/ 9:1] <= shiftreg381[/*idx252=*/ 8:0];
wire [31:0] v380 = shiftreg381[/*idx252=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 9][1] = tloop11delay[9];
assign v0_addr_input[/*idx252=*/ 9][1] = {v380[3:0]};
wire[31:0] v382 = v0_rd_data[/*idx252=*/ 9];
assign v0_rd_en_input[/*idx252=*/ 9][1] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v383 = /*idx252=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg385[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg385[0] <= v382;
always@(posedge clk) shiftreg385[/*idx13=*/ 1:1] <= shiftreg385[/*idx13=*/ 0:0];
wire [31:0] v384 = shiftreg385[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v386 = v1_rd_data[/*idx252=*/ 9][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 9][/*idx13=*/ 1][0] = tloop364delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v387;
mult mult388(v387,
v384,
v386,
tloop364delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v389 = v251[/*idx252=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v390;
add add391(v390,
v387,
v389,
tloop364delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[10] = v390;

//TerminatorOp

//} Unrolled body 9 of loop252.
//DEBUG: /*idx252=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop252.
//DEBUG: /*idx252=*/ 4'd10, expected 10
//printTimeOffset
reg tloop378delay[3:0] = '{default:0} ;
always@(*) tloop378delay[0] <= tloop378;
generate
genvar i393;

for(i393 = 1; i393<= 3; i393= i393 + 1) begin
always@(posedge clk) begin
tloop378delay[i393] <= tloop378delay[i393-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop392 = tloop378delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg395[/*idx252=*/ 10:0] = '{default:0};
always@(*) shiftreg395[0] <= idx11;
always@(posedge clk) shiftreg395[/*idx252=*/ 10:1] <= shiftreg395[/*idx252=*/ 9:0];
wire [31:0] v394 = shiftreg395[/*idx252=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 10][1] = tloop11delay[10];
assign v0_addr_input[/*idx252=*/ 10][1] = {v394[3:0]};
wire[31:0] v396 = v0_rd_data[/*idx252=*/ 10];
assign v0_rd_en_input[/*idx252=*/ 10][1] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v397 = /*idx252=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg399[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg399[0] <= v396;
always@(posedge clk) shiftreg399[/*idx13=*/ 1:1] <= shiftreg399[/*idx13=*/ 0:0];
wire [31:0] v398 = shiftreg399[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v400 = v1_rd_data[/*idx252=*/ 10][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 10][/*idx13=*/ 1][0] = tloop378delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v401;
mult mult402(v401,
v398,
v400,
tloop378delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v403 = v251[/*idx252=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v404;
add add405(v404,
v401,
v403,
tloop378delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[11] = v404;

//TerminatorOp

//} Unrolled body 10 of loop252.
//DEBUG: /*idx252=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop252.
//DEBUG: /*idx252=*/ 4'd11, expected 11
//printTimeOffset
reg tloop392delay[3:0] = '{default:0} ;
always@(*) tloop392delay[0] <= tloop392;
generate
genvar i407;

for(i407 = 1; i407<= 3; i407= i407 + 1) begin
always@(posedge clk) begin
tloop392delay[i407] <= tloop392delay[i407-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop406 = tloop392delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg409[/*idx252=*/ 11:0] = '{default:0};
always@(*) shiftreg409[0] <= idx11;
always@(posedge clk) shiftreg409[/*idx252=*/ 11:1] <= shiftreg409[/*idx252=*/ 10:0];
wire [31:0] v408 = shiftreg409[/*idx252=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 11][1] = tloop11delay[11];
assign v0_addr_input[/*idx252=*/ 11][1] = {v408[3:0]};
wire[31:0] v410 = v0_rd_data[/*idx252=*/ 11];
assign v0_rd_en_input[/*idx252=*/ 11][1] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v411 = /*idx252=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg413[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg413[0] <= v410;
always@(posedge clk) shiftreg413[/*idx13=*/ 1:1] <= shiftreg413[/*idx13=*/ 0:0];
wire [31:0] v412 = shiftreg413[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v414 = v1_rd_data[/*idx252=*/ 11][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 11][/*idx13=*/ 1][0] = tloop392delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v415;
mult mult416(v415,
v412,
v414,
tloop392delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v417 = v251[/*idx252=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v418;
add add419(v418,
v415,
v417,
tloop392delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[12] = v418;

//TerminatorOp

//} Unrolled body 11 of loop252.
//DEBUG: /*idx252=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop252.
//DEBUG: /*idx252=*/ 4'd12, expected 12
//printTimeOffset
reg tloop406delay[3:0] = '{default:0} ;
always@(*) tloop406delay[0] <= tloop406;
generate
genvar i421;

for(i421 = 1; i421<= 3; i421= i421 + 1) begin
always@(posedge clk) begin
tloop406delay[i421] <= tloop406delay[i421-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop420 = tloop406delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg423[/*idx252=*/ 12:0] = '{default:0};
always@(*) shiftreg423[0] <= idx11;
always@(posedge clk) shiftreg423[/*idx252=*/ 12:1] <= shiftreg423[/*idx252=*/ 11:0];
wire [31:0] v422 = shiftreg423[/*idx252=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 12][1] = tloop11delay[12];
assign v0_addr_input[/*idx252=*/ 12][1] = {v422[3:0]};
wire[31:0] v424 = v0_rd_data[/*idx252=*/ 12];
assign v0_rd_en_input[/*idx252=*/ 12][1] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v425 = /*idx252=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg427[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg427[0] <= v424;
always@(posedge clk) shiftreg427[/*idx13=*/ 1:1] <= shiftreg427[/*idx13=*/ 0:0];
wire [31:0] v426 = shiftreg427[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v428 = v1_rd_data[/*idx252=*/ 12][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 12][/*idx13=*/ 1][0] = tloop406delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v429;
mult mult430(v429,
v426,
v428,
tloop406delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v431 = v251[/*idx252=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v432;
add add433(v432,
v429,
v431,
tloop406delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[13] = v432;

//TerminatorOp

//} Unrolled body 12 of loop252.
//DEBUG: /*idx252=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop252.
//DEBUG: /*idx252=*/ 4'd13, expected 13
//printTimeOffset
reg tloop420delay[3:0] = '{default:0} ;
always@(*) tloop420delay[0] <= tloop420;
generate
genvar i435;

for(i435 = 1; i435<= 3; i435= i435 + 1) begin
always@(posedge clk) begin
tloop420delay[i435] <= tloop420delay[i435-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop434 = tloop420delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg437[/*idx252=*/ 13:0] = '{default:0};
always@(*) shiftreg437[0] <= idx11;
always@(posedge clk) shiftreg437[/*idx252=*/ 13:1] <= shiftreg437[/*idx252=*/ 12:0];
wire [31:0] v436 = shiftreg437[/*idx252=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 13][1] = tloop11delay[13];
assign v0_addr_input[/*idx252=*/ 13][1] = {v436[3:0]};
wire[31:0] v438 = v0_rd_data[/*idx252=*/ 13];
assign v0_rd_en_input[/*idx252=*/ 13][1] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v439 = /*idx252=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg441[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg441[0] <= v438;
always@(posedge clk) shiftreg441[/*idx13=*/ 1:1] <= shiftreg441[/*idx13=*/ 0:0];
wire [31:0] v440 = shiftreg441[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v442 = v1_rd_data[/*idx252=*/ 13][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 13][/*idx13=*/ 1][0] = tloop420delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v443;
mult mult444(v443,
v440,
v442,
tloop420delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v445 = v251[/*idx252=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v446;
add add447(v446,
v443,
v445,
tloop420delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[14] = v446;

//TerminatorOp

//} Unrolled body 13 of loop252.
//DEBUG: /*idx252=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop252.
//DEBUG: /*idx252=*/ 4'd14, expected 14
//printTimeOffset
reg tloop434delay[3:0] = '{default:0} ;
always@(*) tloop434delay[0] <= tloop434;
generate
genvar i449;

for(i449 = 1; i449<= 3; i449= i449 + 1) begin
always@(posedge clk) begin
tloop434delay[i449] <= tloop434delay[i449-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop448 = tloop434delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg451[/*idx252=*/ 14:0] = '{default:0};
always@(*) shiftreg451[0] <= idx11;
always@(posedge clk) shiftreg451[/*idx252=*/ 14:1] <= shiftreg451[/*idx252=*/ 13:0];
wire [31:0] v450 = shiftreg451[/*idx252=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 14][1] = tloop11delay[14];
assign v0_addr_input[/*idx252=*/ 14][1] = {v450[3:0]};
wire[31:0] v452 = v0_rd_data[/*idx252=*/ 14];
assign v0_rd_en_input[/*idx252=*/ 14][1] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v453 = /*idx252=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg455[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg455[0] <= v452;
always@(posedge clk) shiftreg455[/*idx13=*/ 1:1] <= shiftreg455[/*idx13=*/ 0:0];
wire [31:0] v454 = shiftreg455[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v456 = v1_rd_data[/*idx252=*/ 14][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 14][/*idx13=*/ 1][0] = tloop434delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v457;
mult mult458(v457,
v454,
v456,
tloop434delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v459 = v251[/*idx252=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v460;
add add461(v460,
v457,
v459,
tloop434delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[15] = v460;

//TerminatorOp

//} Unrolled body 14 of loop252.
//DEBUG: /*idx252=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop252.
//DEBUG: /*idx252=*/ 4'd15, expected 15
//printTimeOffset
reg tloop448delay[3:0] = '{default:0} ;
always@(*) tloop448delay[0] <= tloop448;
generate
genvar i463;

for(i463 = 1; i463<= 3; i463= i463 + 1) begin
always@(posedge clk) begin
tloop448delay[i463] <= tloop448delay[i463-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop462 = tloop448delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg465[/*idx252=*/ 15:0] = '{default:0};
always@(*) shiftreg465[0] <= idx11;
always@(posedge clk) shiftreg465[/*idx252=*/ 15:1] <= shiftreg465[/*idx252=*/ 14:0];
wire [31:0] v464 = shiftreg465[/*idx252=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx252=*/ 15][1] = tloop11delay[15];
assign v0_addr_input[/*idx252=*/ 15][1] = {v464[3:0]};
wire[31:0] v466 = v0_rd_data[/*idx252=*/ 15];
assign v0_rd_en_input[/*idx252=*/ 15][1] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v467 = /*idx252=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg469[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg469[0] <= v466;
always@(posedge clk) shiftreg469[/*idx13=*/ 1:1] <= shiftreg469[/*idx13=*/ 0:0];
wire [31:0] v468 = shiftreg469[/*idx13=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v470 = v1_rd_data[/*idx252=*/ 15][/*idx13=*/ 1];
assign v1_rd_en_input[/*idx252=*/ 15][/*idx13=*/ 1][0] = tloop448delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v471;
mult mult472(v471,
v468,
v470,
tloop448delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v473 = v251[/*idx252=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v474;
add add475(v474,
v471,
v473,
tloop448delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v251[16] = v474;

//TerminatorOp

//} Unrolled body 15 of loop252.
//DEBUG: /*idx252=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t476;
assign t476 = tloop462;
//printTimeOffset
reg t476delay[3:0] = '{default:0} ;
always@(*) t476delay[0] <= t476;
generate
genvar i477;

for(i477 = 1; i477<= 3; i477= i477 + 1) begin
always@(posedge clk) begin
t476delay[i477] <= t476delay[i477-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v478 = v251[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg480[/*idx13=*/ 1:0] = '{default:0};
always@(*) shiftreg480[0] <= idx11;
always@(posedge clk) shiftreg480[/*idx13=*/ 1:1] <= shiftreg480[/*idx13=*/ 0:0];
wire [31:0] v479 = shiftreg480[/*idx13=*/ 1];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg482[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg482[0] <= v479;
always@(posedge clk) shiftreg482[/*v10=*/ 16:1] <= shiftreg482[/*v10=*/ 15:0];
wire [31:0] v481 = shiftreg482[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg484[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg484[0] <= v481;
always@(posedge clk) shiftreg484[/*v8=*/ 3:1] <= shiftreg484[/*v8=*/ 2:0];
wire [31:0] v483 = shiftreg484[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 1][0] = t476delay[3];
assign v2_addr_input[/*idx13=*/ 1][0] = {v483[3:0]};
assign v2_wr_en_input[/*idx13=*/ 1][0] = t476delay[3];
assign v2_wr_data_valid[/*idx13=*/ 1][0] = t476delay[3];
assign v2_wr_data_input[/*idx13=*/ 1][0] = v478;


//TerminatorOp

//} Unrolled body 1 of loop13.
//DEBUG: /*idx13=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop13.
//DEBUG: /*idx13=*/ 2'd2, expected 2
//printTimeOffset
reg tloop249delay[3:0] = '{default:0} ;
always@(*) tloop249delay[0] <= tloop249;
generate
genvar i486;

for(i486 = 1; i486<= 3; i486= i486 + 1) begin
always@(posedge clk) begin
tloop249delay[i486] <= tloop249delay[i486-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop485 = tloop249delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v487[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v487[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop488.
//DEBUG: /*idx488=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop489 = tloop249delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg491[/*idx488=*/ 0:0] = '{default:0};
always@(*) shiftreg491[0] <= idx11;
wire [31:0] v490 = shiftreg491[/*idx488=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 0][2] = tloop11delay[0];
assign v0_addr_input[/*idx488=*/ 0][2] = {v490[3:0]};
wire[31:0] v492 = v0_rd_data[/*idx488=*/ 0];
assign v0_rd_en_input[/*idx488=*/ 0][2] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v493 = /*idx488=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg495[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg495[0] <= v492;
always@(posedge clk) shiftreg495[/*idx13=*/ 2:1] <= shiftreg495[/*idx13=*/ 1:0];
wire [31:0] v494 = shiftreg495[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v496 = v1_rd_data[/*idx488=*/ 0][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 0][/*idx13=*/ 2][0] = tloop249delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v497;
mult mult498(v497,
v494,
v496,
tloop249delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v499 = v487[/*idx488=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v500;
add add501(v500,
v497,
v499,
tloop249delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[1] = v500;

//TerminatorOp

//} Unrolled body 0 of loop488.
//DEBUG: /*idx488=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop488.
//DEBUG: /*idx488=*/ 1'd1, expected 1
//printTimeOffset
reg tloop489delay[3:0] = '{default:0} ;
always@(*) tloop489delay[0] <= tloop489;
generate
genvar i503;

for(i503 = 1; i503<= 3; i503= i503 + 1) begin
always@(posedge clk) begin
tloop489delay[i503] <= tloop489delay[i503-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop502 = tloop489delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg505[/*idx488=*/ 1:0] = '{default:0};
always@(*) shiftreg505[0] <= idx11;
always@(posedge clk) shiftreg505[/*idx488=*/ 1:1] <= shiftreg505[/*idx488=*/ 0:0];
wire [31:0] v504 = shiftreg505[/*idx488=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 1][2] = tloop11delay[1];
assign v0_addr_input[/*idx488=*/ 1][2] = {v504[3:0]};
wire[31:0] v506 = v0_rd_data[/*idx488=*/ 1];
assign v0_rd_en_input[/*idx488=*/ 1][2] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v507 = /*idx488=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg509[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg509[0] <= v506;
always@(posedge clk) shiftreg509[/*idx13=*/ 2:1] <= shiftreg509[/*idx13=*/ 1:0];
wire [31:0] v508 = shiftreg509[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v510 = v1_rd_data[/*idx488=*/ 1][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 1][/*idx13=*/ 2][0] = tloop489delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v511;
mult mult512(v511,
v508,
v510,
tloop489delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v513 = v487[/*idx488=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v514;
add add515(v514,
v511,
v513,
tloop489delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[2] = v514;

//TerminatorOp

//} Unrolled body 1 of loop488.
//DEBUG: /*idx488=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop488.
//DEBUG: /*idx488=*/ 2'd2, expected 2
//printTimeOffset
reg tloop502delay[3:0] = '{default:0} ;
always@(*) tloop502delay[0] <= tloop502;
generate
genvar i517;

for(i517 = 1; i517<= 3; i517= i517 + 1) begin
always@(posedge clk) begin
tloop502delay[i517] <= tloop502delay[i517-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop516 = tloop502delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg519[/*idx488=*/ 2:0] = '{default:0};
always@(*) shiftreg519[0] <= idx11;
always@(posedge clk) shiftreg519[/*idx488=*/ 2:1] <= shiftreg519[/*idx488=*/ 1:0];
wire [31:0] v518 = shiftreg519[/*idx488=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 2][2] = tloop11delay[2];
assign v0_addr_input[/*idx488=*/ 2][2] = {v518[3:0]};
wire[31:0] v520 = v0_rd_data[/*idx488=*/ 2];
assign v0_rd_en_input[/*idx488=*/ 2][2] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v521 = /*idx488=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg523[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg523[0] <= v520;
always@(posedge clk) shiftreg523[/*idx13=*/ 2:1] <= shiftreg523[/*idx13=*/ 1:0];
wire [31:0] v522 = shiftreg523[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v524 = v1_rd_data[/*idx488=*/ 2][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 2][/*idx13=*/ 2][0] = tloop502delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v525;
mult mult526(v525,
v522,
v524,
tloop502delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v527 = v487[/*idx488=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v528;
add add529(v528,
v525,
v527,
tloop502delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[3] = v528;

//TerminatorOp

//} Unrolled body 2 of loop488.
//DEBUG: /*idx488=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop488.
//DEBUG: /*idx488=*/ 2'd3, expected 3
//printTimeOffset
reg tloop516delay[3:0] = '{default:0} ;
always@(*) tloop516delay[0] <= tloop516;
generate
genvar i531;

for(i531 = 1; i531<= 3; i531= i531 + 1) begin
always@(posedge clk) begin
tloop516delay[i531] <= tloop516delay[i531-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop530 = tloop516delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg533[/*idx488=*/ 3:0] = '{default:0};
always@(*) shiftreg533[0] <= idx11;
always@(posedge clk) shiftreg533[/*idx488=*/ 3:1] <= shiftreg533[/*idx488=*/ 2:0];
wire [31:0] v532 = shiftreg533[/*idx488=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 3][2] = tloop11delay[3];
assign v0_addr_input[/*idx488=*/ 3][2] = {v532[3:0]};
wire[31:0] v534 = v0_rd_data[/*idx488=*/ 3];
assign v0_rd_en_input[/*idx488=*/ 3][2] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v535 = /*idx488=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg537[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg537[0] <= v534;
always@(posedge clk) shiftreg537[/*idx13=*/ 2:1] <= shiftreg537[/*idx13=*/ 1:0];
wire [31:0] v536 = shiftreg537[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v538 = v1_rd_data[/*idx488=*/ 3][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 3][/*idx13=*/ 2][0] = tloop516delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v539;
mult mult540(v539,
v536,
v538,
tloop516delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v541 = v487[/*idx488=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v542;
add add543(v542,
v539,
v541,
tloop516delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[4] = v542;

//TerminatorOp

//} Unrolled body 3 of loop488.
//DEBUG: /*idx488=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop488.
//DEBUG: /*idx488=*/ 3'd4, expected 4
//printTimeOffset
reg tloop530delay[3:0] = '{default:0} ;
always@(*) tloop530delay[0] <= tloop530;
generate
genvar i545;

for(i545 = 1; i545<= 3; i545= i545 + 1) begin
always@(posedge clk) begin
tloop530delay[i545] <= tloop530delay[i545-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop544 = tloop530delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg547[/*idx488=*/ 4:0] = '{default:0};
always@(*) shiftreg547[0] <= idx11;
always@(posedge clk) shiftreg547[/*idx488=*/ 4:1] <= shiftreg547[/*idx488=*/ 3:0];
wire [31:0] v546 = shiftreg547[/*idx488=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 4][2] = tloop11delay[4];
assign v0_addr_input[/*idx488=*/ 4][2] = {v546[3:0]};
wire[31:0] v548 = v0_rd_data[/*idx488=*/ 4];
assign v0_rd_en_input[/*idx488=*/ 4][2] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v549 = /*idx488=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg551[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg551[0] <= v548;
always@(posedge clk) shiftreg551[/*idx13=*/ 2:1] <= shiftreg551[/*idx13=*/ 1:0];
wire [31:0] v550 = shiftreg551[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v552 = v1_rd_data[/*idx488=*/ 4][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 4][/*idx13=*/ 2][0] = tloop530delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v553;
mult mult554(v553,
v550,
v552,
tloop530delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v555 = v487[/*idx488=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v556;
add add557(v556,
v553,
v555,
tloop530delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[5] = v556;

//TerminatorOp

//} Unrolled body 4 of loop488.
//DEBUG: /*idx488=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop488.
//DEBUG: /*idx488=*/ 3'd5, expected 5
//printTimeOffset
reg tloop544delay[3:0] = '{default:0} ;
always@(*) tloop544delay[0] <= tloop544;
generate
genvar i559;

for(i559 = 1; i559<= 3; i559= i559 + 1) begin
always@(posedge clk) begin
tloop544delay[i559] <= tloop544delay[i559-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop558 = tloop544delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg561[/*idx488=*/ 5:0] = '{default:0};
always@(*) shiftreg561[0] <= idx11;
always@(posedge clk) shiftreg561[/*idx488=*/ 5:1] <= shiftreg561[/*idx488=*/ 4:0];
wire [31:0] v560 = shiftreg561[/*idx488=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 5][2] = tloop11delay[5];
assign v0_addr_input[/*idx488=*/ 5][2] = {v560[3:0]};
wire[31:0] v562 = v0_rd_data[/*idx488=*/ 5];
assign v0_rd_en_input[/*idx488=*/ 5][2] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v563 = /*idx488=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg565[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg565[0] <= v562;
always@(posedge clk) shiftreg565[/*idx13=*/ 2:1] <= shiftreg565[/*idx13=*/ 1:0];
wire [31:0] v564 = shiftreg565[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v566 = v1_rd_data[/*idx488=*/ 5][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 5][/*idx13=*/ 2][0] = tloop544delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v567;
mult mult568(v567,
v564,
v566,
tloop544delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v569 = v487[/*idx488=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v570;
add add571(v570,
v567,
v569,
tloop544delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[6] = v570;

//TerminatorOp

//} Unrolled body 5 of loop488.
//DEBUG: /*idx488=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop488.
//DEBUG: /*idx488=*/ 3'd6, expected 6
//printTimeOffset
reg tloop558delay[3:0] = '{default:0} ;
always@(*) tloop558delay[0] <= tloop558;
generate
genvar i573;

for(i573 = 1; i573<= 3; i573= i573 + 1) begin
always@(posedge clk) begin
tloop558delay[i573] <= tloop558delay[i573-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop572 = tloop558delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg575[/*idx488=*/ 6:0] = '{default:0};
always@(*) shiftreg575[0] <= idx11;
always@(posedge clk) shiftreg575[/*idx488=*/ 6:1] <= shiftreg575[/*idx488=*/ 5:0];
wire [31:0] v574 = shiftreg575[/*idx488=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 6][2] = tloop11delay[6];
assign v0_addr_input[/*idx488=*/ 6][2] = {v574[3:0]};
wire[31:0] v576 = v0_rd_data[/*idx488=*/ 6];
assign v0_rd_en_input[/*idx488=*/ 6][2] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v577 = /*idx488=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg579[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg579[0] <= v576;
always@(posedge clk) shiftreg579[/*idx13=*/ 2:1] <= shiftreg579[/*idx13=*/ 1:0];
wire [31:0] v578 = shiftreg579[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v580 = v1_rd_data[/*idx488=*/ 6][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 6][/*idx13=*/ 2][0] = tloop558delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v581;
mult mult582(v581,
v578,
v580,
tloop558delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v583 = v487[/*idx488=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v584;
add add585(v584,
v581,
v583,
tloop558delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[7] = v584;

//TerminatorOp

//} Unrolled body 6 of loop488.
//DEBUG: /*idx488=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop488.
//DEBUG: /*idx488=*/ 3'd7, expected 7
//printTimeOffset
reg tloop572delay[3:0] = '{default:0} ;
always@(*) tloop572delay[0] <= tloop572;
generate
genvar i587;

for(i587 = 1; i587<= 3; i587= i587 + 1) begin
always@(posedge clk) begin
tloop572delay[i587] <= tloop572delay[i587-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop586 = tloop572delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg589[/*idx488=*/ 7:0] = '{default:0};
always@(*) shiftreg589[0] <= idx11;
always@(posedge clk) shiftreg589[/*idx488=*/ 7:1] <= shiftreg589[/*idx488=*/ 6:0];
wire [31:0] v588 = shiftreg589[/*idx488=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 7][2] = tloop11delay[7];
assign v0_addr_input[/*idx488=*/ 7][2] = {v588[3:0]};
wire[31:0] v590 = v0_rd_data[/*idx488=*/ 7];
assign v0_rd_en_input[/*idx488=*/ 7][2] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v591 = /*idx488=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg593[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg593[0] <= v590;
always@(posedge clk) shiftreg593[/*idx13=*/ 2:1] <= shiftreg593[/*idx13=*/ 1:0];
wire [31:0] v592 = shiftreg593[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v594 = v1_rd_data[/*idx488=*/ 7][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 7][/*idx13=*/ 2][0] = tloop572delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v595;
mult mult596(v595,
v592,
v594,
tloop572delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v597 = v487[/*idx488=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v598;
add add599(v598,
v595,
v597,
tloop572delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[8] = v598;

//TerminatorOp

//} Unrolled body 7 of loop488.
//DEBUG: /*idx488=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop488.
//DEBUG: /*idx488=*/ 4'd8, expected 8
//printTimeOffset
reg tloop586delay[3:0] = '{default:0} ;
always@(*) tloop586delay[0] <= tloop586;
generate
genvar i601;

for(i601 = 1; i601<= 3; i601= i601 + 1) begin
always@(posedge clk) begin
tloop586delay[i601] <= tloop586delay[i601-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop600 = tloop586delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg603[/*idx488=*/ 8:0] = '{default:0};
always@(*) shiftreg603[0] <= idx11;
always@(posedge clk) shiftreg603[/*idx488=*/ 8:1] <= shiftreg603[/*idx488=*/ 7:0];
wire [31:0] v602 = shiftreg603[/*idx488=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 8][2] = tloop11delay[8];
assign v0_addr_input[/*idx488=*/ 8][2] = {v602[3:0]};
wire[31:0] v604 = v0_rd_data[/*idx488=*/ 8];
assign v0_rd_en_input[/*idx488=*/ 8][2] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v605 = /*idx488=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg607[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg607[0] <= v604;
always@(posedge clk) shiftreg607[/*idx13=*/ 2:1] <= shiftreg607[/*idx13=*/ 1:0];
wire [31:0] v606 = shiftreg607[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v608 = v1_rd_data[/*idx488=*/ 8][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 8][/*idx13=*/ 2][0] = tloop586delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v609;
mult mult610(v609,
v606,
v608,
tloop586delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v611 = v487[/*idx488=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v612;
add add613(v612,
v609,
v611,
tloop586delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[9] = v612;

//TerminatorOp

//} Unrolled body 8 of loop488.
//DEBUG: /*idx488=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop488.
//DEBUG: /*idx488=*/ 4'd9, expected 9
//printTimeOffset
reg tloop600delay[3:0] = '{default:0} ;
always@(*) tloop600delay[0] <= tloop600;
generate
genvar i615;

for(i615 = 1; i615<= 3; i615= i615 + 1) begin
always@(posedge clk) begin
tloop600delay[i615] <= tloop600delay[i615-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop614 = tloop600delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg617[/*idx488=*/ 9:0] = '{default:0};
always@(*) shiftreg617[0] <= idx11;
always@(posedge clk) shiftreg617[/*idx488=*/ 9:1] <= shiftreg617[/*idx488=*/ 8:0];
wire [31:0] v616 = shiftreg617[/*idx488=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 9][2] = tloop11delay[9];
assign v0_addr_input[/*idx488=*/ 9][2] = {v616[3:0]};
wire[31:0] v618 = v0_rd_data[/*idx488=*/ 9];
assign v0_rd_en_input[/*idx488=*/ 9][2] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v619 = /*idx488=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg621[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg621[0] <= v618;
always@(posedge clk) shiftreg621[/*idx13=*/ 2:1] <= shiftreg621[/*idx13=*/ 1:0];
wire [31:0] v620 = shiftreg621[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v622 = v1_rd_data[/*idx488=*/ 9][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 9][/*idx13=*/ 2][0] = tloop600delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v623;
mult mult624(v623,
v620,
v622,
tloop600delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v625 = v487[/*idx488=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v626;
add add627(v626,
v623,
v625,
tloop600delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[10] = v626;

//TerminatorOp

//} Unrolled body 9 of loop488.
//DEBUG: /*idx488=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop488.
//DEBUG: /*idx488=*/ 4'd10, expected 10
//printTimeOffset
reg tloop614delay[3:0] = '{default:0} ;
always@(*) tloop614delay[0] <= tloop614;
generate
genvar i629;

for(i629 = 1; i629<= 3; i629= i629 + 1) begin
always@(posedge clk) begin
tloop614delay[i629] <= tloop614delay[i629-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop628 = tloop614delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg631[/*idx488=*/ 10:0] = '{default:0};
always@(*) shiftreg631[0] <= idx11;
always@(posedge clk) shiftreg631[/*idx488=*/ 10:1] <= shiftreg631[/*idx488=*/ 9:0];
wire [31:0] v630 = shiftreg631[/*idx488=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 10][2] = tloop11delay[10];
assign v0_addr_input[/*idx488=*/ 10][2] = {v630[3:0]};
wire[31:0] v632 = v0_rd_data[/*idx488=*/ 10];
assign v0_rd_en_input[/*idx488=*/ 10][2] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v633 = /*idx488=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg635[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg635[0] <= v632;
always@(posedge clk) shiftreg635[/*idx13=*/ 2:1] <= shiftreg635[/*idx13=*/ 1:0];
wire [31:0] v634 = shiftreg635[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v636 = v1_rd_data[/*idx488=*/ 10][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 10][/*idx13=*/ 2][0] = tloop614delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v637;
mult mult638(v637,
v634,
v636,
tloop614delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v639 = v487[/*idx488=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v640;
add add641(v640,
v637,
v639,
tloop614delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[11] = v640;

//TerminatorOp

//} Unrolled body 10 of loop488.
//DEBUG: /*idx488=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop488.
//DEBUG: /*idx488=*/ 4'd11, expected 11
//printTimeOffset
reg tloop628delay[3:0] = '{default:0} ;
always@(*) tloop628delay[0] <= tloop628;
generate
genvar i643;

for(i643 = 1; i643<= 3; i643= i643 + 1) begin
always@(posedge clk) begin
tloop628delay[i643] <= tloop628delay[i643-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop642 = tloop628delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg645[/*idx488=*/ 11:0] = '{default:0};
always@(*) shiftreg645[0] <= idx11;
always@(posedge clk) shiftreg645[/*idx488=*/ 11:1] <= shiftreg645[/*idx488=*/ 10:0];
wire [31:0] v644 = shiftreg645[/*idx488=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 11][2] = tloop11delay[11];
assign v0_addr_input[/*idx488=*/ 11][2] = {v644[3:0]};
wire[31:0] v646 = v0_rd_data[/*idx488=*/ 11];
assign v0_rd_en_input[/*idx488=*/ 11][2] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v647 = /*idx488=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg649[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg649[0] <= v646;
always@(posedge clk) shiftreg649[/*idx13=*/ 2:1] <= shiftreg649[/*idx13=*/ 1:0];
wire [31:0] v648 = shiftreg649[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v650 = v1_rd_data[/*idx488=*/ 11][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 11][/*idx13=*/ 2][0] = tloop628delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v651;
mult mult652(v651,
v648,
v650,
tloop628delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v653 = v487[/*idx488=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v654;
add add655(v654,
v651,
v653,
tloop628delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[12] = v654;

//TerminatorOp

//} Unrolled body 11 of loop488.
//DEBUG: /*idx488=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop488.
//DEBUG: /*idx488=*/ 4'd12, expected 12
//printTimeOffset
reg tloop642delay[3:0] = '{default:0} ;
always@(*) tloop642delay[0] <= tloop642;
generate
genvar i657;

for(i657 = 1; i657<= 3; i657= i657 + 1) begin
always@(posedge clk) begin
tloop642delay[i657] <= tloop642delay[i657-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop656 = tloop642delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg659[/*idx488=*/ 12:0] = '{default:0};
always@(*) shiftreg659[0] <= idx11;
always@(posedge clk) shiftreg659[/*idx488=*/ 12:1] <= shiftreg659[/*idx488=*/ 11:0];
wire [31:0] v658 = shiftreg659[/*idx488=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 12][2] = tloop11delay[12];
assign v0_addr_input[/*idx488=*/ 12][2] = {v658[3:0]};
wire[31:0] v660 = v0_rd_data[/*idx488=*/ 12];
assign v0_rd_en_input[/*idx488=*/ 12][2] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v661 = /*idx488=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg663[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg663[0] <= v660;
always@(posedge clk) shiftreg663[/*idx13=*/ 2:1] <= shiftreg663[/*idx13=*/ 1:0];
wire [31:0] v662 = shiftreg663[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v664 = v1_rd_data[/*idx488=*/ 12][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 12][/*idx13=*/ 2][0] = tloop642delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v665;
mult mult666(v665,
v662,
v664,
tloop642delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v667 = v487[/*idx488=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v668;
add add669(v668,
v665,
v667,
tloop642delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[13] = v668;

//TerminatorOp

//} Unrolled body 12 of loop488.
//DEBUG: /*idx488=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop488.
//DEBUG: /*idx488=*/ 4'd13, expected 13
//printTimeOffset
reg tloop656delay[3:0] = '{default:0} ;
always@(*) tloop656delay[0] <= tloop656;
generate
genvar i671;

for(i671 = 1; i671<= 3; i671= i671 + 1) begin
always@(posedge clk) begin
tloop656delay[i671] <= tloop656delay[i671-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop670 = tloop656delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg673[/*idx488=*/ 13:0] = '{default:0};
always@(*) shiftreg673[0] <= idx11;
always@(posedge clk) shiftreg673[/*idx488=*/ 13:1] <= shiftreg673[/*idx488=*/ 12:0];
wire [31:0] v672 = shiftreg673[/*idx488=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 13][2] = tloop11delay[13];
assign v0_addr_input[/*idx488=*/ 13][2] = {v672[3:0]};
wire[31:0] v674 = v0_rd_data[/*idx488=*/ 13];
assign v0_rd_en_input[/*idx488=*/ 13][2] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v675 = /*idx488=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg677[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg677[0] <= v674;
always@(posedge clk) shiftreg677[/*idx13=*/ 2:1] <= shiftreg677[/*idx13=*/ 1:0];
wire [31:0] v676 = shiftreg677[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v678 = v1_rd_data[/*idx488=*/ 13][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 13][/*idx13=*/ 2][0] = tloop656delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v679;
mult mult680(v679,
v676,
v678,
tloop656delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v681 = v487[/*idx488=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v682;
add add683(v682,
v679,
v681,
tloop656delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[14] = v682;

//TerminatorOp

//} Unrolled body 13 of loop488.
//DEBUG: /*idx488=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop488.
//DEBUG: /*idx488=*/ 4'd14, expected 14
//printTimeOffset
reg tloop670delay[3:0] = '{default:0} ;
always@(*) tloop670delay[0] <= tloop670;
generate
genvar i685;

for(i685 = 1; i685<= 3; i685= i685 + 1) begin
always@(posedge clk) begin
tloop670delay[i685] <= tloop670delay[i685-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop684 = tloop670delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg687[/*idx488=*/ 14:0] = '{default:0};
always@(*) shiftreg687[0] <= idx11;
always@(posedge clk) shiftreg687[/*idx488=*/ 14:1] <= shiftreg687[/*idx488=*/ 13:0];
wire [31:0] v686 = shiftreg687[/*idx488=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 14][2] = tloop11delay[14];
assign v0_addr_input[/*idx488=*/ 14][2] = {v686[3:0]};
wire[31:0] v688 = v0_rd_data[/*idx488=*/ 14];
assign v0_rd_en_input[/*idx488=*/ 14][2] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v689 = /*idx488=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg691[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg691[0] <= v688;
always@(posedge clk) shiftreg691[/*idx13=*/ 2:1] <= shiftreg691[/*idx13=*/ 1:0];
wire [31:0] v690 = shiftreg691[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v692 = v1_rd_data[/*idx488=*/ 14][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 14][/*idx13=*/ 2][0] = tloop670delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v693;
mult mult694(v693,
v690,
v692,
tloop670delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v695 = v487[/*idx488=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v696;
add add697(v696,
v693,
v695,
tloop670delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[15] = v696;

//TerminatorOp

//} Unrolled body 14 of loop488.
//DEBUG: /*idx488=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop488.
//DEBUG: /*idx488=*/ 4'd15, expected 15
//printTimeOffset
reg tloop684delay[3:0] = '{default:0} ;
always@(*) tloop684delay[0] <= tloop684;
generate
genvar i699;

for(i699 = 1; i699<= 3; i699= i699 + 1) begin
always@(posedge clk) begin
tloop684delay[i699] <= tloop684delay[i699-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop698 = tloop684delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg701[/*idx488=*/ 15:0] = '{default:0};
always@(*) shiftreg701[0] <= idx11;
always@(posedge clk) shiftreg701[/*idx488=*/ 15:1] <= shiftreg701[/*idx488=*/ 14:0];
wire [31:0] v700 = shiftreg701[/*idx488=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx488=*/ 15][2] = tloop11delay[15];
assign v0_addr_input[/*idx488=*/ 15][2] = {v700[3:0]};
wire[31:0] v702 = v0_rd_data[/*idx488=*/ 15];
assign v0_rd_en_input[/*idx488=*/ 15][2] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v703 = /*idx488=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg705[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg705[0] <= v702;
always@(posedge clk) shiftreg705[/*idx13=*/ 2:1] <= shiftreg705[/*idx13=*/ 1:0];
wire [31:0] v704 = shiftreg705[/*idx13=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v706 = v1_rd_data[/*idx488=*/ 15][/*idx13=*/ 2];
assign v1_rd_en_input[/*idx488=*/ 15][/*idx13=*/ 2][0] = tloop684delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v707;
mult mult708(v707,
v704,
v706,
tloop684delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v709 = v487[/*idx488=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v710;
add add711(v710,
v707,
v709,
tloop684delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v487[16] = v710;

//TerminatorOp

//} Unrolled body 15 of loop488.
//DEBUG: /*idx488=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t712;
assign t712 = tloop698;
//printTimeOffset
reg t712delay[3:0] = '{default:0} ;
always@(*) t712delay[0] <= t712;
generate
genvar i713;

for(i713 = 1; i713<= 3; i713= i713 + 1) begin
always@(posedge clk) begin
t712delay[i713] <= t712delay[i713-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v714 = v487[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg716[/*idx13=*/ 2:0] = '{default:0};
always@(*) shiftreg716[0] <= idx11;
always@(posedge clk) shiftreg716[/*idx13=*/ 2:1] <= shiftreg716[/*idx13=*/ 1:0];
wire [31:0] v715 = shiftreg716[/*idx13=*/ 2];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg718[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg718[0] <= v715;
always@(posedge clk) shiftreg718[/*v10=*/ 16:1] <= shiftreg718[/*v10=*/ 15:0];
wire [31:0] v717 = shiftreg718[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg720[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg720[0] <= v717;
always@(posedge clk) shiftreg720[/*v8=*/ 3:1] <= shiftreg720[/*v8=*/ 2:0];
wire [31:0] v719 = shiftreg720[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 2][0] = t712delay[3];
assign v2_addr_input[/*idx13=*/ 2][0] = {v719[3:0]};
assign v2_wr_en_input[/*idx13=*/ 2][0] = t712delay[3];
assign v2_wr_data_valid[/*idx13=*/ 2][0] = t712delay[3];
assign v2_wr_data_input[/*idx13=*/ 2][0] = v714;


//TerminatorOp

//} Unrolled body 2 of loop13.
//DEBUG: /*idx13=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop13.
//DEBUG: /*idx13=*/ 2'd3, expected 3
//printTimeOffset
reg tloop485delay[3:0] = '{default:0} ;
always@(*) tloop485delay[0] <= tloop485;
generate
genvar i722;

for(i722 = 1; i722<= 3; i722= i722 + 1) begin
always@(posedge clk) begin
tloop485delay[i722] <= tloop485delay[i722-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop721 = tloop485delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v723[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v723[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop724.
//DEBUG: /*idx724=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop725 = tloop485delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg727[/*idx724=*/ 0:0] = '{default:0};
always@(*) shiftreg727[0] <= idx11;
wire [31:0] v726 = shiftreg727[/*idx724=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 0][3] = tloop11delay[0];
assign v0_addr_input[/*idx724=*/ 0][3] = {v726[3:0]};
wire[31:0] v728 = v0_rd_data[/*idx724=*/ 0];
assign v0_rd_en_input[/*idx724=*/ 0][3] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v729 = /*idx724=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg731[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg731[0] <= v728;
always@(posedge clk) shiftreg731[/*idx13=*/ 3:1] <= shiftreg731[/*idx13=*/ 2:0];
wire [31:0] v730 = shiftreg731[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v732 = v1_rd_data[/*idx724=*/ 0][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 0][/*idx13=*/ 3][0] = tloop485delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v733;
mult mult734(v733,
v730,
v732,
tloop485delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v735 = v723[/*idx724=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v736;
add add737(v736,
v733,
v735,
tloop485delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[1] = v736;

//TerminatorOp

//} Unrolled body 0 of loop724.
//DEBUG: /*idx724=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop724.
//DEBUG: /*idx724=*/ 1'd1, expected 1
//printTimeOffset
reg tloop725delay[3:0] = '{default:0} ;
always@(*) tloop725delay[0] <= tloop725;
generate
genvar i739;

for(i739 = 1; i739<= 3; i739= i739 + 1) begin
always@(posedge clk) begin
tloop725delay[i739] <= tloop725delay[i739-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop738 = tloop725delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg741[/*idx724=*/ 1:0] = '{default:0};
always@(*) shiftreg741[0] <= idx11;
always@(posedge clk) shiftreg741[/*idx724=*/ 1:1] <= shiftreg741[/*idx724=*/ 0:0];
wire [31:0] v740 = shiftreg741[/*idx724=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 1][3] = tloop11delay[1];
assign v0_addr_input[/*idx724=*/ 1][3] = {v740[3:0]};
wire[31:0] v742 = v0_rd_data[/*idx724=*/ 1];
assign v0_rd_en_input[/*idx724=*/ 1][3] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v743 = /*idx724=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg745[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg745[0] <= v742;
always@(posedge clk) shiftreg745[/*idx13=*/ 3:1] <= shiftreg745[/*idx13=*/ 2:0];
wire [31:0] v744 = shiftreg745[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v746 = v1_rd_data[/*idx724=*/ 1][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 1][/*idx13=*/ 3][0] = tloop725delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v747;
mult mult748(v747,
v744,
v746,
tloop725delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v749 = v723[/*idx724=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v750;
add add751(v750,
v747,
v749,
tloop725delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[2] = v750;

//TerminatorOp

//} Unrolled body 1 of loop724.
//DEBUG: /*idx724=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop724.
//DEBUG: /*idx724=*/ 2'd2, expected 2
//printTimeOffset
reg tloop738delay[3:0] = '{default:0} ;
always@(*) tloop738delay[0] <= tloop738;
generate
genvar i753;

for(i753 = 1; i753<= 3; i753= i753 + 1) begin
always@(posedge clk) begin
tloop738delay[i753] <= tloop738delay[i753-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop752 = tloop738delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg755[/*idx724=*/ 2:0] = '{default:0};
always@(*) shiftreg755[0] <= idx11;
always@(posedge clk) shiftreg755[/*idx724=*/ 2:1] <= shiftreg755[/*idx724=*/ 1:0];
wire [31:0] v754 = shiftreg755[/*idx724=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 2][3] = tloop11delay[2];
assign v0_addr_input[/*idx724=*/ 2][3] = {v754[3:0]};
wire[31:0] v756 = v0_rd_data[/*idx724=*/ 2];
assign v0_rd_en_input[/*idx724=*/ 2][3] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v757 = /*idx724=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg759[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg759[0] <= v756;
always@(posedge clk) shiftreg759[/*idx13=*/ 3:1] <= shiftreg759[/*idx13=*/ 2:0];
wire [31:0] v758 = shiftreg759[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v760 = v1_rd_data[/*idx724=*/ 2][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 2][/*idx13=*/ 3][0] = tloop738delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v761;
mult mult762(v761,
v758,
v760,
tloop738delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v763 = v723[/*idx724=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v764;
add add765(v764,
v761,
v763,
tloop738delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[3] = v764;

//TerminatorOp

//} Unrolled body 2 of loop724.
//DEBUG: /*idx724=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop724.
//DEBUG: /*idx724=*/ 2'd3, expected 3
//printTimeOffset
reg tloop752delay[3:0] = '{default:0} ;
always@(*) tloop752delay[0] <= tloop752;
generate
genvar i767;

for(i767 = 1; i767<= 3; i767= i767 + 1) begin
always@(posedge clk) begin
tloop752delay[i767] <= tloop752delay[i767-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop766 = tloop752delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg769[/*idx724=*/ 3:0] = '{default:0};
always@(*) shiftreg769[0] <= idx11;
always@(posedge clk) shiftreg769[/*idx724=*/ 3:1] <= shiftreg769[/*idx724=*/ 2:0];
wire [31:0] v768 = shiftreg769[/*idx724=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 3][3] = tloop11delay[3];
assign v0_addr_input[/*idx724=*/ 3][3] = {v768[3:0]};
wire[31:0] v770 = v0_rd_data[/*idx724=*/ 3];
assign v0_rd_en_input[/*idx724=*/ 3][3] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v771 = /*idx724=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg773[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg773[0] <= v770;
always@(posedge clk) shiftreg773[/*idx13=*/ 3:1] <= shiftreg773[/*idx13=*/ 2:0];
wire [31:0] v772 = shiftreg773[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v774 = v1_rd_data[/*idx724=*/ 3][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 3][/*idx13=*/ 3][0] = tloop752delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v775;
mult mult776(v775,
v772,
v774,
tloop752delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v777 = v723[/*idx724=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v778;
add add779(v778,
v775,
v777,
tloop752delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[4] = v778;

//TerminatorOp

//} Unrolled body 3 of loop724.
//DEBUG: /*idx724=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop724.
//DEBUG: /*idx724=*/ 3'd4, expected 4
//printTimeOffset
reg tloop766delay[3:0] = '{default:0} ;
always@(*) tloop766delay[0] <= tloop766;
generate
genvar i781;

for(i781 = 1; i781<= 3; i781= i781 + 1) begin
always@(posedge clk) begin
tloop766delay[i781] <= tloop766delay[i781-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop780 = tloop766delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg783[/*idx724=*/ 4:0] = '{default:0};
always@(*) shiftreg783[0] <= idx11;
always@(posedge clk) shiftreg783[/*idx724=*/ 4:1] <= shiftreg783[/*idx724=*/ 3:0];
wire [31:0] v782 = shiftreg783[/*idx724=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 4][3] = tloop11delay[4];
assign v0_addr_input[/*idx724=*/ 4][3] = {v782[3:0]};
wire[31:0] v784 = v0_rd_data[/*idx724=*/ 4];
assign v0_rd_en_input[/*idx724=*/ 4][3] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v785 = /*idx724=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg787[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg787[0] <= v784;
always@(posedge clk) shiftreg787[/*idx13=*/ 3:1] <= shiftreg787[/*idx13=*/ 2:0];
wire [31:0] v786 = shiftreg787[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v788 = v1_rd_data[/*idx724=*/ 4][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 4][/*idx13=*/ 3][0] = tloop766delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v789;
mult mult790(v789,
v786,
v788,
tloop766delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v791 = v723[/*idx724=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v792;
add add793(v792,
v789,
v791,
tloop766delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[5] = v792;

//TerminatorOp

//} Unrolled body 4 of loop724.
//DEBUG: /*idx724=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop724.
//DEBUG: /*idx724=*/ 3'd5, expected 5
//printTimeOffset
reg tloop780delay[3:0] = '{default:0} ;
always@(*) tloop780delay[0] <= tloop780;
generate
genvar i795;

for(i795 = 1; i795<= 3; i795= i795 + 1) begin
always@(posedge clk) begin
tloop780delay[i795] <= tloop780delay[i795-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop794 = tloop780delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg797[/*idx724=*/ 5:0] = '{default:0};
always@(*) shiftreg797[0] <= idx11;
always@(posedge clk) shiftreg797[/*idx724=*/ 5:1] <= shiftreg797[/*idx724=*/ 4:0];
wire [31:0] v796 = shiftreg797[/*idx724=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 5][3] = tloop11delay[5];
assign v0_addr_input[/*idx724=*/ 5][3] = {v796[3:0]};
wire[31:0] v798 = v0_rd_data[/*idx724=*/ 5];
assign v0_rd_en_input[/*idx724=*/ 5][3] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v799 = /*idx724=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg801[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg801[0] <= v798;
always@(posedge clk) shiftreg801[/*idx13=*/ 3:1] <= shiftreg801[/*idx13=*/ 2:0];
wire [31:0] v800 = shiftreg801[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v802 = v1_rd_data[/*idx724=*/ 5][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 5][/*idx13=*/ 3][0] = tloop780delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v803;
mult mult804(v803,
v800,
v802,
tloop780delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v805 = v723[/*idx724=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v806;
add add807(v806,
v803,
v805,
tloop780delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[6] = v806;

//TerminatorOp

//} Unrolled body 5 of loop724.
//DEBUG: /*idx724=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop724.
//DEBUG: /*idx724=*/ 3'd6, expected 6
//printTimeOffset
reg tloop794delay[3:0] = '{default:0} ;
always@(*) tloop794delay[0] <= tloop794;
generate
genvar i809;

for(i809 = 1; i809<= 3; i809= i809 + 1) begin
always@(posedge clk) begin
tloop794delay[i809] <= tloop794delay[i809-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop808 = tloop794delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg811[/*idx724=*/ 6:0] = '{default:0};
always@(*) shiftreg811[0] <= idx11;
always@(posedge clk) shiftreg811[/*idx724=*/ 6:1] <= shiftreg811[/*idx724=*/ 5:0];
wire [31:0] v810 = shiftreg811[/*idx724=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 6][3] = tloop11delay[6];
assign v0_addr_input[/*idx724=*/ 6][3] = {v810[3:0]};
wire[31:0] v812 = v0_rd_data[/*idx724=*/ 6];
assign v0_rd_en_input[/*idx724=*/ 6][3] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v813 = /*idx724=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg815[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg815[0] <= v812;
always@(posedge clk) shiftreg815[/*idx13=*/ 3:1] <= shiftreg815[/*idx13=*/ 2:0];
wire [31:0] v814 = shiftreg815[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v816 = v1_rd_data[/*idx724=*/ 6][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 6][/*idx13=*/ 3][0] = tloop794delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v817;
mult mult818(v817,
v814,
v816,
tloop794delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v819 = v723[/*idx724=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v820;
add add821(v820,
v817,
v819,
tloop794delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[7] = v820;

//TerminatorOp

//} Unrolled body 6 of loop724.
//DEBUG: /*idx724=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop724.
//DEBUG: /*idx724=*/ 3'd7, expected 7
//printTimeOffset
reg tloop808delay[3:0] = '{default:0} ;
always@(*) tloop808delay[0] <= tloop808;
generate
genvar i823;

for(i823 = 1; i823<= 3; i823= i823 + 1) begin
always@(posedge clk) begin
tloop808delay[i823] <= tloop808delay[i823-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop822 = tloop808delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg825[/*idx724=*/ 7:0] = '{default:0};
always@(*) shiftreg825[0] <= idx11;
always@(posedge clk) shiftreg825[/*idx724=*/ 7:1] <= shiftreg825[/*idx724=*/ 6:0];
wire [31:0] v824 = shiftreg825[/*idx724=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 7][3] = tloop11delay[7];
assign v0_addr_input[/*idx724=*/ 7][3] = {v824[3:0]};
wire[31:0] v826 = v0_rd_data[/*idx724=*/ 7];
assign v0_rd_en_input[/*idx724=*/ 7][3] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v827 = /*idx724=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg829[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg829[0] <= v826;
always@(posedge clk) shiftreg829[/*idx13=*/ 3:1] <= shiftreg829[/*idx13=*/ 2:0];
wire [31:0] v828 = shiftreg829[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v830 = v1_rd_data[/*idx724=*/ 7][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 7][/*idx13=*/ 3][0] = tloop808delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v831;
mult mult832(v831,
v828,
v830,
tloop808delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v833 = v723[/*idx724=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v834;
add add835(v834,
v831,
v833,
tloop808delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[8] = v834;

//TerminatorOp

//} Unrolled body 7 of loop724.
//DEBUG: /*idx724=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop724.
//DEBUG: /*idx724=*/ 4'd8, expected 8
//printTimeOffset
reg tloop822delay[3:0] = '{default:0} ;
always@(*) tloop822delay[0] <= tloop822;
generate
genvar i837;

for(i837 = 1; i837<= 3; i837= i837 + 1) begin
always@(posedge clk) begin
tloop822delay[i837] <= tloop822delay[i837-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop836 = tloop822delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg839[/*idx724=*/ 8:0] = '{default:0};
always@(*) shiftreg839[0] <= idx11;
always@(posedge clk) shiftreg839[/*idx724=*/ 8:1] <= shiftreg839[/*idx724=*/ 7:0];
wire [31:0] v838 = shiftreg839[/*idx724=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 8][3] = tloop11delay[8];
assign v0_addr_input[/*idx724=*/ 8][3] = {v838[3:0]};
wire[31:0] v840 = v0_rd_data[/*idx724=*/ 8];
assign v0_rd_en_input[/*idx724=*/ 8][3] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v841 = /*idx724=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg843[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg843[0] <= v840;
always@(posedge clk) shiftreg843[/*idx13=*/ 3:1] <= shiftreg843[/*idx13=*/ 2:0];
wire [31:0] v842 = shiftreg843[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v844 = v1_rd_data[/*idx724=*/ 8][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 8][/*idx13=*/ 3][0] = tloop822delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v845;
mult mult846(v845,
v842,
v844,
tloop822delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v847 = v723[/*idx724=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v848;
add add849(v848,
v845,
v847,
tloop822delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[9] = v848;

//TerminatorOp

//} Unrolled body 8 of loop724.
//DEBUG: /*idx724=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop724.
//DEBUG: /*idx724=*/ 4'd9, expected 9
//printTimeOffset
reg tloop836delay[3:0] = '{default:0} ;
always@(*) tloop836delay[0] <= tloop836;
generate
genvar i851;

for(i851 = 1; i851<= 3; i851= i851 + 1) begin
always@(posedge clk) begin
tloop836delay[i851] <= tloop836delay[i851-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop850 = tloop836delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg853[/*idx724=*/ 9:0] = '{default:0};
always@(*) shiftreg853[0] <= idx11;
always@(posedge clk) shiftreg853[/*idx724=*/ 9:1] <= shiftreg853[/*idx724=*/ 8:0];
wire [31:0] v852 = shiftreg853[/*idx724=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 9][3] = tloop11delay[9];
assign v0_addr_input[/*idx724=*/ 9][3] = {v852[3:0]};
wire[31:0] v854 = v0_rd_data[/*idx724=*/ 9];
assign v0_rd_en_input[/*idx724=*/ 9][3] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v855 = /*idx724=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg857[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg857[0] <= v854;
always@(posedge clk) shiftreg857[/*idx13=*/ 3:1] <= shiftreg857[/*idx13=*/ 2:0];
wire [31:0] v856 = shiftreg857[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v858 = v1_rd_data[/*idx724=*/ 9][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 9][/*idx13=*/ 3][0] = tloop836delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v859;
mult mult860(v859,
v856,
v858,
tloop836delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v861 = v723[/*idx724=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v862;
add add863(v862,
v859,
v861,
tloop836delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[10] = v862;

//TerminatorOp

//} Unrolled body 9 of loop724.
//DEBUG: /*idx724=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop724.
//DEBUG: /*idx724=*/ 4'd10, expected 10
//printTimeOffset
reg tloop850delay[3:0] = '{default:0} ;
always@(*) tloop850delay[0] <= tloop850;
generate
genvar i865;

for(i865 = 1; i865<= 3; i865= i865 + 1) begin
always@(posedge clk) begin
tloop850delay[i865] <= tloop850delay[i865-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop864 = tloop850delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg867[/*idx724=*/ 10:0] = '{default:0};
always@(*) shiftreg867[0] <= idx11;
always@(posedge clk) shiftreg867[/*idx724=*/ 10:1] <= shiftreg867[/*idx724=*/ 9:0];
wire [31:0] v866 = shiftreg867[/*idx724=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 10][3] = tloop11delay[10];
assign v0_addr_input[/*idx724=*/ 10][3] = {v866[3:0]};
wire[31:0] v868 = v0_rd_data[/*idx724=*/ 10];
assign v0_rd_en_input[/*idx724=*/ 10][3] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v869 = /*idx724=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg871[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg871[0] <= v868;
always@(posedge clk) shiftreg871[/*idx13=*/ 3:1] <= shiftreg871[/*idx13=*/ 2:0];
wire [31:0] v870 = shiftreg871[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v872 = v1_rd_data[/*idx724=*/ 10][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 10][/*idx13=*/ 3][0] = tloop850delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v873;
mult mult874(v873,
v870,
v872,
tloop850delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v875 = v723[/*idx724=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v876;
add add877(v876,
v873,
v875,
tloop850delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[11] = v876;

//TerminatorOp

//} Unrolled body 10 of loop724.
//DEBUG: /*idx724=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop724.
//DEBUG: /*idx724=*/ 4'd11, expected 11
//printTimeOffset
reg tloop864delay[3:0] = '{default:0} ;
always@(*) tloop864delay[0] <= tloop864;
generate
genvar i879;

for(i879 = 1; i879<= 3; i879= i879 + 1) begin
always@(posedge clk) begin
tloop864delay[i879] <= tloop864delay[i879-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop878 = tloop864delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg881[/*idx724=*/ 11:0] = '{default:0};
always@(*) shiftreg881[0] <= idx11;
always@(posedge clk) shiftreg881[/*idx724=*/ 11:1] <= shiftreg881[/*idx724=*/ 10:0];
wire [31:0] v880 = shiftreg881[/*idx724=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 11][3] = tloop11delay[11];
assign v0_addr_input[/*idx724=*/ 11][3] = {v880[3:0]};
wire[31:0] v882 = v0_rd_data[/*idx724=*/ 11];
assign v0_rd_en_input[/*idx724=*/ 11][3] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v883 = /*idx724=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg885[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg885[0] <= v882;
always@(posedge clk) shiftreg885[/*idx13=*/ 3:1] <= shiftreg885[/*idx13=*/ 2:0];
wire [31:0] v884 = shiftreg885[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v886 = v1_rd_data[/*idx724=*/ 11][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 11][/*idx13=*/ 3][0] = tloop864delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v887;
mult mult888(v887,
v884,
v886,
tloop864delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v889 = v723[/*idx724=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v890;
add add891(v890,
v887,
v889,
tloop864delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[12] = v890;

//TerminatorOp

//} Unrolled body 11 of loop724.
//DEBUG: /*idx724=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop724.
//DEBUG: /*idx724=*/ 4'd12, expected 12
//printTimeOffset
reg tloop878delay[3:0] = '{default:0} ;
always@(*) tloop878delay[0] <= tloop878;
generate
genvar i893;

for(i893 = 1; i893<= 3; i893= i893 + 1) begin
always@(posedge clk) begin
tloop878delay[i893] <= tloop878delay[i893-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop892 = tloop878delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg895[/*idx724=*/ 12:0] = '{default:0};
always@(*) shiftreg895[0] <= idx11;
always@(posedge clk) shiftreg895[/*idx724=*/ 12:1] <= shiftreg895[/*idx724=*/ 11:0];
wire [31:0] v894 = shiftreg895[/*idx724=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 12][3] = tloop11delay[12];
assign v0_addr_input[/*idx724=*/ 12][3] = {v894[3:0]};
wire[31:0] v896 = v0_rd_data[/*idx724=*/ 12];
assign v0_rd_en_input[/*idx724=*/ 12][3] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v897 = /*idx724=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg899[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg899[0] <= v896;
always@(posedge clk) shiftreg899[/*idx13=*/ 3:1] <= shiftreg899[/*idx13=*/ 2:0];
wire [31:0] v898 = shiftreg899[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v900 = v1_rd_data[/*idx724=*/ 12][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 12][/*idx13=*/ 3][0] = tloop878delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v901;
mult mult902(v901,
v898,
v900,
tloop878delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v903 = v723[/*idx724=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v904;
add add905(v904,
v901,
v903,
tloop878delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[13] = v904;

//TerminatorOp

//} Unrolled body 12 of loop724.
//DEBUG: /*idx724=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop724.
//DEBUG: /*idx724=*/ 4'd13, expected 13
//printTimeOffset
reg tloop892delay[3:0] = '{default:0} ;
always@(*) tloop892delay[0] <= tloop892;
generate
genvar i907;

for(i907 = 1; i907<= 3; i907= i907 + 1) begin
always@(posedge clk) begin
tloop892delay[i907] <= tloop892delay[i907-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop906 = tloop892delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg909[/*idx724=*/ 13:0] = '{default:0};
always@(*) shiftreg909[0] <= idx11;
always@(posedge clk) shiftreg909[/*idx724=*/ 13:1] <= shiftreg909[/*idx724=*/ 12:0];
wire [31:0] v908 = shiftreg909[/*idx724=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 13][3] = tloop11delay[13];
assign v0_addr_input[/*idx724=*/ 13][3] = {v908[3:0]};
wire[31:0] v910 = v0_rd_data[/*idx724=*/ 13];
assign v0_rd_en_input[/*idx724=*/ 13][3] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v911 = /*idx724=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg913[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg913[0] <= v910;
always@(posedge clk) shiftreg913[/*idx13=*/ 3:1] <= shiftreg913[/*idx13=*/ 2:0];
wire [31:0] v912 = shiftreg913[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v914 = v1_rd_data[/*idx724=*/ 13][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 13][/*idx13=*/ 3][0] = tloop892delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v915;
mult mult916(v915,
v912,
v914,
tloop892delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v917 = v723[/*idx724=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v918;
add add919(v918,
v915,
v917,
tloop892delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[14] = v918;

//TerminatorOp

//} Unrolled body 13 of loop724.
//DEBUG: /*idx724=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop724.
//DEBUG: /*idx724=*/ 4'd14, expected 14
//printTimeOffset
reg tloop906delay[3:0] = '{default:0} ;
always@(*) tloop906delay[0] <= tloop906;
generate
genvar i921;

for(i921 = 1; i921<= 3; i921= i921 + 1) begin
always@(posedge clk) begin
tloop906delay[i921] <= tloop906delay[i921-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop920 = tloop906delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg923[/*idx724=*/ 14:0] = '{default:0};
always@(*) shiftreg923[0] <= idx11;
always@(posedge clk) shiftreg923[/*idx724=*/ 14:1] <= shiftreg923[/*idx724=*/ 13:0];
wire [31:0] v922 = shiftreg923[/*idx724=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 14][3] = tloop11delay[14];
assign v0_addr_input[/*idx724=*/ 14][3] = {v922[3:0]};
wire[31:0] v924 = v0_rd_data[/*idx724=*/ 14];
assign v0_rd_en_input[/*idx724=*/ 14][3] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v925 = /*idx724=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg927[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg927[0] <= v924;
always@(posedge clk) shiftreg927[/*idx13=*/ 3:1] <= shiftreg927[/*idx13=*/ 2:0];
wire [31:0] v926 = shiftreg927[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v928 = v1_rd_data[/*idx724=*/ 14][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 14][/*idx13=*/ 3][0] = tloop906delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v929;
mult mult930(v929,
v926,
v928,
tloop906delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v931 = v723[/*idx724=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v932;
add add933(v932,
v929,
v931,
tloop906delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[15] = v932;

//TerminatorOp

//} Unrolled body 14 of loop724.
//DEBUG: /*idx724=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop724.
//DEBUG: /*idx724=*/ 4'd15, expected 15
//printTimeOffset
reg tloop920delay[3:0] = '{default:0} ;
always@(*) tloop920delay[0] <= tloop920;
generate
genvar i935;

for(i935 = 1; i935<= 3; i935= i935 + 1) begin
always@(posedge clk) begin
tloop920delay[i935] <= tloop920delay[i935-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop934 = tloop920delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg937[/*idx724=*/ 15:0] = '{default:0};
always@(*) shiftreg937[0] <= idx11;
always@(posedge clk) shiftreg937[/*idx724=*/ 15:1] <= shiftreg937[/*idx724=*/ 14:0];
wire [31:0] v936 = shiftreg937[/*idx724=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx724=*/ 15][3] = tloop11delay[15];
assign v0_addr_input[/*idx724=*/ 15][3] = {v936[3:0]};
wire[31:0] v938 = v0_rd_data[/*idx724=*/ 15];
assign v0_rd_en_input[/*idx724=*/ 15][3] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v939 = /*idx724=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg941[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg941[0] <= v938;
always@(posedge clk) shiftreg941[/*idx13=*/ 3:1] <= shiftreg941[/*idx13=*/ 2:0];
wire [31:0] v940 = shiftreg941[/*idx13=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v942 = v1_rd_data[/*idx724=*/ 15][/*idx13=*/ 3];
assign v1_rd_en_input[/*idx724=*/ 15][/*idx13=*/ 3][0] = tloop920delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v943;
mult mult944(v943,
v940,
v942,
tloop920delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v945 = v723[/*idx724=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v946;
add add947(v946,
v943,
v945,
tloop920delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v723[16] = v946;

//TerminatorOp

//} Unrolled body 15 of loop724.
//DEBUG: /*idx724=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t948;
assign t948 = tloop934;
//printTimeOffset
reg t948delay[3:0] = '{default:0} ;
always@(*) t948delay[0] <= t948;
generate
genvar i949;

for(i949 = 1; i949<= 3; i949= i949 + 1) begin
always@(posedge clk) begin
t948delay[i949] <= t948delay[i949-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v950 = v723[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg952[/*idx13=*/ 3:0] = '{default:0};
always@(*) shiftreg952[0] <= idx11;
always@(posedge clk) shiftreg952[/*idx13=*/ 3:1] <= shiftreg952[/*idx13=*/ 2:0];
wire [31:0] v951 = shiftreg952[/*idx13=*/ 3];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg954[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg954[0] <= v951;
always@(posedge clk) shiftreg954[/*v10=*/ 16:1] <= shiftreg954[/*v10=*/ 15:0];
wire [31:0] v953 = shiftreg954[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg956[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg956[0] <= v953;
always@(posedge clk) shiftreg956[/*v8=*/ 3:1] <= shiftreg956[/*v8=*/ 2:0];
wire [31:0] v955 = shiftreg956[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 3][0] = t948delay[3];
assign v2_addr_input[/*idx13=*/ 3][0] = {v955[3:0]};
assign v2_wr_en_input[/*idx13=*/ 3][0] = t948delay[3];
assign v2_wr_data_valid[/*idx13=*/ 3][0] = t948delay[3];
assign v2_wr_data_input[/*idx13=*/ 3][0] = v950;


//TerminatorOp

//} Unrolled body 3 of loop13.
//DEBUG: /*idx13=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop13.
//DEBUG: /*idx13=*/ 3'd4, expected 4
//printTimeOffset
reg tloop721delay[3:0] = '{default:0} ;
always@(*) tloop721delay[0] <= tloop721;
generate
genvar i958;

for(i958 = 1; i958<= 3; i958= i958 + 1) begin
always@(posedge clk) begin
tloop721delay[i958] <= tloop721delay[i958-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop957 = tloop721delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v959[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v959[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop960.
//DEBUG: /*idx960=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop961 = tloop721delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg963[/*idx960=*/ 0:0] = '{default:0};
always@(*) shiftreg963[0] <= idx11;
wire [31:0] v962 = shiftreg963[/*idx960=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 0][4] = tloop11delay[0];
assign v0_addr_input[/*idx960=*/ 0][4] = {v962[3:0]};
wire[31:0] v964 = v0_rd_data[/*idx960=*/ 0];
assign v0_rd_en_input[/*idx960=*/ 0][4] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v965 = /*idx960=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg967[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg967[0] <= v964;
always@(posedge clk) shiftreg967[/*idx13=*/ 4:1] <= shiftreg967[/*idx13=*/ 3:0];
wire [31:0] v966 = shiftreg967[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v968 = v1_rd_data[/*idx960=*/ 0][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 0][/*idx13=*/ 4][0] = tloop721delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v969;
mult mult970(v969,
v966,
v968,
tloop721delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v971 = v959[/*idx960=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v972;
add add973(v972,
v969,
v971,
tloop721delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[1] = v972;

//TerminatorOp

//} Unrolled body 0 of loop960.
//DEBUG: /*idx960=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop960.
//DEBUG: /*idx960=*/ 1'd1, expected 1
//printTimeOffset
reg tloop961delay[3:0] = '{default:0} ;
always@(*) tloop961delay[0] <= tloop961;
generate
genvar i975;

for(i975 = 1; i975<= 3; i975= i975 + 1) begin
always@(posedge clk) begin
tloop961delay[i975] <= tloop961delay[i975-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop974 = tloop961delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg977[/*idx960=*/ 1:0] = '{default:0};
always@(*) shiftreg977[0] <= idx11;
always@(posedge clk) shiftreg977[/*idx960=*/ 1:1] <= shiftreg977[/*idx960=*/ 0:0];
wire [31:0] v976 = shiftreg977[/*idx960=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 1][4] = tloop11delay[1];
assign v0_addr_input[/*idx960=*/ 1][4] = {v976[3:0]};
wire[31:0] v978 = v0_rd_data[/*idx960=*/ 1];
assign v0_rd_en_input[/*idx960=*/ 1][4] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v979 = /*idx960=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg981[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg981[0] <= v978;
always@(posedge clk) shiftreg981[/*idx13=*/ 4:1] <= shiftreg981[/*idx13=*/ 3:0];
wire [31:0] v980 = shiftreg981[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v982 = v1_rd_data[/*idx960=*/ 1][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 1][/*idx13=*/ 4][0] = tloop961delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v983;
mult mult984(v983,
v980,
v982,
tloop961delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v985 = v959[/*idx960=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v986;
add add987(v986,
v983,
v985,
tloop961delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[2] = v986;

//TerminatorOp

//} Unrolled body 1 of loop960.
//DEBUG: /*idx960=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop960.
//DEBUG: /*idx960=*/ 2'd2, expected 2
//printTimeOffset
reg tloop974delay[3:0] = '{default:0} ;
always@(*) tloop974delay[0] <= tloop974;
generate
genvar i989;

for(i989 = 1; i989<= 3; i989= i989 + 1) begin
always@(posedge clk) begin
tloop974delay[i989] <= tloop974delay[i989-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop988 = tloop974delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg991[/*idx960=*/ 2:0] = '{default:0};
always@(*) shiftreg991[0] <= idx11;
always@(posedge clk) shiftreg991[/*idx960=*/ 2:1] <= shiftreg991[/*idx960=*/ 1:0];
wire [31:0] v990 = shiftreg991[/*idx960=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 2][4] = tloop11delay[2];
assign v0_addr_input[/*idx960=*/ 2][4] = {v990[3:0]};
wire[31:0] v992 = v0_rd_data[/*idx960=*/ 2];
assign v0_rd_en_input[/*idx960=*/ 2][4] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v993 = /*idx960=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg995[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg995[0] <= v992;
always@(posedge clk) shiftreg995[/*idx13=*/ 4:1] <= shiftreg995[/*idx13=*/ 3:0];
wire [31:0] v994 = shiftreg995[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v996 = v1_rd_data[/*idx960=*/ 2][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 2][/*idx13=*/ 4][0] = tloop974delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v997;
mult mult998(v997,
v994,
v996,
tloop974delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v999 = v959[/*idx960=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1000;
add add1001(v1000,
v997,
v999,
tloop974delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[3] = v1000;

//TerminatorOp

//} Unrolled body 2 of loop960.
//DEBUG: /*idx960=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop960.
//DEBUG: /*idx960=*/ 2'd3, expected 3
//printTimeOffset
reg tloop988delay[3:0] = '{default:0} ;
always@(*) tloop988delay[0] <= tloop988;
generate
genvar i1003;

for(i1003 = 1; i1003<= 3; i1003= i1003 + 1) begin
always@(posedge clk) begin
tloop988delay[i1003] <= tloop988delay[i1003-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1002 = tloop988delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1005[/*idx960=*/ 3:0] = '{default:0};
always@(*) shiftreg1005[0] <= idx11;
always@(posedge clk) shiftreg1005[/*idx960=*/ 3:1] <= shiftreg1005[/*idx960=*/ 2:0];
wire [31:0] v1004 = shiftreg1005[/*idx960=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 3][4] = tloop11delay[3];
assign v0_addr_input[/*idx960=*/ 3][4] = {v1004[3:0]};
wire[31:0] v1006 = v0_rd_data[/*idx960=*/ 3];
assign v0_rd_en_input[/*idx960=*/ 3][4] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1007 = /*idx960=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1009[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1009[0] <= v1006;
always@(posedge clk) shiftreg1009[/*idx13=*/ 4:1] <= shiftreg1009[/*idx13=*/ 3:0];
wire [31:0] v1008 = shiftreg1009[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1010 = v1_rd_data[/*idx960=*/ 3][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 3][/*idx13=*/ 4][0] = tloop988delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1011;
mult mult1012(v1011,
v1008,
v1010,
tloop988delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1013 = v959[/*idx960=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1014;
add add1015(v1014,
v1011,
v1013,
tloop988delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[4] = v1014;

//TerminatorOp

//} Unrolled body 3 of loop960.
//DEBUG: /*idx960=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop960.
//DEBUG: /*idx960=*/ 3'd4, expected 4
//printTimeOffset
reg tloop1002delay[3:0] = '{default:0} ;
always@(*) tloop1002delay[0] <= tloop1002;
generate
genvar i1017;

for(i1017 = 1; i1017<= 3; i1017= i1017 + 1) begin
always@(posedge clk) begin
tloop1002delay[i1017] <= tloop1002delay[i1017-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1016 = tloop1002delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1019[/*idx960=*/ 4:0] = '{default:0};
always@(*) shiftreg1019[0] <= idx11;
always@(posedge clk) shiftreg1019[/*idx960=*/ 4:1] <= shiftreg1019[/*idx960=*/ 3:0];
wire [31:0] v1018 = shiftreg1019[/*idx960=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 4][4] = tloop11delay[4];
assign v0_addr_input[/*idx960=*/ 4][4] = {v1018[3:0]};
wire[31:0] v1020 = v0_rd_data[/*idx960=*/ 4];
assign v0_rd_en_input[/*idx960=*/ 4][4] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1021 = /*idx960=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1023[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1023[0] <= v1020;
always@(posedge clk) shiftreg1023[/*idx13=*/ 4:1] <= shiftreg1023[/*idx13=*/ 3:0];
wire [31:0] v1022 = shiftreg1023[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1024 = v1_rd_data[/*idx960=*/ 4][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 4][/*idx13=*/ 4][0] = tloop1002delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1025;
mult mult1026(v1025,
v1022,
v1024,
tloop1002delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1027 = v959[/*idx960=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1028;
add add1029(v1028,
v1025,
v1027,
tloop1002delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[5] = v1028;

//TerminatorOp

//} Unrolled body 4 of loop960.
//DEBUG: /*idx960=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop960.
//DEBUG: /*idx960=*/ 3'd5, expected 5
//printTimeOffset
reg tloop1016delay[3:0] = '{default:0} ;
always@(*) tloop1016delay[0] <= tloop1016;
generate
genvar i1031;

for(i1031 = 1; i1031<= 3; i1031= i1031 + 1) begin
always@(posedge clk) begin
tloop1016delay[i1031] <= tloop1016delay[i1031-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1030 = tloop1016delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1033[/*idx960=*/ 5:0] = '{default:0};
always@(*) shiftreg1033[0] <= idx11;
always@(posedge clk) shiftreg1033[/*idx960=*/ 5:1] <= shiftreg1033[/*idx960=*/ 4:0];
wire [31:0] v1032 = shiftreg1033[/*idx960=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 5][4] = tloop11delay[5];
assign v0_addr_input[/*idx960=*/ 5][4] = {v1032[3:0]};
wire[31:0] v1034 = v0_rd_data[/*idx960=*/ 5];
assign v0_rd_en_input[/*idx960=*/ 5][4] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1035 = /*idx960=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1037[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1037[0] <= v1034;
always@(posedge clk) shiftreg1037[/*idx13=*/ 4:1] <= shiftreg1037[/*idx13=*/ 3:0];
wire [31:0] v1036 = shiftreg1037[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1038 = v1_rd_data[/*idx960=*/ 5][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 5][/*idx13=*/ 4][0] = tloop1016delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1039;
mult mult1040(v1039,
v1036,
v1038,
tloop1016delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1041 = v959[/*idx960=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1042;
add add1043(v1042,
v1039,
v1041,
tloop1016delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[6] = v1042;

//TerminatorOp

//} Unrolled body 5 of loop960.
//DEBUG: /*idx960=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop960.
//DEBUG: /*idx960=*/ 3'd6, expected 6
//printTimeOffset
reg tloop1030delay[3:0] = '{default:0} ;
always@(*) tloop1030delay[0] <= tloop1030;
generate
genvar i1045;

for(i1045 = 1; i1045<= 3; i1045= i1045 + 1) begin
always@(posedge clk) begin
tloop1030delay[i1045] <= tloop1030delay[i1045-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1044 = tloop1030delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1047[/*idx960=*/ 6:0] = '{default:0};
always@(*) shiftreg1047[0] <= idx11;
always@(posedge clk) shiftreg1047[/*idx960=*/ 6:1] <= shiftreg1047[/*idx960=*/ 5:0];
wire [31:0] v1046 = shiftreg1047[/*idx960=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 6][4] = tloop11delay[6];
assign v0_addr_input[/*idx960=*/ 6][4] = {v1046[3:0]};
wire[31:0] v1048 = v0_rd_data[/*idx960=*/ 6];
assign v0_rd_en_input[/*idx960=*/ 6][4] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1049 = /*idx960=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1051[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1051[0] <= v1048;
always@(posedge clk) shiftreg1051[/*idx13=*/ 4:1] <= shiftreg1051[/*idx13=*/ 3:0];
wire [31:0] v1050 = shiftreg1051[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1052 = v1_rd_data[/*idx960=*/ 6][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 6][/*idx13=*/ 4][0] = tloop1030delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1053;
mult mult1054(v1053,
v1050,
v1052,
tloop1030delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1055 = v959[/*idx960=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1056;
add add1057(v1056,
v1053,
v1055,
tloop1030delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[7] = v1056;

//TerminatorOp

//} Unrolled body 6 of loop960.
//DEBUG: /*idx960=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop960.
//DEBUG: /*idx960=*/ 3'd7, expected 7
//printTimeOffset
reg tloop1044delay[3:0] = '{default:0} ;
always@(*) tloop1044delay[0] <= tloop1044;
generate
genvar i1059;

for(i1059 = 1; i1059<= 3; i1059= i1059 + 1) begin
always@(posedge clk) begin
tloop1044delay[i1059] <= tloop1044delay[i1059-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1058 = tloop1044delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1061[/*idx960=*/ 7:0] = '{default:0};
always@(*) shiftreg1061[0] <= idx11;
always@(posedge clk) shiftreg1061[/*idx960=*/ 7:1] <= shiftreg1061[/*idx960=*/ 6:0];
wire [31:0] v1060 = shiftreg1061[/*idx960=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 7][4] = tloop11delay[7];
assign v0_addr_input[/*idx960=*/ 7][4] = {v1060[3:0]};
wire[31:0] v1062 = v0_rd_data[/*idx960=*/ 7];
assign v0_rd_en_input[/*idx960=*/ 7][4] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1063 = /*idx960=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1065[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1065[0] <= v1062;
always@(posedge clk) shiftreg1065[/*idx13=*/ 4:1] <= shiftreg1065[/*idx13=*/ 3:0];
wire [31:0] v1064 = shiftreg1065[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1066 = v1_rd_data[/*idx960=*/ 7][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 7][/*idx13=*/ 4][0] = tloop1044delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1067;
mult mult1068(v1067,
v1064,
v1066,
tloop1044delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1069 = v959[/*idx960=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1070;
add add1071(v1070,
v1067,
v1069,
tloop1044delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[8] = v1070;

//TerminatorOp

//} Unrolled body 7 of loop960.
//DEBUG: /*idx960=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop960.
//DEBUG: /*idx960=*/ 4'd8, expected 8
//printTimeOffset
reg tloop1058delay[3:0] = '{default:0} ;
always@(*) tloop1058delay[0] <= tloop1058;
generate
genvar i1073;

for(i1073 = 1; i1073<= 3; i1073= i1073 + 1) begin
always@(posedge clk) begin
tloop1058delay[i1073] <= tloop1058delay[i1073-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1072 = tloop1058delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1075[/*idx960=*/ 8:0] = '{default:0};
always@(*) shiftreg1075[0] <= idx11;
always@(posedge clk) shiftreg1075[/*idx960=*/ 8:1] <= shiftreg1075[/*idx960=*/ 7:0];
wire [31:0] v1074 = shiftreg1075[/*idx960=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 8][4] = tloop11delay[8];
assign v0_addr_input[/*idx960=*/ 8][4] = {v1074[3:0]};
wire[31:0] v1076 = v0_rd_data[/*idx960=*/ 8];
assign v0_rd_en_input[/*idx960=*/ 8][4] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1077 = /*idx960=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1079[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1079[0] <= v1076;
always@(posedge clk) shiftreg1079[/*idx13=*/ 4:1] <= shiftreg1079[/*idx13=*/ 3:0];
wire [31:0] v1078 = shiftreg1079[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1080 = v1_rd_data[/*idx960=*/ 8][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 8][/*idx13=*/ 4][0] = tloop1058delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1081;
mult mult1082(v1081,
v1078,
v1080,
tloop1058delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1083 = v959[/*idx960=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1084;
add add1085(v1084,
v1081,
v1083,
tloop1058delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[9] = v1084;

//TerminatorOp

//} Unrolled body 8 of loop960.
//DEBUG: /*idx960=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop960.
//DEBUG: /*idx960=*/ 4'd9, expected 9
//printTimeOffset
reg tloop1072delay[3:0] = '{default:0} ;
always@(*) tloop1072delay[0] <= tloop1072;
generate
genvar i1087;

for(i1087 = 1; i1087<= 3; i1087= i1087 + 1) begin
always@(posedge clk) begin
tloop1072delay[i1087] <= tloop1072delay[i1087-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1086 = tloop1072delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1089[/*idx960=*/ 9:0] = '{default:0};
always@(*) shiftreg1089[0] <= idx11;
always@(posedge clk) shiftreg1089[/*idx960=*/ 9:1] <= shiftreg1089[/*idx960=*/ 8:0];
wire [31:0] v1088 = shiftreg1089[/*idx960=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 9][4] = tloop11delay[9];
assign v0_addr_input[/*idx960=*/ 9][4] = {v1088[3:0]};
wire[31:0] v1090 = v0_rd_data[/*idx960=*/ 9];
assign v0_rd_en_input[/*idx960=*/ 9][4] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1091 = /*idx960=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1093[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1093[0] <= v1090;
always@(posedge clk) shiftreg1093[/*idx13=*/ 4:1] <= shiftreg1093[/*idx13=*/ 3:0];
wire [31:0] v1092 = shiftreg1093[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1094 = v1_rd_data[/*idx960=*/ 9][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 9][/*idx13=*/ 4][0] = tloop1072delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1095;
mult mult1096(v1095,
v1092,
v1094,
tloop1072delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1097 = v959[/*idx960=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1098;
add add1099(v1098,
v1095,
v1097,
tloop1072delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[10] = v1098;

//TerminatorOp

//} Unrolled body 9 of loop960.
//DEBUG: /*idx960=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop960.
//DEBUG: /*idx960=*/ 4'd10, expected 10
//printTimeOffset
reg tloop1086delay[3:0] = '{default:0} ;
always@(*) tloop1086delay[0] <= tloop1086;
generate
genvar i1101;

for(i1101 = 1; i1101<= 3; i1101= i1101 + 1) begin
always@(posedge clk) begin
tloop1086delay[i1101] <= tloop1086delay[i1101-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1100 = tloop1086delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1103[/*idx960=*/ 10:0] = '{default:0};
always@(*) shiftreg1103[0] <= idx11;
always@(posedge clk) shiftreg1103[/*idx960=*/ 10:1] <= shiftreg1103[/*idx960=*/ 9:0];
wire [31:0] v1102 = shiftreg1103[/*idx960=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 10][4] = tloop11delay[10];
assign v0_addr_input[/*idx960=*/ 10][4] = {v1102[3:0]};
wire[31:0] v1104 = v0_rd_data[/*idx960=*/ 10];
assign v0_rd_en_input[/*idx960=*/ 10][4] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1105 = /*idx960=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1107[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1107[0] <= v1104;
always@(posedge clk) shiftreg1107[/*idx13=*/ 4:1] <= shiftreg1107[/*idx13=*/ 3:0];
wire [31:0] v1106 = shiftreg1107[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1108 = v1_rd_data[/*idx960=*/ 10][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 10][/*idx13=*/ 4][0] = tloop1086delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1109;
mult mult1110(v1109,
v1106,
v1108,
tloop1086delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1111 = v959[/*idx960=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1112;
add add1113(v1112,
v1109,
v1111,
tloop1086delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[11] = v1112;

//TerminatorOp

//} Unrolled body 10 of loop960.
//DEBUG: /*idx960=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop960.
//DEBUG: /*idx960=*/ 4'd11, expected 11
//printTimeOffset
reg tloop1100delay[3:0] = '{default:0} ;
always@(*) tloop1100delay[0] <= tloop1100;
generate
genvar i1115;

for(i1115 = 1; i1115<= 3; i1115= i1115 + 1) begin
always@(posedge clk) begin
tloop1100delay[i1115] <= tloop1100delay[i1115-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1114 = tloop1100delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1117[/*idx960=*/ 11:0] = '{default:0};
always@(*) shiftreg1117[0] <= idx11;
always@(posedge clk) shiftreg1117[/*idx960=*/ 11:1] <= shiftreg1117[/*idx960=*/ 10:0];
wire [31:0] v1116 = shiftreg1117[/*idx960=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 11][4] = tloop11delay[11];
assign v0_addr_input[/*idx960=*/ 11][4] = {v1116[3:0]};
wire[31:0] v1118 = v0_rd_data[/*idx960=*/ 11];
assign v0_rd_en_input[/*idx960=*/ 11][4] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1119 = /*idx960=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1121[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1121[0] <= v1118;
always@(posedge clk) shiftreg1121[/*idx13=*/ 4:1] <= shiftreg1121[/*idx13=*/ 3:0];
wire [31:0] v1120 = shiftreg1121[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1122 = v1_rd_data[/*idx960=*/ 11][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 11][/*idx13=*/ 4][0] = tloop1100delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1123;
mult mult1124(v1123,
v1120,
v1122,
tloop1100delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1125 = v959[/*idx960=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1126;
add add1127(v1126,
v1123,
v1125,
tloop1100delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[12] = v1126;

//TerminatorOp

//} Unrolled body 11 of loop960.
//DEBUG: /*idx960=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop960.
//DEBUG: /*idx960=*/ 4'd12, expected 12
//printTimeOffset
reg tloop1114delay[3:0] = '{default:0} ;
always@(*) tloop1114delay[0] <= tloop1114;
generate
genvar i1129;

for(i1129 = 1; i1129<= 3; i1129= i1129 + 1) begin
always@(posedge clk) begin
tloop1114delay[i1129] <= tloop1114delay[i1129-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1128 = tloop1114delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1131[/*idx960=*/ 12:0] = '{default:0};
always@(*) shiftreg1131[0] <= idx11;
always@(posedge clk) shiftreg1131[/*idx960=*/ 12:1] <= shiftreg1131[/*idx960=*/ 11:0];
wire [31:0] v1130 = shiftreg1131[/*idx960=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 12][4] = tloop11delay[12];
assign v0_addr_input[/*idx960=*/ 12][4] = {v1130[3:0]};
wire[31:0] v1132 = v0_rd_data[/*idx960=*/ 12];
assign v0_rd_en_input[/*idx960=*/ 12][4] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1133 = /*idx960=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1135[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1135[0] <= v1132;
always@(posedge clk) shiftreg1135[/*idx13=*/ 4:1] <= shiftreg1135[/*idx13=*/ 3:0];
wire [31:0] v1134 = shiftreg1135[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1136 = v1_rd_data[/*idx960=*/ 12][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 12][/*idx13=*/ 4][0] = tloop1114delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1137;
mult mult1138(v1137,
v1134,
v1136,
tloop1114delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1139 = v959[/*idx960=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1140;
add add1141(v1140,
v1137,
v1139,
tloop1114delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[13] = v1140;

//TerminatorOp

//} Unrolled body 12 of loop960.
//DEBUG: /*idx960=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop960.
//DEBUG: /*idx960=*/ 4'd13, expected 13
//printTimeOffset
reg tloop1128delay[3:0] = '{default:0} ;
always@(*) tloop1128delay[0] <= tloop1128;
generate
genvar i1143;

for(i1143 = 1; i1143<= 3; i1143= i1143 + 1) begin
always@(posedge clk) begin
tloop1128delay[i1143] <= tloop1128delay[i1143-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1142 = tloop1128delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1145[/*idx960=*/ 13:0] = '{default:0};
always@(*) shiftreg1145[0] <= idx11;
always@(posedge clk) shiftreg1145[/*idx960=*/ 13:1] <= shiftreg1145[/*idx960=*/ 12:0];
wire [31:0] v1144 = shiftreg1145[/*idx960=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 13][4] = tloop11delay[13];
assign v0_addr_input[/*idx960=*/ 13][4] = {v1144[3:0]};
wire[31:0] v1146 = v0_rd_data[/*idx960=*/ 13];
assign v0_rd_en_input[/*idx960=*/ 13][4] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1147 = /*idx960=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1149[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1149[0] <= v1146;
always@(posedge clk) shiftreg1149[/*idx13=*/ 4:1] <= shiftreg1149[/*idx13=*/ 3:0];
wire [31:0] v1148 = shiftreg1149[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1150 = v1_rd_data[/*idx960=*/ 13][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 13][/*idx13=*/ 4][0] = tloop1128delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1151;
mult mult1152(v1151,
v1148,
v1150,
tloop1128delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1153 = v959[/*idx960=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1154;
add add1155(v1154,
v1151,
v1153,
tloop1128delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[14] = v1154;

//TerminatorOp

//} Unrolled body 13 of loop960.
//DEBUG: /*idx960=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop960.
//DEBUG: /*idx960=*/ 4'd14, expected 14
//printTimeOffset
reg tloop1142delay[3:0] = '{default:0} ;
always@(*) tloop1142delay[0] <= tloop1142;
generate
genvar i1157;

for(i1157 = 1; i1157<= 3; i1157= i1157 + 1) begin
always@(posedge clk) begin
tloop1142delay[i1157] <= tloop1142delay[i1157-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1156 = tloop1142delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1159[/*idx960=*/ 14:0] = '{default:0};
always@(*) shiftreg1159[0] <= idx11;
always@(posedge clk) shiftreg1159[/*idx960=*/ 14:1] <= shiftreg1159[/*idx960=*/ 13:0];
wire [31:0] v1158 = shiftreg1159[/*idx960=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 14][4] = tloop11delay[14];
assign v0_addr_input[/*idx960=*/ 14][4] = {v1158[3:0]};
wire[31:0] v1160 = v0_rd_data[/*idx960=*/ 14];
assign v0_rd_en_input[/*idx960=*/ 14][4] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1161 = /*idx960=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1163[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1163[0] <= v1160;
always@(posedge clk) shiftreg1163[/*idx13=*/ 4:1] <= shiftreg1163[/*idx13=*/ 3:0];
wire [31:0] v1162 = shiftreg1163[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1164 = v1_rd_data[/*idx960=*/ 14][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 14][/*idx13=*/ 4][0] = tloop1142delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1165;
mult mult1166(v1165,
v1162,
v1164,
tloop1142delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1167 = v959[/*idx960=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1168;
add add1169(v1168,
v1165,
v1167,
tloop1142delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[15] = v1168;

//TerminatorOp

//} Unrolled body 14 of loop960.
//DEBUG: /*idx960=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop960.
//DEBUG: /*idx960=*/ 4'd15, expected 15
//printTimeOffset
reg tloop1156delay[3:0] = '{default:0} ;
always@(*) tloop1156delay[0] <= tloop1156;
generate
genvar i1171;

for(i1171 = 1; i1171<= 3; i1171= i1171 + 1) begin
always@(posedge clk) begin
tloop1156delay[i1171] <= tloop1156delay[i1171-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1170 = tloop1156delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1173[/*idx960=*/ 15:0] = '{default:0};
always@(*) shiftreg1173[0] <= idx11;
always@(posedge clk) shiftreg1173[/*idx960=*/ 15:1] <= shiftreg1173[/*idx960=*/ 14:0];
wire [31:0] v1172 = shiftreg1173[/*idx960=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx960=*/ 15][4] = tloop11delay[15];
assign v0_addr_input[/*idx960=*/ 15][4] = {v1172[3:0]};
wire[31:0] v1174 = v0_rd_data[/*idx960=*/ 15];
assign v0_rd_en_input[/*idx960=*/ 15][4] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1175 = /*idx960=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1177[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1177[0] <= v1174;
always@(posedge clk) shiftreg1177[/*idx13=*/ 4:1] <= shiftreg1177[/*idx13=*/ 3:0];
wire [31:0] v1176 = shiftreg1177[/*idx13=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1178 = v1_rd_data[/*idx960=*/ 15][/*idx13=*/ 4];
assign v1_rd_en_input[/*idx960=*/ 15][/*idx13=*/ 4][0] = tloop1156delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1179;
mult mult1180(v1179,
v1176,
v1178,
tloop1156delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1181 = v959[/*idx960=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1182;
add add1183(v1182,
v1179,
v1181,
tloop1156delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v959[16] = v1182;

//TerminatorOp

//} Unrolled body 15 of loop960.
//DEBUG: /*idx960=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t1184;
assign t1184 = tloop1170;
//printTimeOffset
reg t1184delay[3:0] = '{default:0} ;
always@(*) t1184delay[0] <= t1184;
generate
genvar i1185;

for(i1185 = 1; i1185<= 3; i1185= i1185 + 1) begin
always@(posedge clk) begin
t1184delay[i1185] <= t1184delay[i1185-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v1186 = v959[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg1188[/*idx13=*/ 4:0] = '{default:0};
always@(*) shiftreg1188[0] <= idx11;
always@(posedge clk) shiftreg1188[/*idx13=*/ 4:1] <= shiftreg1188[/*idx13=*/ 3:0];
wire [31:0] v1187 = shiftreg1188[/*idx13=*/ 4];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg1190[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg1190[0] <= v1187;
always@(posedge clk) shiftreg1190[/*v10=*/ 16:1] <= shiftreg1190[/*v10=*/ 15:0];
wire [31:0] v1189 = shiftreg1190[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg1192[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg1192[0] <= v1189;
always@(posedge clk) shiftreg1192[/*v8=*/ 3:1] <= shiftreg1192[/*v8=*/ 2:0];
wire [31:0] v1191 = shiftreg1192[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 4][0] = t1184delay[3];
assign v2_addr_input[/*idx13=*/ 4][0] = {v1191[3:0]};
assign v2_wr_en_input[/*idx13=*/ 4][0] = t1184delay[3];
assign v2_wr_data_valid[/*idx13=*/ 4][0] = t1184delay[3];
assign v2_wr_data_input[/*idx13=*/ 4][0] = v1186;


//TerminatorOp

//} Unrolled body 4 of loop13.
//DEBUG: /*idx13=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop13.
//DEBUG: /*idx13=*/ 3'd5, expected 5
//printTimeOffset
reg tloop957delay[3:0] = '{default:0} ;
always@(*) tloop957delay[0] <= tloop957;
generate
genvar i1194;

for(i1194 = 1; i1194<= 3; i1194= i1194 + 1) begin
always@(posedge clk) begin
tloop957delay[i1194] <= tloop957delay[i1194-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop1193 = tloop957delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v1195[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v1195[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop1196.
//DEBUG: /*idx1196=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1197 = tloop957delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1199[/*idx1196=*/ 0:0] = '{default:0};
always@(*) shiftreg1199[0] <= idx11;
wire [31:0] v1198 = shiftreg1199[/*idx1196=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 0][5] = tloop11delay[0];
assign v0_addr_input[/*idx1196=*/ 0][5] = {v1198[3:0]};
wire[31:0] v1200 = v0_rd_data[/*idx1196=*/ 0];
assign v0_rd_en_input[/*idx1196=*/ 0][5] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1201 = /*idx1196=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1203[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1203[0] <= v1200;
always@(posedge clk) shiftreg1203[/*idx13=*/ 5:1] <= shiftreg1203[/*idx13=*/ 4:0];
wire [31:0] v1202 = shiftreg1203[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1204 = v1_rd_data[/*idx1196=*/ 0][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 0][/*idx13=*/ 5][0] = tloop957delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1205;
mult mult1206(v1205,
v1202,
v1204,
tloop957delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1207 = v1195[/*idx1196=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1208;
add add1209(v1208,
v1205,
v1207,
tloop957delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[1] = v1208;

//TerminatorOp

//} Unrolled body 0 of loop1196.
//DEBUG: /*idx1196=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop1196.
//DEBUG: /*idx1196=*/ 1'd1, expected 1
//printTimeOffset
reg tloop1197delay[3:0] = '{default:0} ;
always@(*) tloop1197delay[0] <= tloop1197;
generate
genvar i1211;

for(i1211 = 1; i1211<= 3; i1211= i1211 + 1) begin
always@(posedge clk) begin
tloop1197delay[i1211] <= tloop1197delay[i1211-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1210 = tloop1197delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1213[/*idx1196=*/ 1:0] = '{default:0};
always@(*) shiftreg1213[0] <= idx11;
always@(posedge clk) shiftreg1213[/*idx1196=*/ 1:1] <= shiftreg1213[/*idx1196=*/ 0:0];
wire [31:0] v1212 = shiftreg1213[/*idx1196=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 1][5] = tloop11delay[1];
assign v0_addr_input[/*idx1196=*/ 1][5] = {v1212[3:0]};
wire[31:0] v1214 = v0_rd_data[/*idx1196=*/ 1];
assign v0_rd_en_input[/*idx1196=*/ 1][5] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1215 = /*idx1196=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1217[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1217[0] <= v1214;
always@(posedge clk) shiftreg1217[/*idx13=*/ 5:1] <= shiftreg1217[/*idx13=*/ 4:0];
wire [31:0] v1216 = shiftreg1217[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1218 = v1_rd_data[/*idx1196=*/ 1][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 1][/*idx13=*/ 5][0] = tloop1197delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1219;
mult mult1220(v1219,
v1216,
v1218,
tloop1197delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1221 = v1195[/*idx1196=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1222;
add add1223(v1222,
v1219,
v1221,
tloop1197delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[2] = v1222;

//TerminatorOp

//} Unrolled body 1 of loop1196.
//DEBUG: /*idx1196=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop1196.
//DEBUG: /*idx1196=*/ 2'd2, expected 2
//printTimeOffset
reg tloop1210delay[3:0] = '{default:0} ;
always@(*) tloop1210delay[0] <= tloop1210;
generate
genvar i1225;

for(i1225 = 1; i1225<= 3; i1225= i1225 + 1) begin
always@(posedge clk) begin
tloop1210delay[i1225] <= tloop1210delay[i1225-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1224 = tloop1210delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1227[/*idx1196=*/ 2:0] = '{default:0};
always@(*) shiftreg1227[0] <= idx11;
always@(posedge clk) shiftreg1227[/*idx1196=*/ 2:1] <= shiftreg1227[/*idx1196=*/ 1:0];
wire [31:0] v1226 = shiftreg1227[/*idx1196=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 2][5] = tloop11delay[2];
assign v0_addr_input[/*idx1196=*/ 2][5] = {v1226[3:0]};
wire[31:0] v1228 = v0_rd_data[/*idx1196=*/ 2];
assign v0_rd_en_input[/*idx1196=*/ 2][5] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1229 = /*idx1196=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1231[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1231[0] <= v1228;
always@(posedge clk) shiftreg1231[/*idx13=*/ 5:1] <= shiftreg1231[/*idx13=*/ 4:0];
wire [31:0] v1230 = shiftreg1231[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1232 = v1_rd_data[/*idx1196=*/ 2][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 2][/*idx13=*/ 5][0] = tloop1210delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1233;
mult mult1234(v1233,
v1230,
v1232,
tloop1210delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1235 = v1195[/*idx1196=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1236;
add add1237(v1236,
v1233,
v1235,
tloop1210delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[3] = v1236;

//TerminatorOp

//} Unrolled body 2 of loop1196.
//DEBUG: /*idx1196=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop1196.
//DEBUG: /*idx1196=*/ 2'd3, expected 3
//printTimeOffset
reg tloop1224delay[3:0] = '{default:0} ;
always@(*) tloop1224delay[0] <= tloop1224;
generate
genvar i1239;

for(i1239 = 1; i1239<= 3; i1239= i1239 + 1) begin
always@(posedge clk) begin
tloop1224delay[i1239] <= tloop1224delay[i1239-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1238 = tloop1224delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1241[/*idx1196=*/ 3:0] = '{default:0};
always@(*) shiftreg1241[0] <= idx11;
always@(posedge clk) shiftreg1241[/*idx1196=*/ 3:1] <= shiftreg1241[/*idx1196=*/ 2:0];
wire [31:0] v1240 = shiftreg1241[/*idx1196=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 3][5] = tloop11delay[3];
assign v0_addr_input[/*idx1196=*/ 3][5] = {v1240[3:0]};
wire[31:0] v1242 = v0_rd_data[/*idx1196=*/ 3];
assign v0_rd_en_input[/*idx1196=*/ 3][5] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1243 = /*idx1196=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1245[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1245[0] <= v1242;
always@(posedge clk) shiftreg1245[/*idx13=*/ 5:1] <= shiftreg1245[/*idx13=*/ 4:0];
wire [31:0] v1244 = shiftreg1245[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1246 = v1_rd_data[/*idx1196=*/ 3][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 3][/*idx13=*/ 5][0] = tloop1224delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1247;
mult mult1248(v1247,
v1244,
v1246,
tloop1224delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1249 = v1195[/*idx1196=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1250;
add add1251(v1250,
v1247,
v1249,
tloop1224delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[4] = v1250;

//TerminatorOp

//} Unrolled body 3 of loop1196.
//DEBUG: /*idx1196=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop1196.
//DEBUG: /*idx1196=*/ 3'd4, expected 4
//printTimeOffset
reg tloop1238delay[3:0] = '{default:0} ;
always@(*) tloop1238delay[0] <= tloop1238;
generate
genvar i1253;

for(i1253 = 1; i1253<= 3; i1253= i1253 + 1) begin
always@(posedge clk) begin
tloop1238delay[i1253] <= tloop1238delay[i1253-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1252 = tloop1238delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1255[/*idx1196=*/ 4:0] = '{default:0};
always@(*) shiftreg1255[0] <= idx11;
always@(posedge clk) shiftreg1255[/*idx1196=*/ 4:1] <= shiftreg1255[/*idx1196=*/ 3:0];
wire [31:0] v1254 = shiftreg1255[/*idx1196=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 4][5] = tloop11delay[4];
assign v0_addr_input[/*idx1196=*/ 4][5] = {v1254[3:0]};
wire[31:0] v1256 = v0_rd_data[/*idx1196=*/ 4];
assign v0_rd_en_input[/*idx1196=*/ 4][5] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1257 = /*idx1196=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1259[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1259[0] <= v1256;
always@(posedge clk) shiftreg1259[/*idx13=*/ 5:1] <= shiftreg1259[/*idx13=*/ 4:0];
wire [31:0] v1258 = shiftreg1259[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1260 = v1_rd_data[/*idx1196=*/ 4][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 4][/*idx13=*/ 5][0] = tloop1238delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1261;
mult mult1262(v1261,
v1258,
v1260,
tloop1238delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1263 = v1195[/*idx1196=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1264;
add add1265(v1264,
v1261,
v1263,
tloop1238delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[5] = v1264;

//TerminatorOp

//} Unrolled body 4 of loop1196.
//DEBUG: /*idx1196=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop1196.
//DEBUG: /*idx1196=*/ 3'd5, expected 5
//printTimeOffset
reg tloop1252delay[3:0] = '{default:0} ;
always@(*) tloop1252delay[0] <= tloop1252;
generate
genvar i1267;

for(i1267 = 1; i1267<= 3; i1267= i1267 + 1) begin
always@(posedge clk) begin
tloop1252delay[i1267] <= tloop1252delay[i1267-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1266 = tloop1252delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1269[/*idx1196=*/ 5:0] = '{default:0};
always@(*) shiftreg1269[0] <= idx11;
always@(posedge clk) shiftreg1269[/*idx1196=*/ 5:1] <= shiftreg1269[/*idx1196=*/ 4:0];
wire [31:0] v1268 = shiftreg1269[/*idx1196=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 5][5] = tloop11delay[5];
assign v0_addr_input[/*idx1196=*/ 5][5] = {v1268[3:0]};
wire[31:0] v1270 = v0_rd_data[/*idx1196=*/ 5];
assign v0_rd_en_input[/*idx1196=*/ 5][5] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1271 = /*idx1196=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1273[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1273[0] <= v1270;
always@(posedge clk) shiftreg1273[/*idx13=*/ 5:1] <= shiftreg1273[/*idx13=*/ 4:0];
wire [31:0] v1272 = shiftreg1273[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1274 = v1_rd_data[/*idx1196=*/ 5][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 5][/*idx13=*/ 5][0] = tloop1252delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1275;
mult mult1276(v1275,
v1272,
v1274,
tloop1252delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1277 = v1195[/*idx1196=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1278;
add add1279(v1278,
v1275,
v1277,
tloop1252delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[6] = v1278;

//TerminatorOp

//} Unrolled body 5 of loop1196.
//DEBUG: /*idx1196=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop1196.
//DEBUG: /*idx1196=*/ 3'd6, expected 6
//printTimeOffset
reg tloop1266delay[3:0] = '{default:0} ;
always@(*) tloop1266delay[0] <= tloop1266;
generate
genvar i1281;

for(i1281 = 1; i1281<= 3; i1281= i1281 + 1) begin
always@(posedge clk) begin
tloop1266delay[i1281] <= tloop1266delay[i1281-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1280 = tloop1266delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1283[/*idx1196=*/ 6:0] = '{default:0};
always@(*) shiftreg1283[0] <= idx11;
always@(posedge clk) shiftreg1283[/*idx1196=*/ 6:1] <= shiftreg1283[/*idx1196=*/ 5:0];
wire [31:0] v1282 = shiftreg1283[/*idx1196=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 6][5] = tloop11delay[6];
assign v0_addr_input[/*idx1196=*/ 6][5] = {v1282[3:0]};
wire[31:0] v1284 = v0_rd_data[/*idx1196=*/ 6];
assign v0_rd_en_input[/*idx1196=*/ 6][5] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1285 = /*idx1196=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1287[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1287[0] <= v1284;
always@(posedge clk) shiftreg1287[/*idx13=*/ 5:1] <= shiftreg1287[/*idx13=*/ 4:0];
wire [31:0] v1286 = shiftreg1287[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1288 = v1_rd_data[/*idx1196=*/ 6][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 6][/*idx13=*/ 5][0] = tloop1266delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1289;
mult mult1290(v1289,
v1286,
v1288,
tloop1266delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1291 = v1195[/*idx1196=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1292;
add add1293(v1292,
v1289,
v1291,
tloop1266delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[7] = v1292;

//TerminatorOp

//} Unrolled body 6 of loop1196.
//DEBUG: /*idx1196=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop1196.
//DEBUG: /*idx1196=*/ 3'd7, expected 7
//printTimeOffset
reg tloop1280delay[3:0] = '{default:0} ;
always@(*) tloop1280delay[0] <= tloop1280;
generate
genvar i1295;

for(i1295 = 1; i1295<= 3; i1295= i1295 + 1) begin
always@(posedge clk) begin
tloop1280delay[i1295] <= tloop1280delay[i1295-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1294 = tloop1280delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1297[/*idx1196=*/ 7:0] = '{default:0};
always@(*) shiftreg1297[0] <= idx11;
always@(posedge clk) shiftreg1297[/*idx1196=*/ 7:1] <= shiftreg1297[/*idx1196=*/ 6:0];
wire [31:0] v1296 = shiftreg1297[/*idx1196=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 7][5] = tloop11delay[7];
assign v0_addr_input[/*idx1196=*/ 7][5] = {v1296[3:0]};
wire[31:0] v1298 = v0_rd_data[/*idx1196=*/ 7];
assign v0_rd_en_input[/*idx1196=*/ 7][5] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1299 = /*idx1196=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1301[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1301[0] <= v1298;
always@(posedge clk) shiftreg1301[/*idx13=*/ 5:1] <= shiftreg1301[/*idx13=*/ 4:0];
wire [31:0] v1300 = shiftreg1301[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1302 = v1_rd_data[/*idx1196=*/ 7][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 7][/*idx13=*/ 5][0] = tloop1280delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1303;
mult mult1304(v1303,
v1300,
v1302,
tloop1280delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1305 = v1195[/*idx1196=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1306;
add add1307(v1306,
v1303,
v1305,
tloop1280delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[8] = v1306;

//TerminatorOp

//} Unrolled body 7 of loop1196.
//DEBUG: /*idx1196=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop1196.
//DEBUG: /*idx1196=*/ 4'd8, expected 8
//printTimeOffset
reg tloop1294delay[3:0] = '{default:0} ;
always@(*) tloop1294delay[0] <= tloop1294;
generate
genvar i1309;

for(i1309 = 1; i1309<= 3; i1309= i1309 + 1) begin
always@(posedge clk) begin
tloop1294delay[i1309] <= tloop1294delay[i1309-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1308 = tloop1294delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1311[/*idx1196=*/ 8:0] = '{default:0};
always@(*) shiftreg1311[0] <= idx11;
always@(posedge clk) shiftreg1311[/*idx1196=*/ 8:1] <= shiftreg1311[/*idx1196=*/ 7:0];
wire [31:0] v1310 = shiftreg1311[/*idx1196=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 8][5] = tloop11delay[8];
assign v0_addr_input[/*idx1196=*/ 8][5] = {v1310[3:0]};
wire[31:0] v1312 = v0_rd_data[/*idx1196=*/ 8];
assign v0_rd_en_input[/*idx1196=*/ 8][5] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1313 = /*idx1196=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1315[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1315[0] <= v1312;
always@(posedge clk) shiftreg1315[/*idx13=*/ 5:1] <= shiftreg1315[/*idx13=*/ 4:0];
wire [31:0] v1314 = shiftreg1315[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1316 = v1_rd_data[/*idx1196=*/ 8][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 8][/*idx13=*/ 5][0] = tloop1294delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1317;
mult mult1318(v1317,
v1314,
v1316,
tloop1294delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1319 = v1195[/*idx1196=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1320;
add add1321(v1320,
v1317,
v1319,
tloop1294delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[9] = v1320;

//TerminatorOp

//} Unrolled body 8 of loop1196.
//DEBUG: /*idx1196=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop1196.
//DEBUG: /*idx1196=*/ 4'd9, expected 9
//printTimeOffset
reg tloop1308delay[3:0] = '{default:0} ;
always@(*) tloop1308delay[0] <= tloop1308;
generate
genvar i1323;

for(i1323 = 1; i1323<= 3; i1323= i1323 + 1) begin
always@(posedge clk) begin
tloop1308delay[i1323] <= tloop1308delay[i1323-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1322 = tloop1308delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1325[/*idx1196=*/ 9:0] = '{default:0};
always@(*) shiftreg1325[0] <= idx11;
always@(posedge clk) shiftreg1325[/*idx1196=*/ 9:1] <= shiftreg1325[/*idx1196=*/ 8:0];
wire [31:0] v1324 = shiftreg1325[/*idx1196=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 9][5] = tloop11delay[9];
assign v0_addr_input[/*idx1196=*/ 9][5] = {v1324[3:0]};
wire[31:0] v1326 = v0_rd_data[/*idx1196=*/ 9];
assign v0_rd_en_input[/*idx1196=*/ 9][5] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1327 = /*idx1196=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1329[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1329[0] <= v1326;
always@(posedge clk) shiftreg1329[/*idx13=*/ 5:1] <= shiftreg1329[/*idx13=*/ 4:0];
wire [31:0] v1328 = shiftreg1329[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1330 = v1_rd_data[/*idx1196=*/ 9][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 9][/*idx13=*/ 5][0] = tloop1308delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1331;
mult mult1332(v1331,
v1328,
v1330,
tloop1308delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1333 = v1195[/*idx1196=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1334;
add add1335(v1334,
v1331,
v1333,
tloop1308delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[10] = v1334;

//TerminatorOp

//} Unrolled body 9 of loop1196.
//DEBUG: /*idx1196=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop1196.
//DEBUG: /*idx1196=*/ 4'd10, expected 10
//printTimeOffset
reg tloop1322delay[3:0] = '{default:0} ;
always@(*) tloop1322delay[0] <= tloop1322;
generate
genvar i1337;

for(i1337 = 1; i1337<= 3; i1337= i1337 + 1) begin
always@(posedge clk) begin
tloop1322delay[i1337] <= tloop1322delay[i1337-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1336 = tloop1322delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1339[/*idx1196=*/ 10:0] = '{default:0};
always@(*) shiftreg1339[0] <= idx11;
always@(posedge clk) shiftreg1339[/*idx1196=*/ 10:1] <= shiftreg1339[/*idx1196=*/ 9:0];
wire [31:0] v1338 = shiftreg1339[/*idx1196=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 10][5] = tloop11delay[10];
assign v0_addr_input[/*idx1196=*/ 10][5] = {v1338[3:0]};
wire[31:0] v1340 = v0_rd_data[/*idx1196=*/ 10];
assign v0_rd_en_input[/*idx1196=*/ 10][5] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1341 = /*idx1196=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1343[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1343[0] <= v1340;
always@(posedge clk) shiftreg1343[/*idx13=*/ 5:1] <= shiftreg1343[/*idx13=*/ 4:0];
wire [31:0] v1342 = shiftreg1343[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1344 = v1_rd_data[/*idx1196=*/ 10][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 10][/*idx13=*/ 5][0] = tloop1322delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1345;
mult mult1346(v1345,
v1342,
v1344,
tloop1322delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1347 = v1195[/*idx1196=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1348;
add add1349(v1348,
v1345,
v1347,
tloop1322delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[11] = v1348;

//TerminatorOp

//} Unrolled body 10 of loop1196.
//DEBUG: /*idx1196=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop1196.
//DEBUG: /*idx1196=*/ 4'd11, expected 11
//printTimeOffset
reg tloop1336delay[3:0] = '{default:0} ;
always@(*) tloop1336delay[0] <= tloop1336;
generate
genvar i1351;

for(i1351 = 1; i1351<= 3; i1351= i1351 + 1) begin
always@(posedge clk) begin
tloop1336delay[i1351] <= tloop1336delay[i1351-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1350 = tloop1336delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1353[/*idx1196=*/ 11:0] = '{default:0};
always@(*) shiftreg1353[0] <= idx11;
always@(posedge clk) shiftreg1353[/*idx1196=*/ 11:1] <= shiftreg1353[/*idx1196=*/ 10:0];
wire [31:0] v1352 = shiftreg1353[/*idx1196=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 11][5] = tloop11delay[11];
assign v0_addr_input[/*idx1196=*/ 11][5] = {v1352[3:0]};
wire[31:0] v1354 = v0_rd_data[/*idx1196=*/ 11];
assign v0_rd_en_input[/*idx1196=*/ 11][5] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1355 = /*idx1196=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1357[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1357[0] <= v1354;
always@(posedge clk) shiftreg1357[/*idx13=*/ 5:1] <= shiftreg1357[/*idx13=*/ 4:0];
wire [31:0] v1356 = shiftreg1357[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1358 = v1_rd_data[/*idx1196=*/ 11][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 11][/*idx13=*/ 5][0] = tloop1336delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1359;
mult mult1360(v1359,
v1356,
v1358,
tloop1336delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1361 = v1195[/*idx1196=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1362;
add add1363(v1362,
v1359,
v1361,
tloop1336delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[12] = v1362;

//TerminatorOp

//} Unrolled body 11 of loop1196.
//DEBUG: /*idx1196=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop1196.
//DEBUG: /*idx1196=*/ 4'd12, expected 12
//printTimeOffset
reg tloop1350delay[3:0] = '{default:0} ;
always@(*) tloop1350delay[0] <= tloop1350;
generate
genvar i1365;

for(i1365 = 1; i1365<= 3; i1365= i1365 + 1) begin
always@(posedge clk) begin
tloop1350delay[i1365] <= tloop1350delay[i1365-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1364 = tloop1350delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1367[/*idx1196=*/ 12:0] = '{default:0};
always@(*) shiftreg1367[0] <= idx11;
always@(posedge clk) shiftreg1367[/*idx1196=*/ 12:1] <= shiftreg1367[/*idx1196=*/ 11:0];
wire [31:0] v1366 = shiftreg1367[/*idx1196=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 12][5] = tloop11delay[12];
assign v0_addr_input[/*idx1196=*/ 12][5] = {v1366[3:0]};
wire[31:0] v1368 = v0_rd_data[/*idx1196=*/ 12];
assign v0_rd_en_input[/*idx1196=*/ 12][5] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1369 = /*idx1196=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1371[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1371[0] <= v1368;
always@(posedge clk) shiftreg1371[/*idx13=*/ 5:1] <= shiftreg1371[/*idx13=*/ 4:0];
wire [31:0] v1370 = shiftreg1371[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1372 = v1_rd_data[/*idx1196=*/ 12][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 12][/*idx13=*/ 5][0] = tloop1350delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1373;
mult mult1374(v1373,
v1370,
v1372,
tloop1350delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1375 = v1195[/*idx1196=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1376;
add add1377(v1376,
v1373,
v1375,
tloop1350delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[13] = v1376;

//TerminatorOp

//} Unrolled body 12 of loop1196.
//DEBUG: /*idx1196=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop1196.
//DEBUG: /*idx1196=*/ 4'd13, expected 13
//printTimeOffset
reg tloop1364delay[3:0] = '{default:0} ;
always@(*) tloop1364delay[0] <= tloop1364;
generate
genvar i1379;

for(i1379 = 1; i1379<= 3; i1379= i1379 + 1) begin
always@(posedge clk) begin
tloop1364delay[i1379] <= tloop1364delay[i1379-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1378 = tloop1364delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1381[/*idx1196=*/ 13:0] = '{default:0};
always@(*) shiftreg1381[0] <= idx11;
always@(posedge clk) shiftreg1381[/*idx1196=*/ 13:1] <= shiftreg1381[/*idx1196=*/ 12:0];
wire [31:0] v1380 = shiftreg1381[/*idx1196=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 13][5] = tloop11delay[13];
assign v0_addr_input[/*idx1196=*/ 13][5] = {v1380[3:0]};
wire[31:0] v1382 = v0_rd_data[/*idx1196=*/ 13];
assign v0_rd_en_input[/*idx1196=*/ 13][5] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1383 = /*idx1196=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1385[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1385[0] <= v1382;
always@(posedge clk) shiftreg1385[/*idx13=*/ 5:1] <= shiftreg1385[/*idx13=*/ 4:0];
wire [31:0] v1384 = shiftreg1385[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1386 = v1_rd_data[/*idx1196=*/ 13][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 13][/*idx13=*/ 5][0] = tloop1364delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1387;
mult mult1388(v1387,
v1384,
v1386,
tloop1364delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1389 = v1195[/*idx1196=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1390;
add add1391(v1390,
v1387,
v1389,
tloop1364delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[14] = v1390;

//TerminatorOp

//} Unrolled body 13 of loop1196.
//DEBUG: /*idx1196=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop1196.
//DEBUG: /*idx1196=*/ 4'd14, expected 14
//printTimeOffset
reg tloop1378delay[3:0] = '{default:0} ;
always@(*) tloop1378delay[0] <= tloop1378;
generate
genvar i1393;

for(i1393 = 1; i1393<= 3; i1393= i1393 + 1) begin
always@(posedge clk) begin
tloop1378delay[i1393] <= tloop1378delay[i1393-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1392 = tloop1378delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1395[/*idx1196=*/ 14:0] = '{default:0};
always@(*) shiftreg1395[0] <= idx11;
always@(posedge clk) shiftreg1395[/*idx1196=*/ 14:1] <= shiftreg1395[/*idx1196=*/ 13:0];
wire [31:0] v1394 = shiftreg1395[/*idx1196=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 14][5] = tloop11delay[14];
assign v0_addr_input[/*idx1196=*/ 14][5] = {v1394[3:0]};
wire[31:0] v1396 = v0_rd_data[/*idx1196=*/ 14];
assign v0_rd_en_input[/*idx1196=*/ 14][5] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1397 = /*idx1196=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1399[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1399[0] <= v1396;
always@(posedge clk) shiftreg1399[/*idx13=*/ 5:1] <= shiftreg1399[/*idx13=*/ 4:0];
wire [31:0] v1398 = shiftreg1399[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1400 = v1_rd_data[/*idx1196=*/ 14][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 14][/*idx13=*/ 5][0] = tloop1378delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1401;
mult mult1402(v1401,
v1398,
v1400,
tloop1378delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1403 = v1195[/*idx1196=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1404;
add add1405(v1404,
v1401,
v1403,
tloop1378delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[15] = v1404;

//TerminatorOp

//} Unrolled body 14 of loop1196.
//DEBUG: /*idx1196=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop1196.
//DEBUG: /*idx1196=*/ 4'd15, expected 15
//printTimeOffset
reg tloop1392delay[3:0] = '{default:0} ;
always@(*) tloop1392delay[0] <= tloop1392;
generate
genvar i1407;

for(i1407 = 1; i1407<= 3; i1407= i1407 + 1) begin
always@(posedge clk) begin
tloop1392delay[i1407] <= tloop1392delay[i1407-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1406 = tloop1392delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1409[/*idx1196=*/ 15:0] = '{default:0};
always@(*) shiftreg1409[0] <= idx11;
always@(posedge clk) shiftreg1409[/*idx1196=*/ 15:1] <= shiftreg1409[/*idx1196=*/ 14:0];
wire [31:0] v1408 = shiftreg1409[/*idx1196=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1196=*/ 15][5] = tloop11delay[15];
assign v0_addr_input[/*idx1196=*/ 15][5] = {v1408[3:0]};
wire[31:0] v1410 = v0_rd_data[/*idx1196=*/ 15];
assign v0_rd_en_input[/*idx1196=*/ 15][5] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1411 = /*idx1196=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1413[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1413[0] <= v1410;
always@(posedge clk) shiftreg1413[/*idx13=*/ 5:1] <= shiftreg1413[/*idx13=*/ 4:0];
wire [31:0] v1412 = shiftreg1413[/*idx13=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1414 = v1_rd_data[/*idx1196=*/ 15][/*idx13=*/ 5];
assign v1_rd_en_input[/*idx1196=*/ 15][/*idx13=*/ 5][0] = tloop1392delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1415;
mult mult1416(v1415,
v1412,
v1414,
tloop1392delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1417 = v1195[/*idx1196=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1418;
add add1419(v1418,
v1415,
v1417,
tloop1392delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1195[16] = v1418;

//TerminatorOp

//} Unrolled body 15 of loop1196.
//DEBUG: /*idx1196=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t1420;
assign t1420 = tloop1406;
//printTimeOffset
reg t1420delay[3:0] = '{default:0} ;
always@(*) t1420delay[0] <= t1420;
generate
genvar i1421;

for(i1421 = 1; i1421<= 3; i1421= i1421 + 1) begin
always@(posedge clk) begin
t1420delay[i1421] <= t1420delay[i1421-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v1422 = v1195[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg1424[/*idx13=*/ 5:0] = '{default:0};
always@(*) shiftreg1424[0] <= idx11;
always@(posedge clk) shiftreg1424[/*idx13=*/ 5:1] <= shiftreg1424[/*idx13=*/ 4:0];
wire [31:0] v1423 = shiftreg1424[/*idx13=*/ 5];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg1426[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg1426[0] <= v1423;
always@(posedge clk) shiftreg1426[/*v10=*/ 16:1] <= shiftreg1426[/*v10=*/ 15:0];
wire [31:0] v1425 = shiftreg1426[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg1428[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg1428[0] <= v1425;
always@(posedge clk) shiftreg1428[/*v8=*/ 3:1] <= shiftreg1428[/*v8=*/ 2:0];
wire [31:0] v1427 = shiftreg1428[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 5][0] = t1420delay[3];
assign v2_addr_input[/*idx13=*/ 5][0] = {v1427[3:0]};
assign v2_wr_en_input[/*idx13=*/ 5][0] = t1420delay[3];
assign v2_wr_data_valid[/*idx13=*/ 5][0] = t1420delay[3];
assign v2_wr_data_input[/*idx13=*/ 5][0] = v1422;


//TerminatorOp

//} Unrolled body 5 of loop13.
//DEBUG: /*idx13=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop13.
//DEBUG: /*idx13=*/ 3'd6, expected 6
//printTimeOffset
reg tloop1193delay[3:0] = '{default:0} ;
always@(*) tloop1193delay[0] <= tloop1193;
generate
genvar i1430;

for(i1430 = 1; i1430<= 3; i1430= i1430 + 1) begin
always@(posedge clk) begin
tloop1193delay[i1430] <= tloop1193delay[i1430-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop1429 = tloop1193delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v1431[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v1431[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop1432.
//DEBUG: /*idx1432=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1433 = tloop1193delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1435[/*idx1432=*/ 0:0] = '{default:0};
always@(*) shiftreg1435[0] <= idx11;
wire [31:0] v1434 = shiftreg1435[/*idx1432=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 0][6] = tloop11delay[0];
assign v0_addr_input[/*idx1432=*/ 0][6] = {v1434[3:0]};
wire[31:0] v1436 = v0_rd_data[/*idx1432=*/ 0];
assign v0_rd_en_input[/*idx1432=*/ 0][6] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1437 = /*idx1432=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1439[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1439[0] <= v1436;
always@(posedge clk) shiftreg1439[/*idx13=*/ 6:1] <= shiftreg1439[/*idx13=*/ 5:0];
wire [31:0] v1438 = shiftreg1439[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1440 = v1_rd_data[/*idx1432=*/ 0][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 0][/*idx13=*/ 6][0] = tloop1193delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1441;
mult mult1442(v1441,
v1438,
v1440,
tloop1193delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1443 = v1431[/*idx1432=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1444;
add add1445(v1444,
v1441,
v1443,
tloop1193delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[1] = v1444;

//TerminatorOp

//} Unrolled body 0 of loop1432.
//DEBUG: /*idx1432=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop1432.
//DEBUG: /*idx1432=*/ 1'd1, expected 1
//printTimeOffset
reg tloop1433delay[3:0] = '{default:0} ;
always@(*) tloop1433delay[0] <= tloop1433;
generate
genvar i1447;

for(i1447 = 1; i1447<= 3; i1447= i1447 + 1) begin
always@(posedge clk) begin
tloop1433delay[i1447] <= tloop1433delay[i1447-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1446 = tloop1433delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1449[/*idx1432=*/ 1:0] = '{default:0};
always@(*) shiftreg1449[0] <= idx11;
always@(posedge clk) shiftreg1449[/*idx1432=*/ 1:1] <= shiftreg1449[/*idx1432=*/ 0:0];
wire [31:0] v1448 = shiftreg1449[/*idx1432=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 1][6] = tloop11delay[1];
assign v0_addr_input[/*idx1432=*/ 1][6] = {v1448[3:0]};
wire[31:0] v1450 = v0_rd_data[/*idx1432=*/ 1];
assign v0_rd_en_input[/*idx1432=*/ 1][6] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1451 = /*idx1432=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1453[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1453[0] <= v1450;
always@(posedge clk) shiftreg1453[/*idx13=*/ 6:1] <= shiftreg1453[/*idx13=*/ 5:0];
wire [31:0] v1452 = shiftreg1453[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1454 = v1_rd_data[/*idx1432=*/ 1][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 1][/*idx13=*/ 6][0] = tloop1433delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1455;
mult mult1456(v1455,
v1452,
v1454,
tloop1433delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1457 = v1431[/*idx1432=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1458;
add add1459(v1458,
v1455,
v1457,
tloop1433delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[2] = v1458;

//TerminatorOp

//} Unrolled body 1 of loop1432.
//DEBUG: /*idx1432=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop1432.
//DEBUG: /*idx1432=*/ 2'd2, expected 2
//printTimeOffset
reg tloop1446delay[3:0] = '{default:0} ;
always@(*) tloop1446delay[0] <= tloop1446;
generate
genvar i1461;

for(i1461 = 1; i1461<= 3; i1461= i1461 + 1) begin
always@(posedge clk) begin
tloop1446delay[i1461] <= tloop1446delay[i1461-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1460 = tloop1446delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1463[/*idx1432=*/ 2:0] = '{default:0};
always@(*) shiftreg1463[0] <= idx11;
always@(posedge clk) shiftreg1463[/*idx1432=*/ 2:1] <= shiftreg1463[/*idx1432=*/ 1:0];
wire [31:0] v1462 = shiftreg1463[/*idx1432=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 2][6] = tloop11delay[2];
assign v0_addr_input[/*idx1432=*/ 2][6] = {v1462[3:0]};
wire[31:0] v1464 = v0_rd_data[/*idx1432=*/ 2];
assign v0_rd_en_input[/*idx1432=*/ 2][6] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1465 = /*idx1432=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1467[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1467[0] <= v1464;
always@(posedge clk) shiftreg1467[/*idx13=*/ 6:1] <= shiftreg1467[/*idx13=*/ 5:0];
wire [31:0] v1466 = shiftreg1467[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1468 = v1_rd_data[/*idx1432=*/ 2][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 2][/*idx13=*/ 6][0] = tloop1446delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1469;
mult mult1470(v1469,
v1466,
v1468,
tloop1446delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1471 = v1431[/*idx1432=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1472;
add add1473(v1472,
v1469,
v1471,
tloop1446delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[3] = v1472;

//TerminatorOp

//} Unrolled body 2 of loop1432.
//DEBUG: /*idx1432=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop1432.
//DEBUG: /*idx1432=*/ 2'd3, expected 3
//printTimeOffset
reg tloop1460delay[3:0] = '{default:0} ;
always@(*) tloop1460delay[0] <= tloop1460;
generate
genvar i1475;

for(i1475 = 1; i1475<= 3; i1475= i1475 + 1) begin
always@(posedge clk) begin
tloop1460delay[i1475] <= tloop1460delay[i1475-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1474 = tloop1460delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1477[/*idx1432=*/ 3:0] = '{default:0};
always@(*) shiftreg1477[0] <= idx11;
always@(posedge clk) shiftreg1477[/*idx1432=*/ 3:1] <= shiftreg1477[/*idx1432=*/ 2:0];
wire [31:0] v1476 = shiftreg1477[/*idx1432=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 3][6] = tloop11delay[3];
assign v0_addr_input[/*idx1432=*/ 3][6] = {v1476[3:0]};
wire[31:0] v1478 = v0_rd_data[/*idx1432=*/ 3];
assign v0_rd_en_input[/*idx1432=*/ 3][6] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1479 = /*idx1432=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1481[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1481[0] <= v1478;
always@(posedge clk) shiftreg1481[/*idx13=*/ 6:1] <= shiftreg1481[/*idx13=*/ 5:0];
wire [31:0] v1480 = shiftreg1481[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1482 = v1_rd_data[/*idx1432=*/ 3][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 3][/*idx13=*/ 6][0] = tloop1460delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1483;
mult mult1484(v1483,
v1480,
v1482,
tloop1460delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1485 = v1431[/*idx1432=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1486;
add add1487(v1486,
v1483,
v1485,
tloop1460delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[4] = v1486;

//TerminatorOp

//} Unrolled body 3 of loop1432.
//DEBUG: /*idx1432=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop1432.
//DEBUG: /*idx1432=*/ 3'd4, expected 4
//printTimeOffset
reg tloop1474delay[3:0] = '{default:0} ;
always@(*) tloop1474delay[0] <= tloop1474;
generate
genvar i1489;

for(i1489 = 1; i1489<= 3; i1489= i1489 + 1) begin
always@(posedge clk) begin
tloop1474delay[i1489] <= tloop1474delay[i1489-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1488 = tloop1474delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1491[/*idx1432=*/ 4:0] = '{default:0};
always@(*) shiftreg1491[0] <= idx11;
always@(posedge clk) shiftreg1491[/*idx1432=*/ 4:1] <= shiftreg1491[/*idx1432=*/ 3:0];
wire [31:0] v1490 = shiftreg1491[/*idx1432=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 4][6] = tloop11delay[4];
assign v0_addr_input[/*idx1432=*/ 4][6] = {v1490[3:0]};
wire[31:0] v1492 = v0_rd_data[/*idx1432=*/ 4];
assign v0_rd_en_input[/*idx1432=*/ 4][6] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1493 = /*idx1432=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1495[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1495[0] <= v1492;
always@(posedge clk) shiftreg1495[/*idx13=*/ 6:1] <= shiftreg1495[/*idx13=*/ 5:0];
wire [31:0] v1494 = shiftreg1495[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1496 = v1_rd_data[/*idx1432=*/ 4][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 4][/*idx13=*/ 6][0] = tloop1474delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1497;
mult mult1498(v1497,
v1494,
v1496,
tloop1474delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1499 = v1431[/*idx1432=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1500;
add add1501(v1500,
v1497,
v1499,
tloop1474delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[5] = v1500;

//TerminatorOp

//} Unrolled body 4 of loop1432.
//DEBUG: /*idx1432=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop1432.
//DEBUG: /*idx1432=*/ 3'd5, expected 5
//printTimeOffset
reg tloop1488delay[3:0] = '{default:0} ;
always@(*) tloop1488delay[0] <= tloop1488;
generate
genvar i1503;

for(i1503 = 1; i1503<= 3; i1503= i1503 + 1) begin
always@(posedge clk) begin
tloop1488delay[i1503] <= tloop1488delay[i1503-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1502 = tloop1488delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1505[/*idx1432=*/ 5:0] = '{default:0};
always@(*) shiftreg1505[0] <= idx11;
always@(posedge clk) shiftreg1505[/*idx1432=*/ 5:1] <= shiftreg1505[/*idx1432=*/ 4:0];
wire [31:0] v1504 = shiftreg1505[/*idx1432=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 5][6] = tloop11delay[5];
assign v0_addr_input[/*idx1432=*/ 5][6] = {v1504[3:0]};
wire[31:0] v1506 = v0_rd_data[/*idx1432=*/ 5];
assign v0_rd_en_input[/*idx1432=*/ 5][6] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1507 = /*idx1432=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1509[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1509[0] <= v1506;
always@(posedge clk) shiftreg1509[/*idx13=*/ 6:1] <= shiftreg1509[/*idx13=*/ 5:0];
wire [31:0] v1508 = shiftreg1509[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1510 = v1_rd_data[/*idx1432=*/ 5][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 5][/*idx13=*/ 6][0] = tloop1488delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1511;
mult mult1512(v1511,
v1508,
v1510,
tloop1488delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1513 = v1431[/*idx1432=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1514;
add add1515(v1514,
v1511,
v1513,
tloop1488delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[6] = v1514;

//TerminatorOp

//} Unrolled body 5 of loop1432.
//DEBUG: /*idx1432=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop1432.
//DEBUG: /*idx1432=*/ 3'd6, expected 6
//printTimeOffset
reg tloop1502delay[3:0] = '{default:0} ;
always@(*) tloop1502delay[0] <= tloop1502;
generate
genvar i1517;

for(i1517 = 1; i1517<= 3; i1517= i1517 + 1) begin
always@(posedge clk) begin
tloop1502delay[i1517] <= tloop1502delay[i1517-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1516 = tloop1502delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1519[/*idx1432=*/ 6:0] = '{default:0};
always@(*) shiftreg1519[0] <= idx11;
always@(posedge clk) shiftreg1519[/*idx1432=*/ 6:1] <= shiftreg1519[/*idx1432=*/ 5:0];
wire [31:0] v1518 = shiftreg1519[/*idx1432=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 6][6] = tloop11delay[6];
assign v0_addr_input[/*idx1432=*/ 6][6] = {v1518[3:0]};
wire[31:0] v1520 = v0_rd_data[/*idx1432=*/ 6];
assign v0_rd_en_input[/*idx1432=*/ 6][6] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1521 = /*idx1432=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1523[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1523[0] <= v1520;
always@(posedge clk) shiftreg1523[/*idx13=*/ 6:1] <= shiftreg1523[/*idx13=*/ 5:0];
wire [31:0] v1522 = shiftreg1523[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1524 = v1_rd_data[/*idx1432=*/ 6][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 6][/*idx13=*/ 6][0] = tloop1502delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1525;
mult mult1526(v1525,
v1522,
v1524,
tloop1502delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1527 = v1431[/*idx1432=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1528;
add add1529(v1528,
v1525,
v1527,
tloop1502delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[7] = v1528;

//TerminatorOp

//} Unrolled body 6 of loop1432.
//DEBUG: /*idx1432=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop1432.
//DEBUG: /*idx1432=*/ 3'd7, expected 7
//printTimeOffset
reg tloop1516delay[3:0] = '{default:0} ;
always@(*) tloop1516delay[0] <= tloop1516;
generate
genvar i1531;

for(i1531 = 1; i1531<= 3; i1531= i1531 + 1) begin
always@(posedge clk) begin
tloop1516delay[i1531] <= tloop1516delay[i1531-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1530 = tloop1516delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1533[/*idx1432=*/ 7:0] = '{default:0};
always@(*) shiftreg1533[0] <= idx11;
always@(posedge clk) shiftreg1533[/*idx1432=*/ 7:1] <= shiftreg1533[/*idx1432=*/ 6:0];
wire [31:0] v1532 = shiftreg1533[/*idx1432=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 7][6] = tloop11delay[7];
assign v0_addr_input[/*idx1432=*/ 7][6] = {v1532[3:0]};
wire[31:0] v1534 = v0_rd_data[/*idx1432=*/ 7];
assign v0_rd_en_input[/*idx1432=*/ 7][6] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1535 = /*idx1432=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1537[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1537[0] <= v1534;
always@(posedge clk) shiftreg1537[/*idx13=*/ 6:1] <= shiftreg1537[/*idx13=*/ 5:0];
wire [31:0] v1536 = shiftreg1537[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1538 = v1_rd_data[/*idx1432=*/ 7][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 7][/*idx13=*/ 6][0] = tloop1516delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1539;
mult mult1540(v1539,
v1536,
v1538,
tloop1516delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1541 = v1431[/*idx1432=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1542;
add add1543(v1542,
v1539,
v1541,
tloop1516delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[8] = v1542;

//TerminatorOp

//} Unrolled body 7 of loop1432.
//DEBUG: /*idx1432=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop1432.
//DEBUG: /*idx1432=*/ 4'd8, expected 8
//printTimeOffset
reg tloop1530delay[3:0] = '{default:0} ;
always@(*) tloop1530delay[0] <= tloop1530;
generate
genvar i1545;

for(i1545 = 1; i1545<= 3; i1545= i1545 + 1) begin
always@(posedge clk) begin
tloop1530delay[i1545] <= tloop1530delay[i1545-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1544 = tloop1530delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1547[/*idx1432=*/ 8:0] = '{default:0};
always@(*) shiftreg1547[0] <= idx11;
always@(posedge clk) shiftreg1547[/*idx1432=*/ 8:1] <= shiftreg1547[/*idx1432=*/ 7:0];
wire [31:0] v1546 = shiftreg1547[/*idx1432=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 8][6] = tloop11delay[8];
assign v0_addr_input[/*idx1432=*/ 8][6] = {v1546[3:0]};
wire[31:0] v1548 = v0_rd_data[/*idx1432=*/ 8];
assign v0_rd_en_input[/*idx1432=*/ 8][6] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1549 = /*idx1432=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1551[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1551[0] <= v1548;
always@(posedge clk) shiftreg1551[/*idx13=*/ 6:1] <= shiftreg1551[/*idx13=*/ 5:0];
wire [31:0] v1550 = shiftreg1551[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1552 = v1_rd_data[/*idx1432=*/ 8][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 8][/*idx13=*/ 6][0] = tloop1530delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1553;
mult mult1554(v1553,
v1550,
v1552,
tloop1530delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1555 = v1431[/*idx1432=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1556;
add add1557(v1556,
v1553,
v1555,
tloop1530delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[9] = v1556;

//TerminatorOp

//} Unrolled body 8 of loop1432.
//DEBUG: /*idx1432=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop1432.
//DEBUG: /*idx1432=*/ 4'd9, expected 9
//printTimeOffset
reg tloop1544delay[3:0] = '{default:0} ;
always@(*) tloop1544delay[0] <= tloop1544;
generate
genvar i1559;

for(i1559 = 1; i1559<= 3; i1559= i1559 + 1) begin
always@(posedge clk) begin
tloop1544delay[i1559] <= tloop1544delay[i1559-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1558 = tloop1544delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1561[/*idx1432=*/ 9:0] = '{default:0};
always@(*) shiftreg1561[0] <= idx11;
always@(posedge clk) shiftreg1561[/*idx1432=*/ 9:1] <= shiftreg1561[/*idx1432=*/ 8:0];
wire [31:0] v1560 = shiftreg1561[/*idx1432=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 9][6] = tloop11delay[9];
assign v0_addr_input[/*idx1432=*/ 9][6] = {v1560[3:0]};
wire[31:0] v1562 = v0_rd_data[/*idx1432=*/ 9];
assign v0_rd_en_input[/*idx1432=*/ 9][6] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1563 = /*idx1432=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1565[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1565[0] <= v1562;
always@(posedge clk) shiftreg1565[/*idx13=*/ 6:1] <= shiftreg1565[/*idx13=*/ 5:0];
wire [31:0] v1564 = shiftreg1565[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1566 = v1_rd_data[/*idx1432=*/ 9][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 9][/*idx13=*/ 6][0] = tloop1544delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1567;
mult mult1568(v1567,
v1564,
v1566,
tloop1544delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1569 = v1431[/*idx1432=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1570;
add add1571(v1570,
v1567,
v1569,
tloop1544delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[10] = v1570;

//TerminatorOp

//} Unrolled body 9 of loop1432.
//DEBUG: /*idx1432=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop1432.
//DEBUG: /*idx1432=*/ 4'd10, expected 10
//printTimeOffset
reg tloop1558delay[3:0] = '{default:0} ;
always@(*) tloop1558delay[0] <= tloop1558;
generate
genvar i1573;

for(i1573 = 1; i1573<= 3; i1573= i1573 + 1) begin
always@(posedge clk) begin
tloop1558delay[i1573] <= tloop1558delay[i1573-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1572 = tloop1558delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1575[/*idx1432=*/ 10:0] = '{default:0};
always@(*) shiftreg1575[0] <= idx11;
always@(posedge clk) shiftreg1575[/*idx1432=*/ 10:1] <= shiftreg1575[/*idx1432=*/ 9:0];
wire [31:0] v1574 = shiftreg1575[/*idx1432=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 10][6] = tloop11delay[10];
assign v0_addr_input[/*idx1432=*/ 10][6] = {v1574[3:0]};
wire[31:0] v1576 = v0_rd_data[/*idx1432=*/ 10];
assign v0_rd_en_input[/*idx1432=*/ 10][6] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1577 = /*idx1432=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1579[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1579[0] <= v1576;
always@(posedge clk) shiftreg1579[/*idx13=*/ 6:1] <= shiftreg1579[/*idx13=*/ 5:0];
wire [31:0] v1578 = shiftreg1579[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1580 = v1_rd_data[/*idx1432=*/ 10][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 10][/*idx13=*/ 6][0] = tloop1558delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1581;
mult mult1582(v1581,
v1578,
v1580,
tloop1558delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1583 = v1431[/*idx1432=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1584;
add add1585(v1584,
v1581,
v1583,
tloop1558delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[11] = v1584;

//TerminatorOp

//} Unrolled body 10 of loop1432.
//DEBUG: /*idx1432=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop1432.
//DEBUG: /*idx1432=*/ 4'd11, expected 11
//printTimeOffset
reg tloop1572delay[3:0] = '{default:0} ;
always@(*) tloop1572delay[0] <= tloop1572;
generate
genvar i1587;

for(i1587 = 1; i1587<= 3; i1587= i1587 + 1) begin
always@(posedge clk) begin
tloop1572delay[i1587] <= tloop1572delay[i1587-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1586 = tloop1572delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1589[/*idx1432=*/ 11:0] = '{default:0};
always@(*) shiftreg1589[0] <= idx11;
always@(posedge clk) shiftreg1589[/*idx1432=*/ 11:1] <= shiftreg1589[/*idx1432=*/ 10:0];
wire [31:0] v1588 = shiftreg1589[/*idx1432=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 11][6] = tloop11delay[11];
assign v0_addr_input[/*idx1432=*/ 11][6] = {v1588[3:0]};
wire[31:0] v1590 = v0_rd_data[/*idx1432=*/ 11];
assign v0_rd_en_input[/*idx1432=*/ 11][6] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1591 = /*idx1432=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1593[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1593[0] <= v1590;
always@(posedge clk) shiftreg1593[/*idx13=*/ 6:1] <= shiftreg1593[/*idx13=*/ 5:0];
wire [31:0] v1592 = shiftreg1593[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1594 = v1_rd_data[/*idx1432=*/ 11][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 11][/*idx13=*/ 6][0] = tloop1572delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1595;
mult mult1596(v1595,
v1592,
v1594,
tloop1572delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1597 = v1431[/*idx1432=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1598;
add add1599(v1598,
v1595,
v1597,
tloop1572delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[12] = v1598;

//TerminatorOp

//} Unrolled body 11 of loop1432.
//DEBUG: /*idx1432=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop1432.
//DEBUG: /*idx1432=*/ 4'd12, expected 12
//printTimeOffset
reg tloop1586delay[3:0] = '{default:0} ;
always@(*) tloop1586delay[0] <= tloop1586;
generate
genvar i1601;

for(i1601 = 1; i1601<= 3; i1601= i1601 + 1) begin
always@(posedge clk) begin
tloop1586delay[i1601] <= tloop1586delay[i1601-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1600 = tloop1586delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1603[/*idx1432=*/ 12:0] = '{default:0};
always@(*) shiftreg1603[0] <= idx11;
always@(posedge clk) shiftreg1603[/*idx1432=*/ 12:1] <= shiftreg1603[/*idx1432=*/ 11:0];
wire [31:0] v1602 = shiftreg1603[/*idx1432=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 12][6] = tloop11delay[12];
assign v0_addr_input[/*idx1432=*/ 12][6] = {v1602[3:0]};
wire[31:0] v1604 = v0_rd_data[/*idx1432=*/ 12];
assign v0_rd_en_input[/*idx1432=*/ 12][6] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1605 = /*idx1432=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1607[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1607[0] <= v1604;
always@(posedge clk) shiftreg1607[/*idx13=*/ 6:1] <= shiftreg1607[/*idx13=*/ 5:0];
wire [31:0] v1606 = shiftreg1607[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1608 = v1_rd_data[/*idx1432=*/ 12][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 12][/*idx13=*/ 6][0] = tloop1586delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1609;
mult mult1610(v1609,
v1606,
v1608,
tloop1586delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1611 = v1431[/*idx1432=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1612;
add add1613(v1612,
v1609,
v1611,
tloop1586delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[13] = v1612;

//TerminatorOp

//} Unrolled body 12 of loop1432.
//DEBUG: /*idx1432=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop1432.
//DEBUG: /*idx1432=*/ 4'd13, expected 13
//printTimeOffset
reg tloop1600delay[3:0] = '{default:0} ;
always@(*) tloop1600delay[0] <= tloop1600;
generate
genvar i1615;

for(i1615 = 1; i1615<= 3; i1615= i1615 + 1) begin
always@(posedge clk) begin
tloop1600delay[i1615] <= tloop1600delay[i1615-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1614 = tloop1600delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1617[/*idx1432=*/ 13:0] = '{default:0};
always@(*) shiftreg1617[0] <= idx11;
always@(posedge clk) shiftreg1617[/*idx1432=*/ 13:1] <= shiftreg1617[/*idx1432=*/ 12:0];
wire [31:0] v1616 = shiftreg1617[/*idx1432=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 13][6] = tloop11delay[13];
assign v0_addr_input[/*idx1432=*/ 13][6] = {v1616[3:0]};
wire[31:0] v1618 = v0_rd_data[/*idx1432=*/ 13];
assign v0_rd_en_input[/*idx1432=*/ 13][6] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1619 = /*idx1432=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1621[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1621[0] <= v1618;
always@(posedge clk) shiftreg1621[/*idx13=*/ 6:1] <= shiftreg1621[/*idx13=*/ 5:0];
wire [31:0] v1620 = shiftreg1621[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1622 = v1_rd_data[/*idx1432=*/ 13][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 13][/*idx13=*/ 6][0] = tloop1600delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1623;
mult mult1624(v1623,
v1620,
v1622,
tloop1600delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1625 = v1431[/*idx1432=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1626;
add add1627(v1626,
v1623,
v1625,
tloop1600delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[14] = v1626;

//TerminatorOp

//} Unrolled body 13 of loop1432.
//DEBUG: /*idx1432=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop1432.
//DEBUG: /*idx1432=*/ 4'd14, expected 14
//printTimeOffset
reg tloop1614delay[3:0] = '{default:0} ;
always@(*) tloop1614delay[0] <= tloop1614;
generate
genvar i1629;

for(i1629 = 1; i1629<= 3; i1629= i1629 + 1) begin
always@(posedge clk) begin
tloop1614delay[i1629] <= tloop1614delay[i1629-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1628 = tloop1614delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1631[/*idx1432=*/ 14:0] = '{default:0};
always@(*) shiftreg1631[0] <= idx11;
always@(posedge clk) shiftreg1631[/*idx1432=*/ 14:1] <= shiftreg1631[/*idx1432=*/ 13:0];
wire [31:0] v1630 = shiftreg1631[/*idx1432=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 14][6] = tloop11delay[14];
assign v0_addr_input[/*idx1432=*/ 14][6] = {v1630[3:0]};
wire[31:0] v1632 = v0_rd_data[/*idx1432=*/ 14];
assign v0_rd_en_input[/*idx1432=*/ 14][6] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1633 = /*idx1432=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1635[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1635[0] <= v1632;
always@(posedge clk) shiftreg1635[/*idx13=*/ 6:1] <= shiftreg1635[/*idx13=*/ 5:0];
wire [31:0] v1634 = shiftreg1635[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1636 = v1_rd_data[/*idx1432=*/ 14][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 14][/*idx13=*/ 6][0] = tloop1614delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1637;
mult mult1638(v1637,
v1634,
v1636,
tloop1614delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1639 = v1431[/*idx1432=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1640;
add add1641(v1640,
v1637,
v1639,
tloop1614delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[15] = v1640;

//TerminatorOp

//} Unrolled body 14 of loop1432.
//DEBUG: /*idx1432=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop1432.
//DEBUG: /*idx1432=*/ 4'd15, expected 15
//printTimeOffset
reg tloop1628delay[3:0] = '{default:0} ;
always@(*) tloop1628delay[0] <= tloop1628;
generate
genvar i1643;

for(i1643 = 1; i1643<= 3; i1643= i1643 + 1) begin
always@(posedge clk) begin
tloop1628delay[i1643] <= tloop1628delay[i1643-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1642 = tloop1628delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1645[/*idx1432=*/ 15:0] = '{default:0};
always@(*) shiftreg1645[0] <= idx11;
always@(posedge clk) shiftreg1645[/*idx1432=*/ 15:1] <= shiftreg1645[/*idx1432=*/ 14:0];
wire [31:0] v1644 = shiftreg1645[/*idx1432=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1432=*/ 15][6] = tloop11delay[15];
assign v0_addr_input[/*idx1432=*/ 15][6] = {v1644[3:0]};
wire[31:0] v1646 = v0_rd_data[/*idx1432=*/ 15];
assign v0_rd_en_input[/*idx1432=*/ 15][6] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1647 = /*idx1432=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1649[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1649[0] <= v1646;
always@(posedge clk) shiftreg1649[/*idx13=*/ 6:1] <= shiftreg1649[/*idx13=*/ 5:0];
wire [31:0] v1648 = shiftreg1649[/*idx13=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1650 = v1_rd_data[/*idx1432=*/ 15][/*idx13=*/ 6];
assign v1_rd_en_input[/*idx1432=*/ 15][/*idx13=*/ 6][0] = tloop1628delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1651;
mult mult1652(v1651,
v1648,
v1650,
tloop1628delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1653 = v1431[/*idx1432=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1654;
add add1655(v1654,
v1651,
v1653,
tloop1628delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1431[16] = v1654;

//TerminatorOp

//} Unrolled body 15 of loop1432.
//DEBUG: /*idx1432=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t1656;
assign t1656 = tloop1642;
//printTimeOffset
reg t1656delay[3:0] = '{default:0} ;
always@(*) t1656delay[0] <= t1656;
generate
genvar i1657;

for(i1657 = 1; i1657<= 3; i1657= i1657 + 1) begin
always@(posedge clk) begin
t1656delay[i1657] <= t1656delay[i1657-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v1658 = v1431[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg1660[/*idx13=*/ 6:0] = '{default:0};
always@(*) shiftreg1660[0] <= idx11;
always@(posedge clk) shiftreg1660[/*idx13=*/ 6:1] <= shiftreg1660[/*idx13=*/ 5:0];
wire [31:0] v1659 = shiftreg1660[/*idx13=*/ 6];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg1662[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg1662[0] <= v1659;
always@(posedge clk) shiftreg1662[/*v10=*/ 16:1] <= shiftreg1662[/*v10=*/ 15:0];
wire [31:0] v1661 = shiftreg1662[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg1664[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg1664[0] <= v1661;
always@(posedge clk) shiftreg1664[/*v8=*/ 3:1] <= shiftreg1664[/*v8=*/ 2:0];
wire [31:0] v1663 = shiftreg1664[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 6][0] = t1656delay[3];
assign v2_addr_input[/*idx13=*/ 6][0] = {v1663[3:0]};
assign v2_wr_en_input[/*idx13=*/ 6][0] = t1656delay[3];
assign v2_wr_data_valid[/*idx13=*/ 6][0] = t1656delay[3];
assign v2_wr_data_input[/*idx13=*/ 6][0] = v1658;


//TerminatorOp

//} Unrolled body 6 of loop13.
//DEBUG: /*idx13=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop13.
//DEBUG: /*idx13=*/ 3'd7, expected 7
//printTimeOffset
reg tloop1429delay[3:0] = '{default:0} ;
always@(*) tloop1429delay[0] <= tloop1429;
generate
genvar i1666;

for(i1666 = 1; i1666<= 3; i1666= i1666 + 1) begin
always@(posedge clk) begin
tloop1429delay[i1666] <= tloop1429delay[i1666-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop1665 = tloop1429delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v1667[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v1667[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop1668.
//DEBUG: /*idx1668=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1669 = tloop1429delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1671[/*idx1668=*/ 0:0] = '{default:0};
always@(*) shiftreg1671[0] <= idx11;
wire [31:0] v1670 = shiftreg1671[/*idx1668=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 0][7] = tloop11delay[0];
assign v0_addr_input[/*idx1668=*/ 0][7] = {v1670[3:0]};
wire[31:0] v1672 = v0_rd_data[/*idx1668=*/ 0];
assign v0_rd_en_input[/*idx1668=*/ 0][7] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1673 = /*idx1668=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1675[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1675[0] <= v1672;
always@(posedge clk) shiftreg1675[/*idx13=*/ 7:1] <= shiftreg1675[/*idx13=*/ 6:0];
wire [31:0] v1674 = shiftreg1675[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1676 = v1_rd_data[/*idx1668=*/ 0][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 0][/*idx13=*/ 7][0] = tloop1429delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1677;
mult mult1678(v1677,
v1674,
v1676,
tloop1429delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1679 = v1667[/*idx1668=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1680;
add add1681(v1680,
v1677,
v1679,
tloop1429delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[1] = v1680;

//TerminatorOp

//} Unrolled body 0 of loop1668.
//DEBUG: /*idx1668=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop1668.
//DEBUG: /*idx1668=*/ 1'd1, expected 1
//printTimeOffset
reg tloop1669delay[3:0] = '{default:0} ;
always@(*) tloop1669delay[0] <= tloop1669;
generate
genvar i1683;

for(i1683 = 1; i1683<= 3; i1683= i1683 + 1) begin
always@(posedge clk) begin
tloop1669delay[i1683] <= tloop1669delay[i1683-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1682 = tloop1669delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1685[/*idx1668=*/ 1:0] = '{default:0};
always@(*) shiftreg1685[0] <= idx11;
always@(posedge clk) shiftreg1685[/*idx1668=*/ 1:1] <= shiftreg1685[/*idx1668=*/ 0:0];
wire [31:0] v1684 = shiftreg1685[/*idx1668=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 1][7] = tloop11delay[1];
assign v0_addr_input[/*idx1668=*/ 1][7] = {v1684[3:0]};
wire[31:0] v1686 = v0_rd_data[/*idx1668=*/ 1];
assign v0_rd_en_input[/*idx1668=*/ 1][7] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1687 = /*idx1668=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1689[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1689[0] <= v1686;
always@(posedge clk) shiftreg1689[/*idx13=*/ 7:1] <= shiftreg1689[/*idx13=*/ 6:0];
wire [31:0] v1688 = shiftreg1689[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1690 = v1_rd_data[/*idx1668=*/ 1][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 1][/*idx13=*/ 7][0] = tloop1669delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1691;
mult mult1692(v1691,
v1688,
v1690,
tloop1669delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1693 = v1667[/*idx1668=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1694;
add add1695(v1694,
v1691,
v1693,
tloop1669delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[2] = v1694;

//TerminatorOp

//} Unrolled body 1 of loop1668.
//DEBUG: /*idx1668=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop1668.
//DEBUG: /*idx1668=*/ 2'd2, expected 2
//printTimeOffset
reg tloop1682delay[3:0] = '{default:0} ;
always@(*) tloop1682delay[0] <= tloop1682;
generate
genvar i1697;

for(i1697 = 1; i1697<= 3; i1697= i1697 + 1) begin
always@(posedge clk) begin
tloop1682delay[i1697] <= tloop1682delay[i1697-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1696 = tloop1682delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1699[/*idx1668=*/ 2:0] = '{default:0};
always@(*) shiftreg1699[0] <= idx11;
always@(posedge clk) shiftreg1699[/*idx1668=*/ 2:1] <= shiftreg1699[/*idx1668=*/ 1:0];
wire [31:0] v1698 = shiftreg1699[/*idx1668=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 2][7] = tloop11delay[2];
assign v0_addr_input[/*idx1668=*/ 2][7] = {v1698[3:0]};
wire[31:0] v1700 = v0_rd_data[/*idx1668=*/ 2];
assign v0_rd_en_input[/*idx1668=*/ 2][7] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1701 = /*idx1668=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1703[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1703[0] <= v1700;
always@(posedge clk) shiftreg1703[/*idx13=*/ 7:1] <= shiftreg1703[/*idx13=*/ 6:0];
wire [31:0] v1702 = shiftreg1703[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1704 = v1_rd_data[/*idx1668=*/ 2][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 2][/*idx13=*/ 7][0] = tloop1682delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1705;
mult mult1706(v1705,
v1702,
v1704,
tloop1682delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1707 = v1667[/*idx1668=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1708;
add add1709(v1708,
v1705,
v1707,
tloop1682delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[3] = v1708;

//TerminatorOp

//} Unrolled body 2 of loop1668.
//DEBUG: /*idx1668=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop1668.
//DEBUG: /*idx1668=*/ 2'd3, expected 3
//printTimeOffset
reg tloop1696delay[3:0] = '{default:0} ;
always@(*) tloop1696delay[0] <= tloop1696;
generate
genvar i1711;

for(i1711 = 1; i1711<= 3; i1711= i1711 + 1) begin
always@(posedge clk) begin
tloop1696delay[i1711] <= tloop1696delay[i1711-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1710 = tloop1696delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1713[/*idx1668=*/ 3:0] = '{default:0};
always@(*) shiftreg1713[0] <= idx11;
always@(posedge clk) shiftreg1713[/*idx1668=*/ 3:1] <= shiftreg1713[/*idx1668=*/ 2:0];
wire [31:0] v1712 = shiftreg1713[/*idx1668=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 3][7] = tloop11delay[3];
assign v0_addr_input[/*idx1668=*/ 3][7] = {v1712[3:0]};
wire[31:0] v1714 = v0_rd_data[/*idx1668=*/ 3];
assign v0_rd_en_input[/*idx1668=*/ 3][7] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1715 = /*idx1668=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1717[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1717[0] <= v1714;
always@(posedge clk) shiftreg1717[/*idx13=*/ 7:1] <= shiftreg1717[/*idx13=*/ 6:0];
wire [31:0] v1716 = shiftreg1717[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1718 = v1_rd_data[/*idx1668=*/ 3][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 3][/*idx13=*/ 7][0] = tloop1696delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1719;
mult mult1720(v1719,
v1716,
v1718,
tloop1696delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1721 = v1667[/*idx1668=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1722;
add add1723(v1722,
v1719,
v1721,
tloop1696delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[4] = v1722;

//TerminatorOp

//} Unrolled body 3 of loop1668.
//DEBUG: /*idx1668=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop1668.
//DEBUG: /*idx1668=*/ 3'd4, expected 4
//printTimeOffset
reg tloop1710delay[3:0] = '{default:0} ;
always@(*) tloop1710delay[0] <= tloop1710;
generate
genvar i1725;

for(i1725 = 1; i1725<= 3; i1725= i1725 + 1) begin
always@(posedge clk) begin
tloop1710delay[i1725] <= tloop1710delay[i1725-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1724 = tloop1710delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1727[/*idx1668=*/ 4:0] = '{default:0};
always@(*) shiftreg1727[0] <= idx11;
always@(posedge clk) shiftreg1727[/*idx1668=*/ 4:1] <= shiftreg1727[/*idx1668=*/ 3:0];
wire [31:0] v1726 = shiftreg1727[/*idx1668=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 4][7] = tloop11delay[4];
assign v0_addr_input[/*idx1668=*/ 4][7] = {v1726[3:0]};
wire[31:0] v1728 = v0_rd_data[/*idx1668=*/ 4];
assign v0_rd_en_input[/*idx1668=*/ 4][7] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1729 = /*idx1668=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1731[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1731[0] <= v1728;
always@(posedge clk) shiftreg1731[/*idx13=*/ 7:1] <= shiftreg1731[/*idx13=*/ 6:0];
wire [31:0] v1730 = shiftreg1731[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1732 = v1_rd_data[/*idx1668=*/ 4][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 4][/*idx13=*/ 7][0] = tloop1710delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1733;
mult mult1734(v1733,
v1730,
v1732,
tloop1710delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1735 = v1667[/*idx1668=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1736;
add add1737(v1736,
v1733,
v1735,
tloop1710delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[5] = v1736;

//TerminatorOp

//} Unrolled body 4 of loop1668.
//DEBUG: /*idx1668=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop1668.
//DEBUG: /*idx1668=*/ 3'd5, expected 5
//printTimeOffset
reg tloop1724delay[3:0] = '{default:0} ;
always@(*) tloop1724delay[0] <= tloop1724;
generate
genvar i1739;

for(i1739 = 1; i1739<= 3; i1739= i1739 + 1) begin
always@(posedge clk) begin
tloop1724delay[i1739] <= tloop1724delay[i1739-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1738 = tloop1724delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1741[/*idx1668=*/ 5:0] = '{default:0};
always@(*) shiftreg1741[0] <= idx11;
always@(posedge clk) shiftreg1741[/*idx1668=*/ 5:1] <= shiftreg1741[/*idx1668=*/ 4:0];
wire [31:0] v1740 = shiftreg1741[/*idx1668=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 5][7] = tloop11delay[5];
assign v0_addr_input[/*idx1668=*/ 5][7] = {v1740[3:0]};
wire[31:0] v1742 = v0_rd_data[/*idx1668=*/ 5];
assign v0_rd_en_input[/*idx1668=*/ 5][7] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1743 = /*idx1668=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1745[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1745[0] <= v1742;
always@(posedge clk) shiftreg1745[/*idx13=*/ 7:1] <= shiftreg1745[/*idx13=*/ 6:0];
wire [31:0] v1744 = shiftreg1745[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1746 = v1_rd_data[/*idx1668=*/ 5][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 5][/*idx13=*/ 7][0] = tloop1724delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1747;
mult mult1748(v1747,
v1744,
v1746,
tloop1724delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1749 = v1667[/*idx1668=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1750;
add add1751(v1750,
v1747,
v1749,
tloop1724delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[6] = v1750;

//TerminatorOp

//} Unrolled body 5 of loop1668.
//DEBUG: /*idx1668=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop1668.
//DEBUG: /*idx1668=*/ 3'd6, expected 6
//printTimeOffset
reg tloop1738delay[3:0] = '{default:0} ;
always@(*) tloop1738delay[0] <= tloop1738;
generate
genvar i1753;

for(i1753 = 1; i1753<= 3; i1753= i1753 + 1) begin
always@(posedge clk) begin
tloop1738delay[i1753] <= tloop1738delay[i1753-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1752 = tloop1738delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1755[/*idx1668=*/ 6:0] = '{default:0};
always@(*) shiftreg1755[0] <= idx11;
always@(posedge clk) shiftreg1755[/*idx1668=*/ 6:1] <= shiftreg1755[/*idx1668=*/ 5:0];
wire [31:0] v1754 = shiftreg1755[/*idx1668=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 6][7] = tloop11delay[6];
assign v0_addr_input[/*idx1668=*/ 6][7] = {v1754[3:0]};
wire[31:0] v1756 = v0_rd_data[/*idx1668=*/ 6];
assign v0_rd_en_input[/*idx1668=*/ 6][7] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1757 = /*idx1668=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1759[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1759[0] <= v1756;
always@(posedge clk) shiftreg1759[/*idx13=*/ 7:1] <= shiftreg1759[/*idx13=*/ 6:0];
wire [31:0] v1758 = shiftreg1759[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1760 = v1_rd_data[/*idx1668=*/ 6][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 6][/*idx13=*/ 7][0] = tloop1738delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1761;
mult mult1762(v1761,
v1758,
v1760,
tloop1738delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1763 = v1667[/*idx1668=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1764;
add add1765(v1764,
v1761,
v1763,
tloop1738delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[7] = v1764;

//TerminatorOp

//} Unrolled body 6 of loop1668.
//DEBUG: /*idx1668=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop1668.
//DEBUG: /*idx1668=*/ 3'd7, expected 7
//printTimeOffset
reg tloop1752delay[3:0] = '{default:0} ;
always@(*) tloop1752delay[0] <= tloop1752;
generate
genvar i1767;

for(i1767 = 1; i1767<= 3; i1767= i1767 + 1) begin
always@(posedge clk) begin
tloop1752delay[i1767] <= tloop1752delay[i1767-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1766 = tloop1752delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1769[/*idx1668=*/ 7:0] = '{default:0};
always@(*) shiftreg1769[0] <= idx11;
always@(posedge clk) shiftreg1769[/*idx1668=*/ 7:1] <= shiftreg1769[/*idx1668=*/ 6:0];
wire [31:0] v1768 = shiftreg1769[/*idx1668=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 7][7] = tloop11delay[7];
assign v0_addr_input[/*idx1668=*/ 7][7] = {v1768[3:0]};
wire[31:0] v1770 = v0_rd_data[/*idx1668=*/ 7];
assign v0_rd_en_input[/*idx1668=*/ 7][7] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1771 = /*idx1668=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1773[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1773[0] <= v1770;
always@(posedge clk) shiftreg1773[/*idx13=*/ 7:1] <= shiftreg1773[/*idx13=*/ 6:0];
wire [31:0] v1772 = shiftreg1773[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1774 = v1_rd_data[/*idx1668=*/ 7][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 7][/*idx13=*/ 7][0] = tloop1752delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1775;
mult mult1776(v1775,
v1772,
v1774,
tloop1752delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1777 = v1667[/*idx1668=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1778;
add add1779(v1778,
v1775,
v1777,
tloop1752delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[8] = v1778;

//TerminatorOp

//} Unrolled body 7 of loop1668.
//DEBUG: /*idx1668=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop1668.
//DEBUG: /*idx1668=*/ 4'd8, expected 8
//printTimeOffset
reg tloop1766delay[3:0] = '{default:0} ;
always@(*) tloop1766delay[0] <= tloop1766;
generate
genvar i1781;

for(i1781 = 1; i1781<= 3; i1781= i1781 + 1) begin
always@(posedge clk) begin
tloop1766delay[i1781] <= tloop1766delay[i1781-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1780 = tloop1766delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1783[/*idx1668=*/ 8:0] = '{default:0};
always@(*) shiftreg1783[0] <= idx11;
always@(posedge clk) shiftreg1783[/*idx1668=*/ 8:1] <= shiftreg1783[/*idx1668=*/ 7:0];
wire [31:0] v1782 = shiftreg1783[/*idx1668=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 8][7] = tloop11delay[8];
assign v0_addr_input[/*idx1668=*/ 8][7] = {v1782[3:0]};
wire[31:0] v1784 = v0_rd_data[/*idx1668=*/ 8];
assign v0_rd_en_input[/*idx1668=*/ 8][7] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1785 = /*idx1668=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1787[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1787[0] <= v1784;
always@(posedge clk) shiftreg1787[/*idx13=*/ 7:1] <= shiftreg1787[/*idx13=*/ 6:0];
wire [31:0] v1786 = shiftreg1787[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1788 = v1_rd_data[/*idx1668=*/ 8][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 8][/*idx13=*/ 7][0] = tloop1766delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1789;
mult mult1790(v1789,
v1786,
v1788,
tloop1766delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1791 = v1667[/*idx1668=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1792;
add add1793(v1792,
v1789,
v1791,
tloop1766delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[9] = v1792;

//TerminatorOp

//} Unrolled body 8 of loop1668.
//DEBUG: /*idx1668=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop1668.
//DEBUG: /*idx1668=*/ 4'd9, expected 9
//printTimeOffset
reg tloop1780delay[3:0] = '{default:0} ;
always@(*) tloop1780delay[0] <= tloop1780;
generate
genvar i1795;

for(i1795 = 1; i1795<= 3; i1795= i1795 + 1) begin
always@(posedge clk) begin
tloop1780delay[i1795] <= tloop1780delay[i1795-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1794 = tloop1780delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1797[/*idx1668=*/ 9:0] = '{default:0};
always@(*) shiftreg1797[0] <= idx11;
always@(posedge clk) shiftreg1797[/*idx1668=*/ 9:1] <= shiftreg1797[/*idx1668=*/ 8:0];
wire [31:0] v1796 = shiftreg1797[/*idx1668=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 9][7] = tloop11delay[9];
assign v0_addr_input[/*idx1668=*/ 9][7] = {v1796[3:0]};
wire[31:0] v1798 = v0_rd_data[/*idx1668=*/ 9];
assign v0_rd_en_input[/*idx1668=*/ 9][7] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1799 = /*idx1668=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1801[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1801[0] <= v1798;
always@(posedge clk) shiftreg1801[/*idx13=*/ 7:1] <= shiftreg1801[/*idx13=*/ 6:0];
wire [31:0] v1800 = shiftreg1801[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1802 = v1_rd_data[/*idx1668=*/ 9][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 9][/*idx13=*/ 7][0] = tloop1780delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1803;
mult mult1804(v1803,
v1800,
v1802,
tloop1780delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1805 = v1667[/*idx1668=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1806;
add add1807(v1806,
v1803,
v1805,
tloop1780delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[10] = v1806;

//TerminatorOp

//} Unrolled body 9 of loop1668.
//DEBUG: /*idx1668=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop1668.
//DEBUG: /*idx1668=*/ 4'd10, expected 10
//printTimeOffset
reg tloop1794delay[3:0] = '{default:0} ;
always@(*) tloop1794delay[0] <= tloop1794;
generate
genvar i1809;

for(i1809 = 1; i1809<= 3; i1809= i1809 + 1) begin
always@(posedge clk) begin
tloop1794delay[i1809] <= tloop1794delay[i1809-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1808 = tloop1794delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1811[/*idx1668=*/ 10:0] = '{default:0};
always@(*) shiftreg1811[0] <= idx11;
always@(posedge clk) shiftreg1811[/*idx1668=*/ 10:1] <= shiftreg1811[/*idx1668=*/ 9:0];
wire [31:0] v1810 = shiftreg1811[/*idx1668=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 10][7] = tloop11delay[10];
assign v0_addr_input[/*idx1668=*/ 10][7] = {v1810[3:0]};
wire[31:0] v1812 = v0_rd_data[/*idx1668=*/ 10];
assign v0_rd_en_input[/*idx1668=*/ 10][7] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1813 = /*idx1668=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1815[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1815[0] <= v1812;
always@(posedge clk) shiftreg1815[/*idx13=*/ 7:1] <= shiftreg1815[/*idx13=*/ 6:0];
wire [31:0] v1814 = shiftreg1815[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1816 = v1_rd_data[/*idx1668=*/ 10][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 10][/*idx13=*/ 7][0] = tloop1794delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1817;
mult mult1818(v1817,
v1814,
v1816,
tloop1794delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1819 = v1667[/*idx1668=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1820;
add add1821(v1820,
v1817,
v1819,
tloop1794delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[11] = v1820;

//TerminatorOp

//} Unrolled body 10 of loop1668.
//DEBUG: /*idx1668=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop1668.
//DEBUG: /*idx1668=*/ 4'd11, expected 11
//printTimeOffset
reg tloop1808delay[3:0] = '{default:0} ;
always@(*) tloop1808delay[0] <= tloop1808;
generate
genvar i1823;

for(i1823 = 1; i1823<= 3; i1823= i1823 + 1) begin
always@(posedge clk) begin
tloop1808delay[i1823] <= tloop1808delay[i1823-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1822 = tloop1808delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1825[/*idx1668=*/ 11:0] = '{default:0};
always@(*) shiftreg1825[0] <= idx11;
always@(posedge clk) shiftreg1825[/*idx1668=*/ 11:1] <= shiftreg1825[/*idx1668=*/ 10:0];
wire [31:0] v1824 = shiftreg1825[/*idx1668=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 11][7] = tloop11delay[11];
assign v0_addr_input[/*idx1668=*/ 11][7] = {v1824[3:0]};
wire[31:0] v1826 = v0_rd_data[/*idx1668=*/ 11];
assign v0_rd_en_input[/*idx1668=*/ 11][7] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1827 = /*idx1668=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1829[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1829[0] <= v1826;
always@(posedge clk) shiftreg1829[/*idx13=*/ 7:1] <= shiftreg1829[/*idx13=*/ 6:0];
wire [31:0] v1828 = shiftreg1829[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1830 = v1_rd_data[/*idx1668=*/ 11][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 11][/*idx13=*/ 7][0] = tloop1808delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1831;
mult mult1832(v1831,
v1828,
v1830,
tloop1808delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1833 = v1667[/*idx1668=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1834;
add add1835(v1834,
v1831,
v1833,
tloop1808delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[12] = v1834;

//TerminatorOp

//} Unrolled body 11 of loop1668.
//DEBUG: /*idx1668=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop1668.
//DEBUG: /*idx1668=*/ 4'd12, expected 12
//printTimeOffset
reg tloop1822delay[3:0] = '{default:0} ;
always@(*) tloop1822delay[0] <= tloop1822;
generate
genvar i1837;

for(i1837 = 1; i1837<= 3; i1837= i1837 + 1) begin
always@(posedge clk) begin
tloop1822delay[i1837] <= tloop1822delay[i1837-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1836 = tloop1822delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1839[/*idx1668=*/ 12:0] = '{default:0};
always@(*) shiftreg1839[0] <= idx11;
always@(posedge clk) shiftreg1839[/*idx1668=*/ 12:1] <= shiftreg1839[/*idx1668=*/ 11:0];
wire [31:0] v1838 = shiftreg1839[/*idx1668=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 12][7] = tloop11delay[12];
assign v0_addr_input[/*idx1668=*/ 12][7] = {v1838[3:0]};
wire[31:0] v1840 = v0_rd_data[/*idx1668=*/ 12];
assign v0_rd_en_input[/*idx1668=*/ 12][7] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1841 = /*idx1668=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1843[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1843[0] <= v1840;
always@(posedge clk) shiftreg1843[/*idx13=*/ 7:1] <= shiftreg1843[/*idx13=*/ 6:0];
wire [31:0] v1842 = shiftreg1843[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1844 = v1_rd_data[/*idx1668=*/ 12][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 12][/*idx13=*/ 7][0] = tloop1822delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1845;
mult mult1846(v1845,
v1842,
v1844,
tloop1822delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1847 = v1667[/*idx1668=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1848;
add add1849(v1848,
v1845,
v1847,
tloop1822delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[13] = v1848;

//TerminatorOp

//} Unrolled body 12 of loop1668.
//DEBUG: /*idx1668=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop1668.
//DEBUG: /*idx1668=*/ 4'd13, expected 13
//printTimeOffset
reg tloop1836delay[3:0] = '{default:0} ;
always@(*) tloop1836delay[0] <= tloop1836;
generate
genvar i1851;

for(i1851 = 1; i1851<= 3; i1851= i1851 + 1) begin
always@(posedge clk) begin
tloop1836delay[i1851] <= tloop1836delay[i1851-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1850 = tloop1836delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1853[/*idx1668=*/ 13:0] = '{default:0};
always@(*) shiftreg1853[0] <= idx11;
always@(posedge clk) shiftreg1853[/*idx1668=*/ 13:1] <= shiftreg1853[/*idx1668=*/ 12:0];
wire [31:0] v1852 = shiftreg1853[/*idx1668=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 13][7] = tloop11delay[13];
assign v0_addr_input[/*idx1668=*/ 13][7] = {v1852[3:0]};
wire[31:0] v1854 = v0_rd_data[/*idx1668=*/ 13];
assign v0_rd_en_input[/*idx1668=*/ 13][7] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1855 = /*idx1668=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1857[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1857[0] <= v1854;
always@(posedge clk) shiftreg1857[/*idx13=*/ 7:1] <= shiftreg1857[/*idx13=*/ 6:0];
wire [31:0] v1856 = shiftreg1857[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1858 = v1_rd_data[/*idx1668=*/ 13][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 13][/*idx13=*/ 7][0] = tloop1836delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1859;
mult mult1860(v1859,
v1856,
v1858,
tloop1836delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1861 = v1667[/*idx1668=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1862;
add add1863(v1862,
v1859,
v1861,
tloop1836delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[14] = v1862;

//TerminatorOp

//} Unrolled body 13 of loop1668.
//DEBUG: /*idx1668=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop1668.
//DEBUG: /*idx1668=*/ 4'd14, expected 14
//printTimeOffset
reg tloop1850delay[3:0] = '{default:0} ;
always@(*) tloop1850delay[0] <= tloop1850;
generate
genvar i1865;

for(i1865 = 1; i1865<= 3; i1865= i1865 + 1) begin
always@(posedge clk) begin
tloop1850delay[i1865] <= tloop1850delay[i1865-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1864 = tloop1850delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1867[/*idx1668=*/ 14:0] = '{default:0};
always@(*) shiftreg1867[0] <= idx11;
always@(posedge clk) shiftreg1867[/*idx1668=*/ 14:1] <= shiftreg1867[/*idx1668=*/ 13:0];
wire [31:0] v1866 = shiftreg1867[/*idx1668=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 14][7] = tloop11delay[14];
assign v0_addr_input[/*idx1668=*/ 14][7] = {v1866[3:0]};
wire[31:0] v1868 = v0_rd_data[/*idx1668=*/ 14];
assign v0_rd_en_input[/*idx1668=*/ 14][7] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1869 = /*idx1668=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1871[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1871[0] <= v1868;
always@(posedge clk) shiftreg1871[/*idx13=*/ 7:1] <= shiftreg1871[/*idx13=*/ 6:0];
wire [31:0] v1870 = shiftreg1871[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1872 = v1_rd_data[/*idx1668=*/ 14][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 14][/*idx13=*/ 7][0] = tloop1850delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1873;
mult mult1874(v1873,
v1870,
v1872,
tloop1850delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1875 = v1667[/*idx1668=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1876;
add add1877(v1876,
v1873,
v1875,
tloop1850delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[15] = v1876;

//TerminatorOp

//} Unrolled body 14 of loop1668.
//DEBUG: /*idx1668=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop1668.
//DEBUG: /*idx1668=*/ 4'd15, expected 15
//printTimeOffset
reg tloop1864delay[3:0] = '{default:0} ;
always@(*) tloop1864delay[0] <= tloop1864;
generate
genvar i1879;

for(i1879 = 1; i1879<= 3; i1879= i1879 + 1) begin
always@(posedge clk) begin
tloop1864delay[i1879] <= tloop1864delay[i1879-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1878 = tloop1864delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1881[/*idx1668=*/ 15:0] = '{default:0};
always@(*) shiftreg1881[0] <= idx11;
always@(posedge clk) shiftreg1881[/*idx1668=*/ 15:1] <= shiftreg1881[/*idx1668=*/ 14:0];
wire [31:0] v1880 = shiftreg1881[/*idx1668=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1668=*/ 15][7] = tloop11delay[15];
assign v0_addr_input[/*idx1668=*/ 15][7] = {v1880[3:0]};
wire[31:0] v1882 = v0_rd_data[/*idx1668=*/ 15];
assign v0_rd_en_input[/*idx1668=*/ 15][7] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1883 = /*idx1668=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1885[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1885[0] <= v1882;
always@(posedge clk) shiftreg1885[/*idx13=*/ 7:1] <= shiftreg1885[/*idx13=*/ 6:0];
wire [31:0] v1884 = shiftreg1885[/*idx13=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1886 = v1_rd_data[/*idx1668=*/ 15][/*idx13=*/ 7];
assign v1_rd_en_input[/*idx1668=*/ 15][/*idx13=*/ 7][0] = tloop1864delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1887;
mult mult1888(v1887,
v1884,
v1886,
tloop1864delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1889 = v1667[/*idx1668=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1890;
add add1891(v1890,
v1887,
v1889,
tloop1864delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1667[16] = v1890;

//TerminatorOp

//} Unrolled body 15 of loop1668.
//DEBUG: /*idx1668=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t1892;
assign t1892 = tloop1878;
//printTimeOffset
reg t1892delay[3:0] = '{default:0} ;
always@(*) t1892delay[0] <= t1892;
generate
genvar i1893;

for(i1893 = 1; i1893<= 3; i1893= i1893 + 1) begin
always@(posedge clk) begin
t1892delay[i1893] <= t1892delay[i1893-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v1894 = v1667[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg1896[/*idx13=*/ 7:0] = '{default:0};
always@(*) shiftreg1896[0] <= idx11;
always@(posedge clk) shiftreg1896[/*idx13=*/ 7:1] <= shiftreg1896[/*idx13=*/ 6:0];
wire [31:0] v1895 = shiftreg1896[/*idx13=*/ 7];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg1898[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg1898[0] <= v1895;
always@(posedge clk) shiftreg1898[/*v10=*/ 16:1] <= shiftreg1898[/*v10=*/ 15:0];
wire [31:0] v1897 = shiftreg1898[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg1900[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg1900[0] <= v1897;
always@(posedge clk) shiftreg1900[/*v8=*/ 3:1] <= shiftreg1900[/*v8=*/ 2:0];
wire [31:0] v1899 = shiftreg1900[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 7][0] = t1892delay[3];
assign v2_addr_input[/*idx13=*/ 7][0] = {v1899[3:0]};
assign v2_wr_en_input[/*idx13=*/ 7][0] = t1892delay[3];
assign v2_wr_data_valid[/*idx13=*/ 7][0] = t1892delay[3];
assign v2_wr_data_input[/*idx13=*/ 7][0] = v1894;


//TerminatorOp

//} Unrolled body 7 of loop13.
//DEBUG: /*idx13=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop13.
//DEBUG: /*idx13=*/ 4'd8, expected 8
//printTimeOffset
reg tloop1665delay[3:0] = '{default:0} ;
always@(*) tloop1665delay[0] <= tloop1665;
generate
genvar i1902;

for(i1902 = 1; i1902<= 3; i1902= i1902 + 1) begin
always@(posedge clk) begin
tloop1665delay[i1902] <= tloop1665delay[i1902-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop1901 = tloop1665delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v1903[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v1903[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop1904.
//DEBUG: /*idx1904=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1905 = tloop1665delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1907[/*idx1904=*/ 0:0] = '{default:0};
always@(*) shiftreg1907[0] <= idx11;
wire [31:0] v1906 = shiftreg1907[/*idx1904=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 0][8] = tloop11delay[0];
assign v0_addr_input[/*idx1904=*/ 0][8] = {v1906[3:0]};
wire[31:0] v1908 = v0_rd_data[/*idx1904=*/ 0];
assign v0_rd_en_input[/*idx1904=*/ 0][8] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1909 = /*idx1904=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1911[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg1911[0] <= v1908;
always@(posedge clk) shiftreg1911[/*idx13=*/ 8:1] <= shiftreg1911[/*idx13=*/ 7:0];
wire [31:0] v1910 = shiftreg1911[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1912 = v1_rd_data[/*idx1904=*/ 0][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 0][/*idx13=*/ 8][0] = tloop1665delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1913;
mult mult1914(v1913,
v1910,
v1912,
tloop1665delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1915 = v1903[/*idx1904=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1916;
add add1917(v1916,
v1913,
v1915,
tloop1665delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[1] = v1916;

//TerminatorOp

//} Unrolled body 0 of loop1904.
//DEBUG: /*idx1904=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop1904.
//DEBUG: /*idx1904=*/ 1'd1, expected 1
//printTimeOffset
reg tloop1905delay[3:0] = '{default:0} ;
always@(*) tloop1905delay[0] <= tloop1905;
generate
genvar i1919;

for(i1919 = 1; i1919<= 3; i1919= i1919 + 1) begin
always@(posedge clk) begin
tloop1905delay[i1919] <= tloop1905delay[i1919-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1918 = tloop1905delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1921[/*idx1904=*/ 1:0] = '{default:0};
always@(*) shiftreg1921[0] <= idx11;
always@(posedge clk) shiftreg1921[/*idx1904=*/ 1:1] <= shiftreg1921[/*idx1904=*/ 0:0];
wire [31:0] v1920 = shiftreg1921[/*idx1904=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 1][8] = tloop11delay[1];
assign v0_addr_input[/*idx1904=*/ 1][8] = {v1920[3:0]};
wire[31:0] v1922 = v0_rd_data[/*idx1904=*/ 1];
assign v0_rd_en_input[/*idx1904=*/ 1][8] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1923 = /*idx1904=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1925[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg1925[0] <= v1922;
always@(posedge clk) shiftreg1925[/*idx13=*/ 8:1] <= shiftreg1925[/*idx13=*/ 7:0];
wire [31:0] v1924 = shiftreg1925[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1926 = v1_rd_data[/*idx1904=*/ 1][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 1][/*idx13=*/ 8][0] = tloop1905delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1927;
mult mult1928(v1927,
v1924,
v1926,
tloop1905delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1929 = v1903[/*idx1904=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1930;
add add1931(v1930,
v1927,
v1929,
tloop1905delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[2] = v1930;

//TerminatorOp

//} Unrolled body 1 of loop1904.
//DEBUG: /*idx1904=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop1904.
//DEBUG: /*idx1904=*/ 2'd2, expected 2
//printTimeOffset
reg tloop1918delay[3:0] = '{default:0} ;
always@(*) tloop1918delay[0] <= tloop1918;
generate
genvar i1933;

for(i1933 = 1; i1933<= 3; i1933= i1933 + 1) begin
always@(posedge clk) begin
tloop1918delay[i1933] <= tloop1918delay[i1933-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1932 = tloop1918delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1935[/*idx1904=*/ 2:0] = '{default:0};
always@(*) shiftreg1935[0] <= idx11;
always@(posedge clk) shiftreg1935[/*idx1904=*/ 2:1] <= shiftreg1935[/*idx1904=*/ 1:0];
wire [31:0] v1934 = shiftreg1935[/*idx1904=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 2][8] = tloop11delay[2];
assign v0_addr_input[/*idx1904=*/ 2][8] = {v1934[3:0]};
wire[31:0] v1936 = v0_rd_data[/*idx1904=*/ 2];
assign v0_rd_en_input[/*idx1904=*/ 2][8] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1937 = /*idx1904=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1939[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg1939[0] <= v1936;
always@(posedge clk) shiftreg1939[/*idx13=*/ 8:1] <= shiftreg1939[/*idx13=*/ 7:0];
wire [31:0] v1938 = shiftreg1939[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1940 = v1_rd_data[/*idx1904=*/ 2][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 2][/*idx13=*/ 8][0] = tloop1918delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1941;
mult mult1942(v1941,
v1938,
v1940,
tloop1918delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1943 = v1903[/*idx1904=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1944;
add add1945(v1944,
v1941,
v1943,
tloop1918delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[3] = v1944;

//TerminatorOp

//} Unrolled body 2 of loop1904.
//DEBUG: /*idx1904=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop1904.
//DEBUG: /*idx1904=*/ 2'd3, expected 3
//printTimeOffset
reg tloop1932delay[3:0] = '{default:0} ;
always@(*) tloop1932delay[0] <= tloop1932;
generate
genvar i1947;

for(i1947 = 1; i1947<= 3; i1947= i1947 + 1) begin
always@(posedge clk) begin
tloop1932delay[i1947] <= tloop1932delay[i1947-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1946 = tloop1932delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1949[/*idx1904=*/ 3:0] = '{default:0};
always@(*) shiftreg1949[0] <= idx11;
always@(posedge clk) shiftreg1949[/*idx1904=*/ 3:1] <= shiftreg1949[/*idx1904=*/ 2:0];
wire [31:0] v1948 = shiftreg1949[/*idx1904=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 3][8] = tloop11delay[3];
assign v0_addr_input[/*idx1904=*/ 3][8] = {v1948[3:0]};
wire[31:0] v1950 = v0_rd_data[/*idx1904=*/ 3];
assign v0_rd_en_input[/*idx1904=*/ 3][8] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1951 = /*idx1904=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1953[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg1953[0] <= v1950;
always@(posedge clk) shiftreg1953[/*idx13=*/ 8:1] <= shiftreg1953[/*idx13=*/ 7:0];
wire [31:0] v1952 = shiftreg1953[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1954 = v1_rd_data[/*idx1904=*/ 3][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 3][/*idx13=*/ 8][0] = tloop1932delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1955;
mult mult1956(v1955,
v1952,
v1954,
tloop1932delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1957 = v1903[/*idx1904=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1958;
add add1959(v1958,
v1955,
v1957,
tloop1932delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[4] = v1958;

//TerminatorOp

//} Unrolled body 3 of loop1904.
//DEBUG: /*idx1904=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop1904.
//DEBUG: /*idx1904=*/ 3'd4, expected 4
//printTimeOffset
reg tloop1946delay[3:0] = '{default:0} ;
always@(*) tloop1946delay[0] <= tloop1946;
generate
genvar i1961;

for(i1961 = 1; i1961<= 3; i1961= i1961 + 1) begin
always@(posedge clk) begin
tloop1946delay[i1961] <= tloop1946delay[i1961-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1960 = tloop1946delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1963[/*idx1904=*/ 4:0] = '{default:0};
always@(*) shiftreg1963[0] <= idx11;
always@(posedge clk) shiftreg1963[/*idx1904=*/ 4:1] <= shiftreg1963[/*idx1904=*/ 3:0];
wire [31:0] v1962 = shiftreg1963[/*idx1904=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 4][8] = tloop11delay[4];
assign v0_addr_input[/*idx1904=*/ 4][8] = {v1962[3:0]};
wire[31:0] v1964 = v0_rd_data[/*idx1904=*/ 4];
assign v0_rd_en_input[/*idx1904=*/ 4][8] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1965 = /*idx1904=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1967[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg1967[0] <= v1964;
always@(posedge clk) shiftreg1967[/*idx13=*/ 8:1] <= shiftreg1967[/*idx13=*/ 7:0];
wire [31:0] v1966 = shiftreg1967[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1968 = v1_rd_data[/*idx1904=*/ 4][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 4][/*idx13=*/ 8][0] = tloop1946delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1969;
mult mult1970(v1969,
v1966,
v1968,
tloop1946delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1971 = v1903[/*idx1904=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1972;
add add1973(v1972,
v1969,
v1971,
tloop1946delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[5] = v1972;

//TerminatorOp

//} Unrolled body 4 of loop1904.
//DEBUG: /*idx1904=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop1904.
//DEBUG: /*idx1904=*/ 3'd5, expected 5
//printTimeOffset
reg tloop1960delay[3:0] = '{default:0} ;
always@(*) tloop1960delay[0] <= tloop1960;
generate
genvar i1975;

for(i1975 = 1; i1975<= 3; i1975= i1975 + 1) begin
always@(posedge clk) begin
tloop1960delay[i1975] <= tloop1960delay[i1975-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1974 = tloop1960delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1977[/*idx1904=*/ 5:0] = '{default:0};
always@(*) shiftreg1977[0] <= idx11;
always@(posedge clk) shiftreg1977[/*idx1904=*/ 5:1] <= shiftreg1977[/*idx1904=*/ 4:0];
wire [31:0] v1976 = shiftreg1977[/*idx1904=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 5][8] = tloop11delay[5];
assign v0_addr_input[/*idx1904=*/ 5][8] = {v1976[3:0]};
wire[31:0] v1978 = v0_rd_data[/*idx1904=*/ 5];
assign v0_rd_en_input[/*idx1904=*/ 5][8] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1979 = /*idx1904=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1981[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg1981[0] <= v1978;
always@(posedge clk) shiftreg1981[/*idx13=*/ 8:1] <= shiftreg1981[/*idx13=*/ 7:0];
wire [31:0] v1980 = shiftreg1981[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1982 = v1_rd_data[/*idx1904=*/ 5][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 5][/*idx13=*/ 8][0] = tloop1960delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1983;
mult mult1984(v1983,
v1980,
v1982,
tloop1960delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1985 = v1903[/*idx1904=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v1986;
add add1987(v1986,
v1983,
v1985,
tloop1960delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[6] = v1986;

//TerminatorOp

//} Unrolled body 5 of loop1904.
//DEBUG: /*idx1904=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop1904.
//DEBUG: /*idx1904=*/ 3'd6, expected 6
//printTimeOffset
reg tloop1974delay[3:0] = '{default:0} ;
always@(*) tloop1974delay[0] <= tloop1974;
generate
genvar i1989;

for(i1989 = 1; i1989<= 3; i1989= i1989 + 1) begin
always@(posedge clk) begin
tloop1974delay[i1989] <= tloop1974delay[i1989-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop1988 = tloop1974delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg1991[/*idx1904=*/ 6:0] = '{default:0};
always@(*) shiftreg1991[0] <= idx11;
always@(posedge clk) shiftreg1991[/*idx1904=*/ 6:1] <= shiftreg1991[/*idx1904=*/ 5:0];
wire [31:0] v1990 = shiftreg1991[/*idx1904=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 6][8] = tloop11delay[6];
assign v0_addr_input[/*idx1904=*/ 6][8] = {v1990[3:0]};
wire[31:0] v1992 = v0_rd_data[/*idx1904=*/ 6];
assign v0_rd_en_input[/*idx1904=*/ 6][8] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v1993 = /*idx1904=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg1995[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg1995[0] <= v1992;
always@(posedge clk) shiftreg1995[/*idx13=*/ 8:1] <= shiftreg1995[/*idx13=*/ 7:0];
wire [31:0] v1994 = shiftreg1995[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v1996 = v1_rd_data[/*idx1904=*/ 6][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 6][/*idx13=*/ 8][0] = tloop1974delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v1997;
mult mult1998(v1997,
v1994,
v1996,
tloop1974delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v1999 = v1903[/*idx1904=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2000;
add add2001(v2000,
v1997,
v1999,
tloop1974delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[7] = v2000;

//TerminatorOp

//} Unrolled body 6 of loop1904.
//DEBUG: /*idx1904=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop1904.
//DEBUG: /*idx1904=*/ 3'd7, expected 7
//printTimeOffset
reg tloop1988delay[3:0] = '{default:0} ;
always@(*) tloop1988delay[0] <= tloop1988;
generate
genvar i2003;

for(i2003 = 1; i2003<= 3; i2003= i2003 + 1) begin
always@(posedge clk) begin
tloop1988delay[i2003] <= tloop1988delay[i2003-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2002 = tloop1988delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2005[/*idx1904=*/ 7:0] = '{default:0};
always@(*) shiftreg2005[0] <= idx11;
always@(posedge clk) shiftreg2005[/*idx1904=*/ 7:1] <= shiftreg2005[/*idx1904=*/ 6:0];
wire [31:0] v2004 = shiftreg2005[/*idx1904=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 7][8] = tloop11delay[7];
assign v0_addr_input[/*idx1904=*/ 7][8] = {v2004[3:0]};
wire[31:0] v2006 = v0_rd_data[/*idx1904=*/ 7];
assign v0_rd_en_input[/*idx1904=*/ 7][8] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2007 = /*idx1904=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2009[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2009[0] <= v2006;
always@(posedge clk) shiftreg2009[/*idx13=*/ 8:1] <= shiftreg2009[/*idx13=*/ 7:0];
wire [31:0] v2008 = shiftreg2009[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2010 = v1_rd_data[/*idx1904=*/ 7][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 7][/*idx13=*/ 8][0] = tloop1988delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2011;
mult mult2012(v2011,
v2008,
v2010,
tloop1988delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2013 = v1903[/*idx1904=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2014;
add add2015(v2014,
v2011,
v2013,
tloop1988delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[8] = v2014;

//TerminatorOp

//} Unrolled body 7 of loop1904.
//DEBUG: /*idx1904=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop1904.
//DEBUG: /*idx1904=*/ 4'd8, expected 8
//printTimeOffset
reg tloop2002delay[3:0] = '{default:0} ;
always@(*) tloop2002delay[0] <= tloop2002;
generate
genvar i2017;

for(i2017 = 1; i2017<= 3; i2017= i2017 + 1) begin
always@(posedge clk) begin
tloop2002delay[i2017] <= tloop2002delay[i2017-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2016 = tloop2002delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2019[/*idx1904=*/ 8:0] = '{default:0};
always@(*) shiftreg2019[0] <= idx11;
always@(posedge clk) shiftreg2019[/*idx1904=*/ 8:1] <= shiftreg2019[/*idx1904=*/ 7:0];
wire [31:0] v2018 = shiftreg2019[/*idx1904=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 8][8] = tloop11delay[8];
assign v0_addr_input[/*idx1904=*/ 8][8] = {v2018[3:0]};
wire[31:0] v2020 = v0_rd_data[/*idx1904=*/ 8];
assign v0_rd_en_input[/*idx1904=*/ 8][8] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2021 = /*idx1904=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2023[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2023[0] <= v2020;
always@(posedge clk) shiftreg2023[/*idx13=*/ 8:1] <= shiftreg2023[/*idx13=*/ 7:0];
wire [31:0] v2022 = shiftreg2023[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2024 = v1_rd_data[/*idx1904=*/ 8][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 8][/*idx13=*/ 8][0] = tloop2002delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2025;
mult mult2026(v2025,
v2022,
v2024,
tloop2002delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2027 = v1903[/*idx1904=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2028;
add add2029(v2028,
v2025,
v2027,
tloop2002delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[9] = v2028;

//TerminatorOp

//} Unrolled body 8 of loop1904.
//DEBUG: /*idx1904=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop1904.
//DEBUG: /*idx1904=*/ 4'd9, expected 9
//printTimeOffset
reg tloop2016delay[3:0] = '{default:0} ;
always@(*) tloop2016delay[0] <= tloop2016;
generate
genvar i2031;

for(i2031 = 1; i2031<= 3; i2031= i2031 + 1) begin
always@(posedge clk) begin
tloop2016delay[i2031] <= tloop2016delay[i2031-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2030 = tloop2016delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2033[/*idx1904=*/ 9:0] = '{default:0};
always@(*) shiftreg2033[0] <= idx11;
always@(posedge clk) shiftreg2033[/*idx1904=*/ 9:1] <= shiftreg2033[/*idx1904=*/ 8:0];
wire [31:0] v2032 = shiftreg2033[/*idx1904=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 9][8] = tloop11delay[9];
assign v0_addr_input[/*idx1904=*/ 9][8] = {v2032[3:0]};
wire[31:0] v2034 = v0_rd_data[/*idx1904=*/ 9];
assign v0_rd_en_input[/*idx1904=*/ 9][8] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2035 = /*idx1904=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2037[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2037[0] <= v2034;
always@(posedge clk) shiftreg2037[/*idx13=*/ 8:1] <= shiftreg2037[/*idx13=*/ 7:0];
wire [31:0] v2036 = shiftreg2037[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2038 = v1_rd_data[/*idx1904=*/ 9][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 9][/*idx13=*/ 8][0] = tloop2016delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2039;
mult mult2040(v2039,
v2036,
v2038,
tloop2016delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2041 = v1903[/*idx1904=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2042;
add add2043(v2042,
v2039,
v2041,
tloop2016delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[10] = v2042;

//TerminatorOp

//} Unrolled body 9 of loop1904.
//DEBUG: /*idx1904=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop1904.
//DEBUG: /*idx1904=*/ 4'd10, expected 10
//printTimeOffset
reg tloop2030delay[3:0] = '{default:0} ;
always@(*) tloop2030delay[0] <= tloop2030;
generate
genvar i2045;

for(i2045 = 1; i2045<= 3; i2045= i2045 + 1) begin
always@(posedge clk) begin
tloop2030delay[i2045] <= tloop2030delay[i2045-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2044 = tloop2030delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2047[/*idx1904=*/ 10:0] = '{default:0};
always@(*) shiftreg2047[0] <= idx11;
always@(posedge clk) shiftreg2047[/*idx1904=*/ 10:1] <= shiftreg2047[/*idx1904=*/ 9:0];
wire [31:0] v2046 = shiftreg2047[/*idx1904=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 10][8] = tloop11delay[10];
assign v0_addr_input[/*idx1904=*/ 10][8] = {v2046[3:0]};
wire[31:0] v2048 = v0_rd_data[/*idx1904=*/ 10];
assign v0_rd_en_input[/*idx1904=*/ 10][8] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2049 = /*idx1904=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2051[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2051[0] <= v2048;
always@(posedge clk) shiftreg2051[/*idx13=*/ 8:1] <= shiftreg2051[/*idx13=*/ 7:0];
wire [31:0] v2050 = shiftreg2051[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2052 = v1_rd_data[/*idx1904=*/ 10][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 10][/*idx13=*/ 8][0] = tloop2030delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2053;
mult mult2054(v2053,
v2050,
v2052,
tloop2030delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2055 = v1903[/*idx1904=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2056;
add add2057(v2056,
v2053,
v2055,
tloop2030delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[11] = v2056;

//TerminatorOp

//} Unrolled body 10 of loop1904.
//DEBUG: /*idx1904=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop1904.
//DEBUG: /*idx1904=*/ 4'd11, expected 11
//printTimeOffset
reg tloop2044delay[3:0] = '{default:0} ;
always@(*) tloop2044delay[0] <= tloop2044;
generate
genvar i2059;

for(i2059 = 1; i2059<= 3; i2059= i2059 + 1) begin
always@(posedge clk) begin
tloop2044delay[i2059] <= tloop2044delay[i2059-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2058 = tloop2044delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2061[/*idx1904=*/ 11:0] = '{default:0};
always@(*) shiftreg2061[0] <= idx11;
always@(posedge clk) shiftreg2061[/*idx1904=*/ 11:1] <= shiftreg2061[/*idx1904=*/ 10:0];
wire [31:0] v2060 = shiftreg2061[/*idx1904=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 11][8] = tloop11delay[11];
assign v0_addr_input[/*idx1904=*/ 11][8] = {v2060[3:0]};
wire[31:0] v2062 = v0_rd_data[/*idx1904=*/ 11];
assign v0_rd_en_input[/*idx1904=*/ 11][8] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2063 = /*idx1904=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2065[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2065[0] <= v2062;
always@(posedge clk) shiftreg2065[/*idx13=*/ 8:1] <= shiftreg2065[/*idx13=*/ 7:0];
wire [31:0] v2064 = shiftreg2065[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2066 = v1_rd_data[/*idx1904=*/ 11][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 11][/*idx13=*/ 8][0] = tloop2044delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2067;
mult mult2068(v2067,
v2064,
v2066,
tloop2044delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2069 = v1903[/*idx1904=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2070;
add add2071(v2070,
v2067,
v2069,
tloop2044delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[12] = v2070;

//TerminatorOp

//} Unrolled body 11 of loop1904.
//DEBUG: /*idx1904=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop1904.
//DEBUG: /*idx1904=*/ 4'd12, expected 12
//printTimeOffset
reg tloop2058delay[3:0] = '{default:0} ;
always@(*) tloop2058delay[0] <= tloop2058;
generate
genvar i2073;

for(i2073 = 1; i2073<= 3; i2073= i2073 + 1) begin
always@(posedge clk) begin
tloop2058delay[i2073] <= tloop2058delay[i2073-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2072 = tloop2058delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2075[/*idx1904=*/ 12:0] = '{default:0};
always@(*) shiftreg2075[0] <= idx11;
always@(posedge clk) shiftreg2075[/*idx1904=*/ 12:1] <= shiftreg2075[/*idx1904=*/ 11:0];
wire [31:0] v2074 = shiftreg2075[/*idx1904=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 12][8] = tloop11delay[12];
assign v0_addr_input[/*idx1904=*/ 12][8] = {v2074[3:0]};
wire[31:0] v2076 = v0_rd_data[/*idx1904=*/ 12];
assign v0_rd_en_input[/*idx1904=*/ 12][8] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2077 = /*idx1904=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2079[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2079[0] <= v2076;
always@(posedge clk) shiftreg2079[/*idx13=*/ 8:1] <= shiftreg2079[/*idx13=*/ 7:0];
wire [31:0] v2078 = shiftreg2079[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2080 = v1_rd_data[/*idx1904=*/ 12][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 12][/*idx13=*/ 8][0] = tloop2058delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2081;
mult mult2082(v2081,
v2078,
v2080,
tloop2058delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2083 = v1903[/*idx1904=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2084;
add add2085(v2084,
v2081,
v2083,
tloop2058delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[13] = v2084;

//TerminatorOp

//} Unrolled body 12 of loop1904.
//DEBUG: /*idx1904=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop1904.
//DEBUG: /*idx1904=*/ 4'd13, expected 13
//printTimeOffset
reg tloop2072delay[3:0] = '{default:0} ;
always@(*) tloop2072delay[0] <= tloop2072;
generate
genvar i2087;

for(i2087 = 1; i2087<= 3; i2087= i2087 + 1) begin
always@(posedge clk) begin
tloop2072delay[i2087] <= tloop2072delay[i2087-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2086 = tloop2072delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2089[/*idx1904=*/ 13:0] = '{default:0};
always@(*) shiftreg2089[0] <= idx11;
always@(posedge clk) shiftreg2089[/*idx1904=*/ 13:1] <= shiftreg2089[/*idx1904=*/ 12:0];
wire [31:0] v2088 = shiftreg2089[/*idx1904=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 13][8] = tloop11delay[13];
assign v0_addr_input[/*idx1904=*/ 13][8] = {v2088[3:0]};
wire[31:0] v2090 = v0_rd_data[/*idx1904=*/ 13];
assign v0_rd_en_input[/*idx1904=*/ 13][8] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2091 = /*idx1904=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2093[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2093[0] <= v2090;
always@(posedge clk) shiftreg2093[/*idx13=*/ 8:1] <= shiftreg2093[/*idx13=*/ 7:0];
wire [31:0] v2092 = shiftreg2093[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2094 = v1_rd_data[/*idx1904=*/ 13][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 13][/*idx13=*/ 8][0] = tloop2072delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2095;
mult mult2096(v2095,
v2092,
v2094,
tloop2072delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2097 = v1903[/*idx1904=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2098;
add add2099(v2098,
v2095,
v2097,
tloop2072delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[14] = v2098;

//TerminatorOp

//} Unrolled body 13 of loop1904.
//DEBUG: /*idx1904=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop1904.
//DEBUG: /*idx1904=*/ 4'd14, expected 14
//printTimeOffset
reg tloop2086delay[3:0] = '{default:0} ;
always@(*) tloop2086delay[0] <= tloop2086;
generate
genvar i2101;

for(i2101 = 1; i2101<= 3; i2101= i2101 + 1) begin
always@(posedge clk) begin
tloop2086delay[i2101] <= tloop2086delay[i2101-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2100 = tloop2086delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2103[/*idx1904=*/ 14:0] = '{default:0};
always@(*) shiftreg2103[0] <= idx11;
always@(posedge clk) shiftreg2103[/*idx1904=*/ 14:1] <= shiftreg2103[/*idx1904=*/ 13:0];
wire [31:0] v2102 = shiftreg2103[/*idx1904=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 14][8] = tloop11delay[14];
assign v0_addr_input[/*idx1904=*/ 14][8] = {v2102[3:0]};
wire[31:0] v2104 = v0_rd_data[/*idx1904=*/ 14];
assign v0_rd_en_input[/*idx1904=*/ 14][8] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2105 = /*idx1904=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2107[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2107[0] <= v2104;
always@(posedge clk) shiftreg2107[/*idx13=*/ 8:1] <= shiftreg2107[/*idx13=*/ 7:0];
wire [31:0] v2106 = shiftreg2107[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2108 = v1_rd_data[/*idx1904=*/ 14][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 14][/*idx13=*/ 8][0] = tloop2086delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2109;
mult mult2110(v2109,
v2106,
v2108,
tloop2086delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2111 = v1903[/*idx1904=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2112;
add add2113(v2112,
v2109,
v2111,
tloop2086delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[15] = v2112;

//TerminatorOp

//} Unrolled body 14 of loop1904.
//DEBUG: /*idx1904=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop1904.
//DEBUG: /*idx1904=*/ 4'd15, expected 15
//printTimeOffset
reg tloop2100delay[3:0] = '{default:0} ;
always@(*) tloop2100delay[0] <= tloop2100;
generate
genvar i2115;

for(i2115 = 1; i2115<= 3; i2115= i2115 + 1) begin
always@(posedge clk) begin
tloop2100delay[i2115] <= tloop2100delay[i2115-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2114 = tloop2100delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2117[/*idx1904=*/ 15:0] = '{default:0};
always@(*) shiftreg2117[0] <= idx11;
always@(posedge clk) shiftreg2117[/*idx1904=*/ 15:1] <= shiftreg2117[/*idx1904=*/ 14:0];
wire [31:0] v2116 = shiftreg2117[/*idx1904=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx1904=*/ 15][8] = tloop11delay[15];
assign v0_addr_input[/*idx1904=*/ 15][8] = {v2116[3:0]};
wire[31:0] v2118 = v0_rd_data[/*idx1904=*/ 15];
assign v0_rd_en_input[/*idx1904=*/ 15][8] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2119 = /*idx1904=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2121[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2121[0] <= v2118;
always@(posedge clk) shiftreg2121[/*idx13=*/ 8:1] <= shiftreg2121[/*idx13=*/ 7:0];
wire [31:0] v2120 = shiftreg2121[/*idx13=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2122 = v1_rd_data[/*idx1904=*/ 15][/*idx13=*/ 8];
assign v1_rd_en_input[/*idx1904=*/ 15][/*idx13=*/ 8][0] = tloop2100delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2123;
mult mult2124(v2123,
v2120,
v2122,
tloop2100delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2125 = v1903[/*idx1904=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2126;
add add2127(v2126,
v2123,
v2125,
tloop2100delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v1903[16] = v2126;

//TerminatorOp

//} Unrolled body 15 of loop1904.
//DEBUG: /*idx1904=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t2128;
assign t2128 = tloop2114;
//printTimeOffset
reg t2128delay[3:0] = '{default:0} ;
always@(*) t2128delay[0] <= t2128;
generate
genvar i2129;

for(i2129 = 1; i2129<= 3; i2129= i2129 + 1) begin
always@(posedge clk) begin
t2128delay[i2129] <= t2128delay[i2129-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v2130 = v1903[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg2132[/*idx13=*/ 8:0] = '{default:0};
always@(*) shiftreg2132[0] <= idx11;
always@(posedge clk) shiftreg2132[/*idx13=*/ 8:1] <= shiftreg2132[/*idx13=*/ 7:0];
wire [31:0] v2131 = shiftreg2132[/*idx13=*/ 8];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg2134[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg2134[0] <= v2131;
always@(posedge clk) shiftreg2134[/*v10=*/ 16:1] <= shiftreg2134[/*v10=*/ 15:0];
wire [31:0] v2133 = shiftreg2134[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg2136[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg2136[0] <= v2133;
always@(posedge clk) shiftreg2136[/*v8=*/ 3:1] <= shiftreg2136[/*v8=*/ 2:0];
wire [31:0] v2135 = shiftreg2136[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 8][0] = t2128delay[3];
assign v2_addr_input[/*idx13=*/ 8][0] = {v2135[3:0]};
assign v2_wr_en_input[/*idx13=*/ 8][0] = t2128delay[3];
assign v2_wr_data_valid[/*idx13=*/ 8][0] = t2128delay[3];
assign v2_wr_data_input[/*idx13=*/ 8][0] = v2130;


//TerminatorOp

//} Unrolled body 8 of loop13.
//DEBUG: /*idx13=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop13.
//DEBUG: /*idx13=*/ 4'd9, expected 9
//printTimeOffset
reg tloop1901delay[3:0] = '{default:0} ;
always@(*) tloop1901delay[0] <= tloop1901;
generate
genvar i2138;

for(i2138 = 1; i2138<= 3; i2138= i2138 + 1) begin
always@(posedge clk) begin
tloop1901delay[i2138] <= tloop1901delay[i2138-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop2137 = tloop1901delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v2139[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v2139[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop2140.
//DEBUG: /*idx2140=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2141 = tloop1901delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2143[/*idx2140=*/ 0:0] = '{default:0};
always@(*) shiftreg2143[0] <= idx11;
wire [31:0] v2142 = shiftreg2143[/*idx2140=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 0][9] = tloop11delay[0];
assign v0_addr_input[/*idx2140=*/ 0][9] = {v2142[3:0]};
wire[31:0] v2144 = v0_rd_data[/*idx2140=*/ 0];
assign v0_rd_en_input[/*idx2140=*/ 0][9] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2145 = /*idx2140=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2147[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2147[0] <= v2144;
always@(posedge clk) shiftreg2147[/*idx13=*/ 9:1] <= shiftreg2147[/*idx13=*/ 8:0];
wire [31:0] v2146 = shiftreg2147[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2148 = v1_rd_data[/*idx2140=*/ 0][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 0][/*idx13=*/ 9][0] = tloop1901delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2149;
mult mult2150(v2149,
v2146,
v2148,
tloop1901delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2151 = v2139[/*idx2140=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2152;
add add2153(v2152,
v2149,
v2151,
tloop1901delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[1] = v2152;

//TerminatorOp

//} Unrolled body 0 of loop2140.
//DEBUG: /*idx2140=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop2140.
//DEBUG: /*idx2140=*/ 1'd1, expected 1
//printTimeOffset
reg tloop2141delay[3:0] = '{default:0} ;
always@(*) tloop2141delay[0] <= tloop2141;
generate
genvar i2155;

for(i2155 = 1; i2155<= 3; i2155= i2155 + 1) begin
always@(posedge clk) begin
tloop2141delay[i2155] <= tloop2141delay[i2155-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2154 = tloop2141delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2157[/*idx2140=*/ 1:0] = '{default:0};
always@(*) shiftreg2157[0] <= idx11;
always@(posedge clk) shiftreg2157[/*idx2140=*/ 1:1] <= shiftreg2157[/*idx2140=*/ 0:0];
wire [31:0] v2156 = shiftreg2157[/*idx2140=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 1][9] = tloop11delay[1];
assign v0_addr_input[/*idx2140=*/ 1][9] = {v2156[3:0]};
wire[31:0] v2158 = v0_rd_data[/*idx2140=*/ 1];
assign v0_rd_en_input[/*idx2140=*/ 1][9] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2159 = /*idx2140=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2161[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2161[0] <= v2158;
always@(posedge clk) shiftreg2161[/*idx13=*/ 9:1] <= shiftreg2161[/*idx13=*/ 8:0];
wire [31:0] v2160 = shiftreg2161[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2162 = v1_rd_data[/*idx2140=*/ 1][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 1][/*idx13=*/ 9][0] = tloop2141delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2163;
mult mult2164(v2163,
v2160,
v2162,
tloop2141delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2165 = v2139[/*idx2140=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2166;
add add2167(v2166,
v2163,
v2165,
tloop2141delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[2] = v2166;

//TerminatorOp

//} Unrolled body 1 of loop2140.
//DEBUG: /*idx2140=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop2140.
//DEBUG: /*idx2140=*/ 2'd2, expected 2
//printTimeOffset
reg tloop2154delay[3:0] = '{default:0} ;
always@(*) tloop2154delay[0] <= tloop2154;
generate
genvar i2169;

for(i2169 = 1; i2169<= 3; i2169= i2169 + 1) begin
always@(posedge clk) begin
tloop2154delay[i2169] <= tloop2154delay[i2169-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2168 = tloop2154delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2171[/*idx2140=*/ 2:0] = '{default:0};
always@(*) shiftreg2171[0] <= idx11;
always@(posedge clk) shiftreg2171[/*idx2140=*/ 2:1] <= shiftreg2171[/*idx2140=*/ 1:0];
wire [31:0] v2170 = shiftreg2171[/*idx2140=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 2][9] = tloop11delay[2];
assign v0_addr_input[/*idx2140=*/ 2][9] = {v2170[3:0]};
wire[31:0] v2172 = v0_rd_data[/*idx2140=*/ 2];
assign v0_rd_en_input[/*idx2140=*/ 2][9] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2173 = /*idx2140=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2175[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2175[0] <= v2172;
always@(posedge clk) shiftreg2175[/*idx13=*/ 9:1] <= shiftreg2175[/*idx13=*/ 8:0];
wire [31:0] v2174 = shiftreg2175[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2176 = v1_rd_data[/*idx2140=*/ 2][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 2][/*idx13=*/ 9][0] = tloop2154delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2177;
mult mult2178(v2177,
v2174,
v2176,
tloop2154delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2179 = v2139[/*idx2140=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2180;
add add2181(v2180,
v2177,
v2179,
tloop2154delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[3] = v2180;

//TerminatorOp

//} Unrolled body 2 of loop2140.
//DEBUG: /*idx2140=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop2140.
//DEBUG: /*idx2140=*/ 2'd3, expected 3
//printTimeOffset
reg tloop2168delay[3:0] = '{default:0} ;
always@(*) tloop2168delay[0] <= tloop2168;
generate
genvar i2183;

for(i2183 = 1; i2183<= 3; i2183= i2183 + 1) begin
always@(posedge clk) begin
tloop2168delay[i2183] <= tloop2168delay[i2183-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2182 = tloop2168delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2185[/*idx2140=*/ 3:0] = '{default:0};
always@(*) shiftreg2185[0] <= idx11;
always@(posedge clk) shiftreg2185[/*idx2140=*/ 3:1] <= shiftreg2185[/*idx2140=*/ 2:0];
wire [31:0] v2184 = shiftreg2185[/*idx2140=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 3][9] = tloop11delay[3];
assign v0_addr_input[/*idx2140=*/ 3][9] = {v2184[3:0]};
wire[31:0] v2186 = v0_rd_data[/*idx2140=*/ 3];
assign v0_rd_en_input[/*idx2140=*/ 3][9] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2187 = /*idx2140=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2189[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2189[0] <= v2186;
always@(posedge clk) shiftreg2189[/*idx13=*/ 9:1] <= shiftreg2189[/*idx13=*/ 8:0];
wire [31:0] v2188 = shiftreg2189[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2190 = v1_rd_data[/*idx2140=*/ 3][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 3][/*idx13=*/ 9][0] = tloop2168delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2191;
mult mult2192(v2191,
v2188,
v2190,
tloop2168delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2193 = v2139[/*idx2140=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2194;
add add2195(v2194,
v2191,
v2193,
tloop2168delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[4] = v2194;

//TerminatorOp

//} Unrolled body 3 of loop2140.
//DEBUG: /*idx2140=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop2140.
//DEBUG: /*idx2140=*/ 3'd4, expected 4
//printTimeOffset
reg tloop2182delay[3:0] = '{default:0} ;
always@(*) tloop2182delay[0] <= tloop2182;
generate
genvar i2197;

for(i2197 = 1; i2197<= 3; i2197= i2197 + 1) begin
always@(posedge clk) begin
tloop2182delay[i2197] <= tloop2182delay[i2197-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2196 = tloop2182delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2199[/*idx2140=*/ 4:0] = '{default:0};
always@(*) shiftreg2199[0] <= idx11;
always@(posedge clk) shiftreg2199[/*idx2140=*/ 4:1] <= shiftreg2199[/*idx2140=*/ 3:0];
wire [31:0] v2198 = shiftreg2199[/*idx2140=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 4][9] = tloop11delay[4];
assign v0_addr_input[/*idx2140=*/ 4][9] = {v2198[3:0]};
wire[31:0] v2200 = v0_rd_data[/*idx2140=*/ 4];
assign v0_rd_en_input[/*idx2140=*/ 4][9] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2201 = /*idx2140=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2203[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2203[0] <= v2200;
always@(posedge clk) shiftreg2203[/*idx13=*/ 9:1] <= shiftreg2203[/*idx13=*/ 8:0];
wire [31:0] v2202 = shiftreg2203[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2204 = v1_rd_data[/*idx2140=*/ 4][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 4][/*idx13=*/ 9][0] = tloop2182delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2205;
mult mult2206(v2205,
v2202,
v2204,
tloop2182delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2207 = v2139[/*idx2140=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2208;
add add2209(v2208,
v2205,
v2207,
tloop2182delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[5] = v2208;

//TerminatorOp

//} Unrolled body 4 of loop2140.
//DEBUG: /*idx2140=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop2140.
//DEBUG: /*idx2140=*/ 3'd5, expected 5
//printTimeOffset
reg tloop2196delay[3:0] = '{default:0} ;
always@(*) tloop2196delay[0] <= tloop2196;
generate
genvar i2211;

for(i2211 = 1; i2211<= 3; i2211= i2211 + 1) begin
always@(posedge clk) begin
tloop2196delay[i2211] <= tloop2196delay[i2211-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2210 = tloop2196delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2213[/*idx2140=*/ 5:0] = '{default:0};
always@(*) shiftreg2213[0] <= idx11;
always@(posedge clk) shiftreg2213[/*idx2140=*/ 5:1] <= shiftreg2213[/*idx2140=*/ 4:0];
wire [31:0] v2212 = shiftreg2213[/*idx2140=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 5][9] = tloop11delay[5];
assign v0_addr_input[/*idx2140=*/ 5][9] = {v2212[3:0]};
wire[31:0] v2214 = v0_rd_data[/*idx2140=*/ 5];
assign v0_rd_en_input[/*idx2140=*/ 5][9] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2215 = /*idx2140=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2217[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2217[0] <= v2214;
always@(posedge clk) shiftreg2217[/*idx13=*/ 9:1] <= shiftreg2217[/*idx13=*/ 8:0];
wire [31:0] v2216 = shiftreg2217[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2218 = v1_rd_data[/*idx2140=*/ 5][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 5][/*idx13=*/ 9][0] = tloop2196delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2219;
mult mult2220(v2219,
v2216,
v2218,
tloop2196delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2221 = v2139[/*idx2140=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2222;
add add2223(v2222,
v2219,
v2221,
tloop2196delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[6] = v2222;

//TerminatorOp

//} Unrolled body 5 of loop2140.
//DEBUG: /*idx2140=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop2140.
//DEBUG: /*idx2140=*/ 3'd6, expected 6
//printTimeOffset
reg tloop2210delay[3:0] = '{default:0} ;
always@(*) tloop2210delay[0] <= tloop2210;
generate
genvar i2225;

for(i2225 = 1; i2225<= 3; i2225= i2225 + 1) begin
always@(posedge clk) begin
tloop2210delay[i2225] <= tloop2210delay[i2225-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2224 = tloop2210delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2227[/*idx2140=*/ 6:0] = '{default:0};
always@(*) shiftreg2227[0] <= idx11;
always@(posedge clk) shiftreg2227[/*idx2140=*/ 6:1] <= shiftreg2227[/*idx2140=*/ 5:0];
wire [31:0] v2226 = shiftreg2227[/*idx2140=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 6][9] = tloop11delay[6];
assign v0_addr_input[/*idx2140=*/ 6][9] = {v2226[3:0]};
wire[31:0] v2228 = v0_rd_data[/*idx2140=*/ 6];
assign v0_rd_en_input[/*idx2140=*/ 6][9] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2229 = /*idx2140=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2231[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2231[0] <= v2228;
always@(posedge clk) shiftreg2231[/*idx13=*/ 9:1] <= shiftreg2231[/*idx13=*/ 8:0];
wire [31:0] v2230 = shiftreg2231[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2232 = v1_rd_data[/*idx2140=*/ 6][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 6][/*idx13=*/ 9][0] = tloop2210delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2233;
mult mult2234(v2233,
v2230,
v2232,
tloop2210delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2235 = v2139[/*idx2140=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2236;
add add2237(v2236,
v2233,
v2235,
tloop2210delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[7] = v2236;

//TerminatorOp

//} Unrolled body 6 of loop2140.
//DEBUG: /*idx2140=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop2140.
//DEBUG: /*idx2140=*/ 3'd7, expected 7
//printTimeOffset
reg tloop2224delay[3:0] = '{default:0} ;
always@(*) tloop2224delay[0] <= tloop2224;
generate
genvar i2239;

for(i2239 = 1; i2239<= 3; i2239= i2239 + 1) begin
always@(posedge clk) begin
tloop2224delay[i2239] <= tloop2224delay[i2239-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2238 = tloop2224delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2241[/*idx2140=*/ 7:0] = '{default:0};
always@(*) shiftreg2241[0] <= idx11;
always@(posedge clk) shiftreg2241[/*idx2140=*/ 7:1] <= shiftreg2241[/*idx2140=*/ 6:0];
wire [31:0] v2240 = shiftreg2241[/*idx2140=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 7][9] = tloop11delay[7];
assign v0_addr_input[/*idx2140=*/ 7][9] = {v2240[3:0]};
wire[31:0] v2242 = v0_rd_data[/*idx2140=*/ 7];
assign v0_rd_en_input[/*idx2140=*/ 7][9] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2243 = /*idx2140=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2245[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2245[0] <= v2242;
always@(posedge clk) shiftreg2245[/*idx13=*/ 9:1] <= shiftreg2245[/*idx13=*/ 8:0];
wire [31:0] v2244 = shiftreg2245[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2246 = v1_rd_data[/*idx2140=*/ 7][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 7][/*idx13=*/ 9][0] = tloop2224delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2247;
mult mult2248(v2247,
v2244,
v2246,
tloop2224delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2249 = v2139[/*idx2140=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2250;
add add2251(v2250,
v2247,
v2249,
tloop2224delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[8] = v2250;

//TerminatorOp

//} Unrolled body 7 of loop2140.
//DEBUG: /*idx2140=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop2140.
//DEBUG: /*idx2140=*/ 4'd8, expected 8
//printTimeOffset
reg tloop2238delay[3:0] = '{default:0} ;
always@(*) tloop2238delay[0] <= tloop2238;
generate
genvar i2253;

for(i2253 = 1; i2253<= 3; i2253= i2253 + 1) begin
always@(posedge clk) begin
tloop2238delay[i2253] <= tloop2238delay[i2253-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2252 = tloop2238delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2255[/*idx2140=*/ 8:0] = '{default:0};
always@(*) shiftreg2255[0] <= idx11;
always@(posedge clk) shiftreg2255[/*idx2140=*/ 8:1] <= shiftreg2255[/*idx2140=*/ 7:0];
wire [31:0] v2254 = shiftreg2255[/*idx2140=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 8][9] = tloop11delay[8];
assign v0_addr_input[/*idx2140=*/ 8][9] = {v2254[3:0]};
wire[31:0] v2256 = v0_rd_data[/*idx2140=*/ 8];
assign v0_rd_en_input[/*idx2140=*/ 8][9] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2257 = /*idx2140=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2259[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2259[0] <= v2256;
always@(posedge clk) shiftreg2259[/*idx13=*/ 9:1] <= shiftreg2259[/*idx13=*/ 8:0];
wire [31:0] v2258 = shiftreg2259[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2260 = v1_rd_data[/*idx2140=*/ 8][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 8][/*idx13=*/ 9][0] = tloop2238delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2261;
mult mult2262(v2261,
v2258,
v2260,
tloop2238delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2263 = v2139[/*idx2140=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2264;
add add2265(v2264,
v2261,
v2263,
tloop2238delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[9] = v2264;

//TerminatorOp

//} Unrolled body 8 of loop2140.
//DEBUG: /*idx2140=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop2140.
//DEBUG: /*idx2140=*/ 4'd9, expected 9
//printTimeOffset
reg tloop2252delay[3:0] = '{default:0} ;
always@(*) tloop2252delay[0] <= tloop2252;
generate
genvar i2267;

for(i2267 = 1; i2267<= 3; i2267= i2267 + 1) begin
always@(posedge clk) begin
tloop2252delay[i2267] <= tloop2252delay[i2267-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2266 = tloop2252delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2269[/*idx2140=*/ 9:0] = '{default:0};
always@(*) shiftreg2269[0] <= idx11;
always@(posedge clk) shiftreg2269[/*idx2140=*/ 9:1] <= shiftreg2269[/*idx2140=*/ 8:0];
wire [31:0] v2268 = shiftreg2269[/*idx2140=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 9][9] = tloop11delay[9];
assign v0_addr_input[/*idx2140=*/ 9][9] = {v2268[3:0]};
wire[31:0] v2270 = v0_rd_data[/*idx2140=*/ 9];
assign v0_rd_en_input[/*idx2140=*/ 9][9] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2271 = /*idx2140=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2273[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2273[0] <= v2270;
always@(posedge clk) shiftreg2273[/*idx13=*/ 9:1] <= shiftreg2273[/*idx13=*/ 8:0];
wire [31:0] v2272 = shiftreg2273[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2274 = v1_rd_data[/*idx2140=*/ 9][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 9][/*idx13=*/ 9][0] = tloop2252delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2275;
mult mult2276(v2275,
v2272,
v2274,
tloop2252delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2277 = v2139[/*idx2140=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2278;
add add2279(v2278,
v2275,
v2277,
tloop2252delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[10] = v2278;

//TerminatorOp

//} Unrolled body 9 of loop2140.
//DEBUG: /*idx2140=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop2140.
//DEBUG: /*idx2140=*/ 4'd10, expected 10
//printTimeOffset
reg tloop2266delay[3:0] = '{default:0} ;
always@(*) tloop2266delay[0] <= tloop2266;
generate
genvar i2281;

for(i2281 = 1; i2281<= 3; i2281= i2281 + 1) begin
always@(posedge clk) begin
tloop2266delay[i2281] <= tloop2266delay[i2281-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2280 = tloop2266delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2283[/*idx2140=*/ 10:0] = '{default:0};
always@(*) shiftreg2283[0] <= idx11;
always@(posedge clk) shiftreg2283[/*idx2140=*/ 10:1] <= shiftreg2283[/*idx2140=*/ 9:0];
wire [31:0] v2282 = shiftreg2283[/*idx2140=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 10][9] = tloop11delay[10];
assign v0_addr_input[/*idx2140=*/ 10][9] = {v2282[3:0]};
wire[31:0] v2284 = v0_rd_data[/*idx2140=*/ 10];
assign v0_rd_en_input[/*idx2140=*/ 10][9] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2285 = /*idx2140=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2287[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2287[0] <= v2284;
always@(posedge clk) shiftreg2287[/*idx13=*/ 9:1] <= shiftreg2287[/*idx13=*/ 8:0];
wire [31:0] v2286 = shiftreg2287[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2288 = v1_rd_data[/*idx2140=*/ 10][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 10][/*idx13=*/ 9][0] = tloop2266delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2289;
mult mult2290(v2289,
v2286,
v2288,
tloop2266delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2291 = v2139[/*idx2140=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2292;
add add2293(v2292,
v2289,
v2291,
tloop2266delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[11] = v2292;

//TerminatorOp

//} Unrolled body 10 of loop2140.
//DEBUG: /*idx2140=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop2140.
//DEBUG: /*idx2140=*/ 4'd11, expected 11
//printTimeOffset
reg tloop2280delay[3:0] = '{default:0} ;
always@(*) tloop2280delay[0] <= tloop2280;
generate
genvar i2295;

for(i2295 = 1; i2295<= 3; i2295= i2295 + 1) begin
always@(posedge clk) begin
tloop2280delay[i2295] <= tloop2280delay[i2295-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2294 = tloop2280delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2297[/*idx2140=*/ 11:0] = '{default:0};
always@(*) shiftreg2297[0] <= idx11;
always@(posedge clk) shiftreg2297[/*idx2140=*/ 11:1] <= shiftreg2297[/*idx2140=*/ 10:0];
wire [31:0] v2296 = shiftreg2297[/*idx2140=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 11][9] = tloop11delay[11];
assign v0_addr_input[/*idx2140=*/ 11][9] = {v2296[3:0]};
wire[31:0] v2298 = v0_rd_data[/*idx2140=*/ 11];
assign v0_rd_en_input[/*idx2140=*/ 11][9] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2299 = /*idx2140=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2301[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2301[0] <= v2298;
always@(posedge clk) shiftreg2301[/*idx13=*/ 9:1] <= shiftreg2301[/*idx13=*/ 8:0];
wire [31:0] v2300 = shiftreg2301[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2302 = v1_rd_data[/*idx2140=*/ 11][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 11][/*idx13=*/ 9][0] = tloop2280delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2303;
mult mult2304(v2303,
v2300,
v2302,
tloop2280delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2305 = v2139[/*idx2140=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2306;
add add2307(v2306,
v2303,
v2305,
tloop2280delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[12] = v2306;

//TerminatorOp

//} Unrolled body 11 of loop2140.
//DEBUG: /*idx2140=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop2140.
//DEBUG: /*idx2140=*/ 4'd12, expected 12
//printTimeOffset
reg tloop2294delay[3:0] = '{default:0} ;
always@(*) tloop2294delay[0] <= tloop2294;
generate
genvar i2309;

for(i2309 = 1; i2309<= 3; i2309= i2309 + 1) begin
always@(posedge clk) begin
tloop2294delay[i2309] <= tloop2294delay[i2309-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2308 = tloop2294delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2311[/*idx2140=*/ 12:0] = '{default:0};
always@(*) shiftreg2311[0] <= idx11;
always@(posedge clk) shiftreg2311[/*idx2140=*/ 12:1] <= shiftreg2311[/*idx2140=*/ 11:0];
wire [31:0] v2310 = shiftreg2311[/*idx2140=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 12][9] = tloop11delay[12];
assign v0_addr_input[/*idx2140=*/ 12][9] = {v2310[3:0]};
wire[31:0] v2312 = v0_rd_data[/*idx2140=*/ 12];
assign v0_rd_en_input[/*idx2140=*/ 12][9] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2313 = /*idx2140=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2315[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2315[0] <= v2312;
always@(posedge clk) shiftreg2315[/*idx13=*/ 9:1] <= shiftreg2315[/*idx13=*/ 8:0];
wire [31:0] v2314 = shiftreg2315[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2316 = v1_rd_data[/*idx2140=*/ 12][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 12][/*idx13=*/ 9][0] = tloop2294delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2317;
mult mult2318(v2317,
v2314,
v2316,
tloop2294delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2319 = v2139[/*idx2140=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2320;
add add2321(v2320,
v2317,
v2319,
tloop2294delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[13] = v2320;

//TerminatorOp

//} Unrolled body 12 of loop2140.
//DEBUG: /*idx2140=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop2140.
//DEBUG: /*idx2140=*/ 4'd13, expected 13
//printTimeOffset
reg tloop2308delay[3:0] = '{default:0} ;
always@(*) tloop2308delay[0] <= tloop2308;
generate
genvar i2323;

for(i2323 = 1; i2323<= 3; i2323= i2323 + 1) begin
always@(posedge clk) begin
tloop2308delay[i2323] <= tloop2308delay[i2323-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2322 = tloop2308delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2325[/*idx2140=*/ 13:0] = '{default:0};
always@(*) shiftreg2325[0] <= idx11;
always@(posedge clk) shiftreg2325[/*idx2140=*/ 13:1] <= shiftreg2325[/*idx2140=*/ 12:0];
wire [31:0] v2324 = shiftreg2325[/*idx2140=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 13][9] = tloop11delay[13];
assign v0_addr_input[/*idx2140=*/ 13][9] = {v2324[3:0]};
wire[31:0] v2326 = v0_rd_data[/*idx2140=*/ 13];
assign v0_rd_en_input[/*idx2140=*/ 13][9] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2327 = /*idx2140=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2329[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2329[0] <= v2326;
always@(posedge clk) shiftreg2329[/*idx13=*/ 9:1] <= shiftreg2329[/*idx13=*/ 8:0];
wire [31:0] v2328 = shiftreg2329[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2330 = v1_rd_data[/*idx2140=*/ 13][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 13][/*idx13=*/ 9][0] = tloop2308delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2331;
mult mult2332(v2331,
v2328,
v2330,
tloop2308delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2333 = v2139[/*idx2140=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2334;
add add2335(v2334,
v2331,
v2333,
tloop2308delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[14] = v2334;

//TerminatorOp

//} Unrolled body 13 of loop2140.
//DEBUG: /*idx2140=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop2140.
//DEBUG: /*idx2140=*/ 4'd14, expected 14
//printTimeOffset
reg tloop2322delay[3:0] = '{default:0} ;
always@(*) tloop2322delay[0] <= tloop2322;
generate
genvar i2337;

for(i2337 = 1; i2337<= 3; i2337= i2337 + 1) begin
always@(posedge clk) begin
tloop2322delay[i2337] <= tloop2322delay[i2337-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2336 = tloop2322delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2339[/*idx2140=*/ 14:0] = '{default:0};
always@(*) shiftreg2339[0] <= idx11;
always@(posedge clk) shiftreg2339[/*idx2140=*/ 14:1] <= shiftreg2339[/*idx2140=*/ 13:0];
wire [31:0] v2338 = shiftreg2339[/*idx2140=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 14][9] = tloop11delay[14];
assign v0_addr_input[/*idx2140=*/ 14][9] = {v2338[3:0]};
wire[31:0] v2340 = v0_rd_data[/*idx2140=*/ 14];
assign v0_rd_en_input[/*idx2140=*/ 14][9] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2341 = /*idx2140=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2343[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2343[0] <= v2340;
always@(posedge clk) shiftreg2343[/*idx13=*/ 9:1] <= shiftreg2343[/*idx13=*/ 8:0];
wire [31:0] v2342 = shiftreg2343[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2344 = v1_rd_data[/*idx2140=*/ 14][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 14][/*idx13=*/ 9][0] = tloop2322delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2345;
mult mult2346(v2345,
v2342,
v2344,
tloop2322delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2347 = v2139[/*idx2140=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2348;
add add2349(v2348,
v2345,
v2347,
tloop2322delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[15] = v2348;

//TerminatorOp

//} Unrolled body 14 of loop2140.
//DEBUG: /*idx2140=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop2140.
//DEBUG: /*idx2140=*/ 4'd15, expected 15
//printTimeOffset
reg tloop2336delay[3:0] = '{default:0} ;
always@(*) tloop2336delay[0] <= tloop2336;
generate
genvar i2351;

for(i2351 = 1; i2351<= 3; i2351= i2351 + 1) begin
always@(posedge clk) begin
tloop2336delay[i2351] <= tloop2336delay[i2351-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2350 = tloop2336delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2353[/*idx2140=*/ 15:0] = '{default:0};
always@(*) shiftreg2353[0] <= idx11;
always@(posedge clk) shiftreg2353[/*idx2140=*/ 15:1] <= shiftreg2353[/*idx2140=*/ 14:0];
wire [31:0] v2352 = shiftreg2353[/*idx2140=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2140=*/ 15][9] = tloop11delay[15];
assign v0_addr_input[/*idx2140=*/ 15][9] = {v2352[3:0]};
wire[31:0] v2354 = v0_rd_data[/*idx2140=*/ 15];
assign v0_rd_en_input[/*idx2140=*/ 15][9] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2355 = /*idx2140=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2357[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2357[0] <= v2354;
always@(posedge clk) shiftreg2357[/*idx13=*/ 9:1] <= shiftreg2357[/*idx13=*/ 8:0];
wire [31:0] v2356 = shiftreg2357[/*idx13=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2358 = v1_rd_data[/*idx2140=*/ 15][/*idx13=*/ 9];
assign v1_rd_en_input[/*idx2140=*/ 15][/*idx13=*/ 9][0] = tloop2336delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2359;
mult mult2360(v2359,
v2356,
v2358,
tloop2336delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2361 = v2139[/*idx2140=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2362;
add add2363(v2362,
v2359,
v2361,
tloop2336delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2139[16] = v2362;

//TerminatorOp

//} Unrolled body 15 of loop2140.
//DEBUG: /*idx2140=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t2364;
assign t2364 = tloop2350;
//printTimeOffset
reg t2364delay[3:0] = '{default:0} ;
always@(*) t2364delay[0] <= t2364;
generate
genvar i2365;

for(i2365 = 1; i2365<= 3; i2365= i2365 + 1) begin
always@(posedge clk) begin
t2364delay[i2365] <= t2364delay[i2365-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v2366 = v2139[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg2368[/*idx13=*/ 9:0] = '{default:0};
always@(*) shiftreg2368[0] <= idx11;
always@(posedge clk) shiftreg2368[/*idx13=*/ 9:1] <= shiftreg2368[/*idx13=*/ 8:0];
wire [31:0] v2367 = shiftreg2368[/*idx13=*/ 9];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg2370[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg2370[0] <= v2367;
always@(posedge clk) shiftreg2370[/*v10=*/ 16:1] <= shiftreg2370[/*v10=*/ 15:0];
wire [31:0] v2369 = shiftreg2370[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg2372[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg2372[0] <= v2369;
always@(posedge clk) shiftreg2372[/*v8=*/ 3:1] <= shiftreg2372[/*v8=*/ 2:0];
wire [31:0] v2371 = shiftreg2372[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 9][0] = t2364delay[3];
assign v2_addr_input[/*idx13=*/ 9][0] = {v2371[3:0]};
assign v2_wr_en_input[/*idx13=*/ 9][0] = t2364delay[3];
assign v2_wr_data_valid[/*idx13=*/ 9][0] = t2364delay[3];
assign v2_wr_data_input[/*idx13=*/ 9][0] = v2366;


//TerminatorOp

//} Unrolled body 9 of loop13.
//DEBUG: /*idx13=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop13.
//DEBUG: /*idx13=*/ 4'd10, expected 10
//printTimeOffset
reg tloop2137delay[3:0] = '{default:0} ;
always@(*) tloop2137delay[0] <= tloop2137;
generate
genvar i2374;

for(i2374 = 1; i2374<= 3; i2374= i2374 + 1) begin
always@(posedge clk) begin
tloop2137delay[i2374] <= tloop2137delay[i2374-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop2373 = tloop2137delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v2375[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v2375[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop2376.
//DEBUG: /*idx2376=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2377 = tloop2137delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2379[/*idx2376=*/ 0:0] = '{default:0};
always@(*) shiftreg2379[0] <= idx11;
wire [31:0] v2378 = shiftreg2379[/*idx2376=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 0][10] = tloop11delay[0];
assign v0_addr_input[/*idx2376=*/ 0][10] = {v2378[3:0]};
wire[31:0] v2380 = v0_rd_data[/*idx2376=*/ 0];
assign v0_rd_en_input[/*idx2376=*/ 0][10] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2381 = /*idx2376=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2383[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2383[0] <= v2380;
always@(posedge clk) shiftreg2383[/*idx13=*/ 10:1] <= shiftreg2383[/*idx13=*/ 9:0];
wire [31:0] v2382 = shiftreg2383[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2384 = v1_rd_data[/*idx2376=*/ 0][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 0][/*idx13=*/ 10][0] = tloop2137delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2385;
mult mult2386(v2385,
v2382,
v2384,
tloop2137delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2387 = v2375[/*idx2376=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2388;
add add2389(v2388,
v2385,
v2387,
tloop2137delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[1] = v2388;

//TerminatorOp

//} Unrolled body 0 of loop2376.
//DEBUG: /*idx2376=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop2376.
//DEBUG: /*idx2376=*/ 1'd1, expected 1
//printTimeOffset
reg tloop2377delay[3:0] = '{default:0} ;
always@(*) tloop2377delay[0] <= tloop2377;
generate
genvar i2391;

for(i2391 = 1; i2391<= 3; i2391= i2391 + 1) begin
always@(posedge clk) begin
tloop2377delay[i2391] <= tloop2377delay[i2391-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2390 = tloop2377delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2393[/*idx2376=*/ 1:0] = '{default:0};
always@(*) shiftreg2393[0] <= idx11;
always@(posedge clk) shiftreg2393[/*idx2376=*/ 1:1] <= shiftreg2393[/*idx2376=*/ 0:0];
wire [31:0] v2392 = shiftreg2393[/*idx2376=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 1][10] = tloop11delay[1];
assign v0_addr_input[/*idx2376=*/ 1][10] = {v2392[3:0]};
wire[31:0] v2394 = v0_rd_data[/*idx2376=*/ 1];
assign v0_rd_en_input[/*idx2376=*/ 1][10] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2395 = /*idx2376=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2397[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2397[0] <= v2394;
always@(posedge clk) shiftreg2397[/*idx13=*/ 10:1] <= shiftreg2397[/*idx13=*/ 9:0];
wire [31:0] v2396 = shiftreg2397[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2398 = v1_rd_data[/*idx2376=*/ 1][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 1][/*idx13=*/ 10][0] = tloop2377delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2399;
mult mult2400(v2399,
v2396,
v2398,
tloop2377delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2401 = v2375[/*idx2376=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2402;
add add2403(v2402,
v2399,
v2401,
tloop2377delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[2] = v2402;

//TerminatorOp

//} Unrolled body 1 of loop2376.
//DEBUG: /*idx2376=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop2376.
//DEBUG: /*idx2376=*/ 2'd2, expected 2
//printTimeOffset
reg tloop2390delay[3:0] = '{default:0} ;
always@(*) tloop2390delay[0] <= tloop2390;
generate
genvar i2405;

for(i2405 = 1; i2405<= 3; i2405= i2405 + 1) begin
always@(posedge clk) begin
tloop2390delay[i2405] <= tloop2390delay[i2405-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2404 = tloop2390delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2407[/*idx2376=*/ 2:0] = '{default:0};
always@(*) shiftreg2407[0] <= idx11;
always@(posedge clk) shiftreg2407[/*idx2376=*/ 2:1] <= shiftreg2407[/*idx2376=*/ 1:0];
wire [31:0] v2406 = shiftreg2407[/*idx2376=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 2][10] = tloop11delay[2];
assign v0_addr_input[/*idx2376=*/ 2][10] = {v2406[3:0]};
wire[31:0] v2408 = v0_rd_data[/*idx2376=*/ 2];
assign v0_rd_en_input[/*idx2376=*/ 2][10] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2409 = /*idx2376=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2411[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2411[0] <= v2408;
always@(posedge clk) shiftreg2411[/*idx13=*/ 10:1] <= shiftreg2411[/*idx13=*/ 9:0];
wire [31:0] v2410 = shiftreg2411[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2412 = v1_rd_data[/*idx2376=*/ 2][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 2][/*idx13=*/ 10][0] = tloop2390delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2413;
mult mult2414(v2413,
v2410,
v2412,
tloop2390delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2415 = v2375[/*idx2376=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2416;
add add2417(v2416,
v2413,
v2415,
tloop2390delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[3] = v2416;

//TerminatorOp

//} Unrolled body 2 of loop2376.
//DEBUG: /*idx2376=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop2376.
//DEBUG: /*idx2376=*/ 2'd3, expected 3
//printTimeOffset
reg tloop2404delay[3:0] = '{default:0} ;
always@(*) tloop2404delay[0] <= tloop2404;
generate
genvar i2419;

for(i2419 = 1; i2419<= 3; i2419= i2419 + 1) begin
always@(posedge clk) begin
tloop2404delay[i2419] <= tloop2404delay[i2419-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2418 = tloop2404delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2421[/*idx2376=*/ 3:0] = '{default:0};
always@(*) shiftreg2421[0] <= idx11;
always@(posedge clk) shiftreg2421[/*idx2376=*/ 3:1] <= shiftreg2421[/*idx2376=*/ 2:0];
wire [31:0] v2420 = shiftreg2421[/*idx2376=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 3][10] = tloop11delay[3];
assign v0_addr_input[/*idx2376=*/ 3][10] = {v2420[3:0]};
wire[31:0] v2422 = v0_rd_data[/*idx2376=*/ 3];
assign v0_rd_en_input[/*idx2376=*/ 3][10] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2423 = /*idx2376=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2425[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2425[0] <= v2422;
always@(posedge clk) shiftreg2425[/*idx13=*/ 10:1] <= shiftreg2425[/*idx13=*/ 9:0];
wire [31:0] v2424 = shiftreg2425[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2426 = v1_rd_data[/*idx2376=*/ 3][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 3][/*idx13=*/ 10][0] = tloop2404delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2427;
mult mult2428(v2427,
v2424,
v2426,
tloop2404delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2429 = v2375[/*idx2376=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2430;
add add2431(v2430,
v2427,
v2429,
tloop2404delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[4] = v2430;

//TerminatorOp

//} Unrolled body 3 of loop2376.
//DEBUG: /*idx2376=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop2376.
//DEBUG: /*idx2376=*/ 3'd4, expected 4
//printTimeOffset
reg tloop2418delay[3:0] = '{default:0} ;
always@(*) tloop2418delay[0] <= tloop2418;
generate
genvar i2433;

for(i2433 = 1; i2433<= 3; i2433= i2433 + 1) begin
always@(posedge clk) begin
tloop2418delay[i2433] <= tloop2418delay[i2433-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2432 = tloop2418delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2435[/*idx2376=*/ 4:0] = '{default:0};
always@(*) shiftreg2435[0] <= idx11;
always@(posedge clk) shiftreg2435[/*idx2376=*/ 4:1] <= shiftreg2435[/*idx2376=*/ 3:0];
wire [31:0] v2434 = shiftreg2435[/*idx2376=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 4][10] = tloop11delay[4];
assign v0_addr_input[/*idx2376=*/ 4][10] = {v2434[3:0]};
wire[31:0] v2436 = v0_rd_data[/*idx2376=*/ 4];
assign v0_rd_en_input[/*idx2376=*/ 4][10] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2437 = /*idx2376=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2439[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2439[0] <= v2436;
always@(posedge clk) shiftreg2439[/*idx13=*/ 10:1] <= shiftreg2439[/*idx13=*/ 9:0];
wire [31:0] v2438 = shiftreg2439[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2440 = v1_rd_data[/*idx2376=*/ 4][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 4][/*idx13=*/ 10][0] = tloop2418delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2441;
mult mult2442(v2441,
v2438,
v2440,
tloop2418delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2443 = v2375[/*idx2376=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2444;
add add2445(v2444,
v2441,
v2443,
tloop2418delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[5] = v2444;

//TerminatorOp

//} Unrolled body 4 of loop2376.
//DEBUG: /*idx2376=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop2376.
//DEBUG: /*idx2376=*/ 3'd5, expected 5
//printTimeOffset
reg tloop2432delay[3:0] = '{default:0} ;
always@(*) tloop2432delay[0] <= tloop2432;
generate
genvar i2447;

for(i2447 = 1; i2447<= 3; i2447= i2447 + 1) begin
always@(posedge clk) begin
tloop2432delay[i2447] <= tloop2432delay[i2447-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2446 = tloop2432delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2449[/*idx2376=*/ 5:0] = '{default:0};
always@(*) shiftreg2449[0] <= idx11;
always@(posedge clk) shiftreg2449[/*idx2376=*/ 5:1] <= shiftreg2449[/*idx2376=*/ 4:0];
wire [31:0] v2448 = shiftreg2449[/*idx2376=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 5][10] = tloop11delay[5];
assign v0_addr_input[/*idx2376=*/ 5][10] = {v2448[3:0]};
wire[31:0] v2450 = v0_rd_data[/*idx2376=*/ 5];
assign v0_rd_en_input[/*idx2376=*/ 5][10] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2451 = /*idx2376=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2453[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2453[0] <= v2450;
always@(posedge clk) shiftreg2453[/*idx13=*/ 10:1] <= shiftreg2453[/*idx13=*/ 9:0];
wire [31:0] v2452 = shiftreg2453[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2454 = v1_rd_data[/*idx2376=*/ 5][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 5][/*idx13=*/ 10][0] = tloop2432delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2455;
mult mult2456(v2455,
v2452,
v2454,
tloop2432delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2457 = v2375[/*idx2376=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2458;
add add2459(v2458,
v2455,
v2457,
tloop2432delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[6] = v2458;

//TerminatorOp

//} Unrolled body 5 of loop2376.
//DEBUG: /*idx2376=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop2376.
//DEBUG: /*idx2376=*/ 3'd6, expected 6
//printTimeOffset
reg tloop2446delay[3:0] = '{default:0} ;
always@(*) tloop2446delay[0] <= tloop2446;
generate
genvar i2461;

for(i2461 = 1; i2461<= 3; i2461= i2461 + 1) begin
always@(posedge clk) begin
tloop2446delay[i2461] <= tloop2446delay[i2461-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2460 = tloop2446delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2463[/*idx2376=*/ 6:0] = '{default:0};
always@(*) shiftreg2463[0] <= idx11;
always@(posedge clk) shiftreg2463[/*idx2376=*/ 6:1] <= shiftreg2463[/*idx2376=*/ 5:0];
wire [31:0] v2462 = shiftreg2463[/*idx2376=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 6][10] = tloop11delay[6];
assign v0_addr_input[/*idx2376=*/ 6][10] = {v2462[3:0]};
wire[31:0] v2464 = v0_rd_data[/*idx2376=*/ 6];
assign v0_rd_en_input[/*idx2376=*/ 6][10] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2465 = /*idx2376=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2467[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2467[0] <= v2464;
always@(posedge clk) shiftreg2467[/*idx13=*/ 10:1] <= shiftreg2467[/*idx13=*/ 9:0];
wire [31:0] v2466 = shiftreg2467[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2468 = v1_rd_data[/*idx2376=*/ 6][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 6][/*idx13=*/ 10][0] = tloop2446delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2469;
mult mult2470(v2469,
v2466,
v2468,
tloop2446delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2471 = v2375[/*idx2376=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2472;
add add2473(v2472,
v2469,
v2471,
tloop2446delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[7] = v2472;

//TerminatorOp

//} Unrolled body 6 of loop2376.
//DEBUG: /*idx2376=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop2376.
//DEBUG: /*idx2376=*/ 3'd7, expected 7
//printTimeOffset
reg tloop2460delay[3:0] = '{default:0} ;
always@(*) tloop2460delay[0] <= tloop2460;
generate
genvar i2475;

for(i2475 = 1; i2475<= 3; i2475= i2475 + 1) begin
always@(posedge clk) begin
tloop2460delay[i2475] <= tloop2460delay[i2475-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2474 = tloop2460delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2477[/*idx2376=*/ 7:0] = '{default:0};
always@(*) shiftreg2477[0] <= idx11;
always@(posedge clk) shiftreg2477[/*idx2376=*/ 7:1] <= shiftreg2477[/*idx2376=*/ 6:0];
wire [31:0] v2476 = shiftreg2477[/*idx2376=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 7][10] = tloop11delay[7];
assign v0_addr_input[/*idx2376=*/ 7][10] = {v2476[3:0]};
wire[31:0] v2478 = v0_rd_data[/*idx2376=*/ 7];
assign v0_rd_en_input[/*idx2376=*/ 7][10] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2479 = /*idx2376=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2481[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2481[0] <= v2478;
always@(posedge clk) shiftreg2481[/*idx13=*/ 10:1] <= shiftreg2481[/*idx13=*/ 9:0];
wire [31:0] v2480 = shiftreg2481[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2482 = v1_rd_data[/*idx2376=*/ 7][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 7][/*idx13=*/ 10][0] = tloop2460delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2483;
mult mult2484(v2483,
v2480,
v2482,
tloop2460delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2485 = v2375[/*idx2376=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2486;
add add2487(v2486,
v2483,
v2485,
tloop2460delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[8] = v2486;

//TerminatorOp

//} Unrolled body 7 of loop2376.
//DEBUG: /*idx2376=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop2376.
//DEBUG: /*idx2376=*/ 4'd8, expected 8
//printTimeOffset
reg tloop2474delay[3:0] = '{default:0} ;
always@(*) tloop2474delay[0] <= tloop2474;
generate
genvar i2489;

for(i2489 = 1; i2489<= 3; i2489= i2489 + 1) begin
always@(posedge clk) begin
tloop2474delay[i2489] <= tloop2474delay[i2489-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2488 = tloop2474delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2491[/*idx2376=*/ 8:0] = '{default:0};
always@(*) shiftreg2491[0] <= idx11;
always@(posedge clk) shiftreg2491[/*idx2376=*/ 8:1] <= shiftreg2491[/*idx2376=*/ 7:0];
wire [31:0] v2490 = shiftreg2491[/*idx2376=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 8][10] = tloop11delay[8];
assign v0_addr_input[/*idx2376=*/ 8][10] = {v2490[3:0]};
wire[31:0] v2492 = v0_rd_data[/*idx2376=*/ 8];
assign v0_rd_en_input[/*idx2376=*/ 8][10] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2493 = /*idx2376=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2495[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2495[0] <= v2492;
always@(posedge clk) shiftreg2495[/*idx13=*/ 10:1] <= shiftreg2495[/*idx13=*/ 9:0];
wire [31:0] v2494 = shiftreg2495[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2496 = v1_rd_data[/*idx2376=*/ 8][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 8][/*idx13=*/ 10][0] = tloop2474delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2497;
mult mult2498(v2497,
v2494,
v2496,
tloop2474delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2499 = v2375[/*idx2376=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2500;
add add2501(v2500,
v2497,
v2499,
tloop2474delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[9] = v2500;

//TerminatorOp

//} Unrolled body 8 of loop2376.
//DEBUG: /*idx2376=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop2376.
//DEBUG: /*idx2376=*/ 4'd9, expected 9
//printTimeOffset
reg tloop2488delay[3:0] = '{default:0} ;
always@(*) tloop2488delay[0] <= tloop2488;
generate
genvar i2503;

for(i2503 = 1; i2503<= 3; i2503= i2503 + 1) begin
always@(posedge clk) begin
tloop2488delay[i2503] <= tloop2488delay[i2503-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2502 = tloop2488delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2505[/*idx2376=*/ 9:0] = '{default:0};
always@(*) shiftreg2505[0] <= idx11;
always@(posedge clk) shiftreg2505[/*idx2376=*/ 9:1] <= shiftreg2505[/*idx2376=*/ 8:0];
wire [31:0] v2504 = shiftreg2505[/*idx2376=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 9][10] = tloop11delay[9];
assign v0_addr_input[/*idx2376=*/ 9][10] = {v2504[3:0]};
wire[31:0] v2506 = v0_rd_data[/*idx2376=*/ 9];
assign v0_rd_en_input[/*idx2376=*/ 9][10] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2507 = /*idx2376=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2509[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2509[0] <= v2506;
always@(posedge clk) shiftreg2509[/*idx13=*/ 10:1] <= shiftreg2509[/*idx13=*/ 9:0];
wire [31:0] v2508 = shiftreg2509[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2510 = v1_rd_data[/*idx2376=*/ 9][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 9][/*idx13=*/ 10][0] = tloop2488delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2511;
mult mult2512(v2511,
v2508,
v2510,
tloop2488delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2513 = v2375[/*idx2376=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2514;
add add2515(v2514,
v2511,
v2513,
tloop2488delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[10] = v2514;

//TerminatorOp

//} Unrolled body 9 of loop2376.
//DEBUG: /*idx2376=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop2376.
//DEBUG: /*idx2376=*/ 4'd10, expected 10
//printTimeOffset
reg tloop2502delay[3:0] = '{default:0} ;
always@(*) tloop2502delay[0] <= tloop2502;
generate
genvar i2517;

for(i2517 = 1; i2517<= 3; i2517= i2517 + 1) begin
always@(posedge clk) begin
tloop2502delay[i2517] <= tloop2502delay[i2517-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2516 = tloop2502delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2519[/*idx2376=*/ 10:0] = '{default:0};
always@(*) shiftreg2519[0] <= idx11;
always@(posedge clk) shiftreg2519[/*idx2376=*/ 10:1] <= shiftreg2519[/*idx2376=*/ 9:0];
wire [31:0] v2518 = shiftreg2519[/*idx2376=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 10][10] = tloop11delay[10];
assign v0_addr_input[/*idx2376=*/ 10][10] = {v2518[3:0]};
wire[31:0] v2520 = v0_rd_data[/*idx2376=*/ 10];
assign v0_rd_en_input[/*idx2376=*/ 10][10] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2521 = /*idx2376=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2523[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2523[0] <= v2520;
always@(posedge clk) shiftreg2523[/*idx13=*/ 10:1] <= shiftreg2523[/*idx13=*/ 9:0];
wire [31:0] v2522 = shiftreg2523[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2524 = v1_rd_data[/*idx2376=*/ 10][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 10][/*idx13=*/ 10][0] = tloop2502delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2525;
mult mult2526(v2525,
v2522,
v2524,
tloop2502delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2527 = v2375[/*idx2376=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2528;
add add2529(v2528,
v2525,
v2527,
tloop2502delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[11] = v2528;

//TerminatorOp

//} Unrolled body 10 of loop2376.
//DEBUG: /*idx2376=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop2376.
//DEBUG: /*idx2376=*/ 4'd11, expected 11
//printTimeOffset
reg tloop2516delay[3:0] = '{default:0} ;
always@(*) tloop2516delay[0] <= tloop2516;
generate
genvar i2531;

for(i2531 = 1; i2531<= 3; i2531= i2531 + 1) begin
always@(posedge clk) begin
tloop2516delay[i2531] <= tloop2516delay[i2531-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2530 = tloop2516delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2533[/*idx2376=*/ 11:0] = '{default:0};
always@(*) shiftreg2533[0] <= idx11;
always@(posedge clk) shiftreg2533[/*idx2376=*/ 11:1] <= shiftreg2533[/*idx2376=*/ 10:0];
wire [31:0] v2532 = shiftreg2533[/*idx2376=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 11][10] = tloop11delay[11];
assign v0_addr_input[/*idx2376=*/ 11][10] = {v2532[3:0]};
wire[31:0] v2534 = v0_rd_data[/*idx2376=*/ 11];
assign v0_rd_en_input[/*idx2376=*/ 11][10] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2535 = /*idx2376=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2537[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2537[0] <= v2534;
always@(posedge clk) shiftreg2537[/*idx13=*/ 10:1] <= shiftreg2537[/*idx13=*/ 9:0];
wire [31:0] v2536 = shiftreg2537[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2538 = v1_rd_data[/*idx2376=*/ 11][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 11][/*idx13=*/ 10][0] = tloop2516delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2539;
mult mult2540(v2539,
v2536,
v2538,
tloop2516delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2541 = v2375[/*idx2376=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2542;
add add2543(v2542,
v2539,
v2541,
tloop2516delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[12] = v2542;

//TerminatorOp

//} Unrolled body 11 of loop2376.
//DEBUG: /*idx2376=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop2376.
//DEBUG: /*idx2376=*/ 4'd12, expected 12
//printTimeOffset
reg tloop2530delay[3:0] = '{default:0} ;
always@(*) tloop2530delay[0] <= tloop2530;
generate
genvar i2545;

for(i2545 = 1; i2545<= 3; i2545= i2545 + 1) begin
always@(posedge clk) begin
tloop2530delay[i2545] <= tloop2530delay[i2545-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2544 = tloop2530delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2547[/*idx2376=*/ 12:0] = '{default:0};
always@(*) shiftreg2547[0] <= idx11;
always@(posedge clk) shiftreg2547[/*idx2376=*/ 12:1] <= shiftreg2547[/*idx2376=*/ 11:0];
wire [31:0] v2546 = shiftreg2547[/*idx2376=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 12][10] = tloop11delay[12];
assign v0_addr_input[/*idx2376=*/ 12][10] = {v2546[3:0]};
wire[31:0] v2548 = v0_rd_data[/*idx2376=*/ 12];
assign v0_rd_en_input[/*idx2376=*/ 12][10] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2549 = /*idx2376=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2551[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2551[0] <= v2548;
always@(posedge clk) shiftreg2551[/*idx13=*/ 10:1] <= shiftreg2551[/*idx13=*/ 9:0];
wire [31:0] v2550 = shiftreg2551[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2552 = v1_rd_data[/*idx2376=*/ 12][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 12][/*idx13=*/ 10][0] = tloop2530delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2553;
mult mult2554(v2553,
v2550,
v2552,
tloop2530delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2555 = v2375[/*idx2376=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2556;
add add2557(v2556,
v2553,
v2555,
tloop2530delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[13] = v2556;

//TerminatorOp

//} Unrolled body 12 of loop2376.
//DEBUG: /*idx2376=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop2376.
//DEBUG: /*idx2376=*/ 4'd13, expected 13
//printTimeOffset
reg tloop2544delay[3:0] = '{default:0} ;
always@(*) tloop2544delay[0] <= tloop2544;
generate
genvar i2559;

for(i2559 = 1; i2559<= 3; i2559= i2559 + 1) begin
always@(posedge clk) begin
tloop2544delay[i2559] <= tloop2544delay[i2559-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2558 = tloop2544delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2561[/*idx2376=*/ 13:0] = '{default:0};
always@(*) shiftreg2561[0] <= idx11;
always@(posedge clk) shiftreg2561[/*idx2376=*/ 13:1] <= shiftreg2561[/*idx2376=*/ 12:0];
wire [31:0] v2560 = shiftreg2561[/*idx2376=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 13][10] = tloop11delay[13];
assign v0_addr_input[/*idx2376=*/ 13][10] = {v2560[3:0]};
wire[31:0] v2562 = v0_rd_data[/*idx2376=*/ 13];
assign v0_rd_en_input[/*idx2376=*/ 13][10] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2563 = /*idx2376=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2565[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2565[0] <= v2562;
always@(posedge clk) shiftreg2565[/*idx13=*/ 10:1] <= shiftreg2565[/*idx13=*/ 9:0];
wire [31:0] v2564 = shiftreg2565[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2566 = v1_rd_data[/*idx2376=*/ 13][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 13][/*idx13=*/ 10][0] = tloop2544delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2567;
mult mult2568(v2567,
v2564,
v2566,
tloop2544delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2569 = v2375[/*idx2376=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2570;
add add2571(v2570,
v2567,
v2569,
tloop2544delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[14] = v2570;

//TerminatorOp

//} Unrolled body 13 of loop2376.
//DEBUG: /*idx2376=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop2376.
//DEBUG: /*idx2376=*/ 4'd14, expected 14
//printTimeOffset
reg tloop2558delay[3:0] = '{default:0} ;
always@(*) tloop2558delay[0] <= tloop2558;
generate
genvar i2573;

for(i2573 = 1; i2573<= 3; i2573= i2573 + 1) begin
always@(posedge clk) begin
tloop2558delay[i2573] <= tloop2558delay[i2573-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2572 = tloop2558delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2575[/*idx2376=*/ 14:0] = '{default:0};
always@(*) shiftreg2575[0] <= idx11;
always@(posedge clk) shiftreg2575[/*idx2376=*/ 14:1] <= shiftreg2575[/*idx2376=*/ 13:0];
wire [31:0] v2574 = shiftreg2575[/*idx2376=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 14][10] = tloop11delay[14];
assign v0_addr_input[/*idx2376=*/ 14][10] = {v2574[3:0]};
wire[31:0] v2576 = v0_rd_data[/*idx2376=*/ 14];
assign v0_rd_en_input[/*idx2376=*/ 14][10] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2577 = /*idx2376=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2579[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2579[0] <= v2576;
always@(posedge clk) shiftreg2579[/*idx13=*/ 10:1] <= shiftreg2579[/*idx13=*/ 9:0];
wire [31:0] v2578 = shiftreg2579[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2580 = v1_rd_data[/*idx2376=*/ 14][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 14][/*idx13=*/ 10][0] = tloop2558delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2581;
mult mult2582(v2581,
v2578,
v2580,
tloop2558delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2583 = v2375[/*idx2376=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2584;
add add2585(v2584,
v2581,
v2583,
tloop2558delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[15] = v2584;

//TerminatorOp

//} Unrolled body 14 of loop2376.
//DEBUG: /*idx2376=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop2376.
//DEBUG: /*idx2376=*/ 4'd15, expected 15
//printTimeOffset
reg tloop2572delay[3:0] = '{default:0} ;
always@(*) tloop2572delay[0] <= tloop2572;
generate
genvar i2587;

for(i2587 = 1; i2587<= 3; i2587= i2587 + 1) begin
always@(posedge clk) begin
tloop2572delay[i2587] <= tloop2572delay[i2587-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2586 = tloop2572delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2589[/*idx2376=*/ 15:0] = '{default:0};
always@(*) shiftreg2589[0] <= idx11;
always@(posedge clk) shiftreg2589[/*idx2376=*/ 15:1] <= shiftreg2589[/*idx2376=*/ 14:0];
wire [31:0] v2588 = shiftreg2589[/*idx2376=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2376=*/ 15][10] = tloop11delay[15];
assign v0_addr_input[/*idx2376=*/ 15][10] = {v2588[3:0]};
wire[31:0] v2590 = v0_rd_data[/*idx2376=*/ 15];
assign v0_rd_en_input[/*idx2376=*/ 15][10] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2591 = /*idx2376=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2593[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2593[0] <= v2590;
always@(posedge clk) shiftreg2593[/*idx13=*/ 10:1] <= shiftreg2593[/*idx13=*/ 9:0];
wire [31:0] v2592 = shiftreg2593[/*idx13=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2594 = v1_rd_data[/*idx2376=*/ 15][/*idx13=*/ 10];
assign v1_rd_en_input[/*idx2376=*/ 15][/*idx13=*/ 10][0] = tloop2572delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2595;
mult mult2596(v2595,
v2592,
v2594,
tloop2572delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2597 = v2375[/*idx2376=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2598;
add add2599(v2598,
v2595,
v2597,
tloop2572delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2375[16] = v2598;

//TerminatorOp

//} Unrolled body 15 of loop2376.
//DEBUG: /*idx2376=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t2600;
assign t2600 = tloop2586;
//printTimeOffset
reg t2600delay[3:0] = '{default:0} ;
always@(*) t2600delay[0] <= t2600;
generate
genvar i2601;

for(i2601 = 1; i2601<= 3; i2601= i2601 + 1) begin
always@(posedge clk) begin
t2600delay[i2601] <= t2600delay[i2601-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v2602 = v2375[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg2604[/*idx13=*/ 10:0] = '{default:0};
always@(*) shiftreg2604[0] <= idx11;
always@(posedge clk) shiftreg2604[/*idx13=*/ 10:1] <= shiftreg2604[/*idx13=*/ 9:0];
wire [31:0] v2603 = shiftreg2604[/*idx13=*/ 10];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg2606[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg2606[0] <= v2603;
always@(posedge clk) shiftreg2606[/*v10=*/ 16:1] <= shiftreg2606[/*v10=*/ 15:0];
wire [31:0] v2605 = shiftreg2606[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg2608[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg2608[0] <= v2605;
always@(posedge clk) shiftreg2608[/*v8=*/ 3:1] <= shiftreg2608[/*v8=*/ 2:0];
wire [31:0] v2607 = shiftreg2608[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 10][0] = t2600delay[3];
assign v2_addr_input[/*idx13=*/ 10][0] = {v2607[3:0]};
assign v2_wr_en_input[/*idx13=*/ 10][0] = t2600delay[3];
assign v2_wr_data_valid[/*idx13=*/ 10][0] = t2600delay[3];
assign v2_wr_data_input[/*idx13=*/ 10][0] = v2602;


//TerminatorOp

//} Unrolled body 10 of loop13.
//DEBUG: /*idx13=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop13.
//DEBUG: /*idx13=*/ 4'd11, expected 11
//printTimeOffset
reg tloop2373delay[3:0] = '{default:0} ;
always@(*) tloop2373delay[0] <= tloop2373;
generate
genvar i2610;

for(i2610 = 1; i2610<= 3; i2610= i2610 + 1) begin
always@(posedge clk) begin
tloop2373delay[i2610] <= tloop2373delay[i2610-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop2609 = tloop2373delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v2611[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v2611[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop2612.
//DEBUG: /*idx2612=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2613 = tloop2373delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2615[/*idx2612=*/ 0:0] = '{default:0};
always@(*) shiftreg2615[0] <= idx11;
wire [31:0] v2614 = shiftreg2615[/*idx2612=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 0][11] = tloop11delay[0];
assign v0_addr_input[/*idx2612=*/ 0][11] = {v2614[3:0]};
wire[31:0] v2616 = v0_rd_data[/*idx2612=*/ 0];
assign v0_rd_en_input[/*idx2612=*/ 0][11] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2617 = /*idx2612=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2619[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2619[0] <= v2616;
always@(posedge clk) shiftreg2619[/*idx13=*/ 11:1] <= shiftreg2619[/*idx13=*/ 10:0];
wire [31:0] v2618 = shiftreg2619[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2620 = v1_rd_data[/*idx2612=*/ 0][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 0][/*idx13=*/ 11][0] = tloop2373delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2621;
mult mult2622(v2621,
v2618,
v2620,
tloop2373delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2623 = v2611[/*idx2612=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2624;
add add2625(v2624,
v2621,
v2623,
tloop2373delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[1] = v2624;

//TerminatorOp

//} Unrolled body 0 of loop2612.
//DEBUG: /*idx2612=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop2612.
//DEBUG: /*idx2612=*/ 1'd1, expected 1
//printTimeOffset
reg tloop2613delay[3:0] = '{default:0} ;
always@(*) tloop2613delay[0] <= tloop2613;
generate
genvar i2627;

for(i2627 = 1; i2627<= 3; i2627= i2627 + 1) begin
always@(posedge clk) begin
tloop2613delay[i2627] <= tloop2613delay[i2627-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2626 = tloop2613delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2629[/*idx2612=*/ 1:0] = '{default:0};
always@(*) shiftreg2629[0] <= idx11;
always@(posedge clk) shiftreg2629[/*idx2612=*/ 1:1] <= shiftreg2629[/*idx2612=*/ 0:0];
wire [31:0] v2628 = shiftreg2629[/*idx2612=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 1][11] = tloop11delay[1];
assign v0_addr_input[/*idx2612=*/ 1][11] = {v2628[3:0]};
wire[31:0] v2630 = v0_rd_data[/*idx2612=*/ 1];
assign v0_rd_en_input[/*idx2612=*/ 1][11] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2631 = /*idx2612=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2633[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2633[0] <= v2630;
always@(posedge clk) shiftreg2633[/*idx13=*/ 11:1] <= shiftreg2633[/*idx13=*/ 10:0];
wire [31:0] v2632 = shiftreg2633[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2634 = v1_rd_data[/*idx2612=*/ 1][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 1][/*idx13=*/ 11][0] = tloop2613delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2635;
mult mult2636(v2635,
v2632,
v2634,
tloop2613delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2637 = v2611[/*idx2612=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2638;
add add2639(v2638,
v2635,
v2637,
tloop2613delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[2] = v2638;

//TerminatorOp

//} Unrolled body 1 of loop2612.
//DEBUG: /*idx2612=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop2612.
//DEBUG: /*idx2612=*/ 2'd2, expected 2
//printTimeOffset
reg tloop2626delay[3:0] = '{default:0} ;
always@(*) tloop2626delay[0] <= tloop2626;
generate
genvar i2641;

for(i2641 = 1; i2641<= 3; i2641= i2641 + 1) begin
always@(posedge clk) begin
tloop2626delay[i2641] <= tloop2626delay[i2641-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2640 = tloop2626delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2643[/*idx2612=*/ 2:0] = '{default:0};
always@(*) shiftreg2643[0] <= idx11;
always@(posedge clk) shiftreg2643[/*idx2612=*/ 2:1] <= shiftreg2643[/*idx2612=*/ 1:0];
wire [31:0] v2642 = shiftreg2643[/*idx2612=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 2][11] = tloop11delay[2];
assign v0_addr_input[/*idx2612=*/ 2][11] = {v2642[3:0]};
wire[31:0] v2644 = v0_rd_data[/*idx2612=*/ 2];
assign v0_rd_en_input[/*idx2612=*/ 2][11] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2645 = /*idx2612=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2647[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2647[0] <= v2644;
always@(posedge clk) shiftreg2647[/*idx13=*/ 11:1] <= shiftreg2647[/*idx13=*/ 10:0];
wire [31:0] v2646 = shiftreg2647[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2648 = v1_rd_data[/*idx2612=*/ 2][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 2][/*idx13=*/ 11][0] = tloop2626delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2649;
mult mult2650(v2649,
v2646,
v2648,
tloop2626delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2651 = v2611[/*idx2612=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2652;
add add2653(v2652,
v2649,
v2651,
tloop2626delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[3] = v2652;

//TerminatorOp

//} Unrolled body 2 of loop2612.
//DEBUG: /*idx2612=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop2612.
//DEBUG: /*idx2612=*/ 2'd3, expected 3
//printTimeOffset
reg tloop2640delay[3:0] = '{default:0} ;
always@(*) tloop2640delay[0] <= tloop2640;
generate
genvar i2655;

for(i2655 = 1; i2655<= 3; i2655= i2655 + 1) begin
always@(posedge clk) begin
tloop2640delay[i2655] <= tloop2640delay[i2655-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2654 = tloop2640delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2657[/*idx2612=*/ 3:0] = '{default:0};
always@(*) shiftreg2657[0] <= idx11;
always@(posedge clk) shiftreg2657[/*idx2612=*/ 3:1] <= shiftreg2657[/*idx2612=*/ 2:0];
wire [31:0] v2656 = shiftreg2657[/*idx2612=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 3][11] = tloop11delay[3];
assign v0_addr_input[/*idx2612=*/ 3][11] = {v2656[3:0]};
wire[31:0] v2658 = v0_rd_data[/*idx2612=*/ 3];
assign v0_rd_en_input[/*idx2612=*/ 3][11] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2659 = /*idx2612=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2661[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2661[0] <= v2658;
always@(posedge clk) shiftreg2661[/*idx13=*/ 11:1] <= shiftreg2661[/*idx13=*/ 10:0];
wire [31:0] v2660 = shiftreg2661[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2662 = v1_rd_data[/*idx2612=*/ 3][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 3][/*idx13=*/ 11][0] = tloop2640delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2663;
mult mult2664(v2663,
v2660,
v2662,
tloop2640delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2665 = v2611[/*idx2612=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2666;
add add2667(v2666,
v2663,
v2665,
tloop2640delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[4] = v2666;

//TerminatorOp

//} Unrolled body 3 of loop2612.
//DEBUG: /*idx2612=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop2612.
//DEBUG: /*idx2612=*/ 3'd4, expected 4
//printTimeOffset
reg tloop2654delay[3:0] = '{default:0} ;
always@(*) tloop2654delay[0] <= tloop2654;
generate
genvar i2669;

for(i2669 = 1; i2669<= 3; i2669= i2669 + 1) begin
always@(posedge clk) begin
tloop2654delay[i2669] <= tloop2654delay[i2669-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2668 = tloop2654delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2671[/*idx2612=*/ 4:0] = '{default:0};
always@(*) shiftreg2671[0] <= idx11;
always@(posedge clk) shiftreg2671[/*idx2612=*/ 4:1] <= shiftreg2671[/*idx2612=*/ 3:0];
wire [31:0] v2670 = shiftreg2671[/*idx2612=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 4][11] = tloop11delay[4];
assign v0_addr_input[/*idx2612=*/ 4][11] = {v2670[3:0]};
wire[31:0] v2672 = v0_rd_data[/*idx2612=*/ 4];
assign v0_rd_en_input[/*idx2612=*/ 4][11] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2673 = /*idx2612=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2675[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2675[0] <= v2672;
always@(posedge clk) shiftreg2675[/*idx13=*/ 11:1] <= shiftreg2675[/*idx13=*/ 10:0];
wire [31:0] v2674 = shiftreg2675[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2676 = v1_rd_data[/*idx2612=*/ 4][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 4][/*idx13=*/ 11][0] = tloop2654delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2677;
mult mult2678(v2677,
v2674,
v2676,
tloop2654delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2679 = v2611[/*idx2612=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2680;
add add2681(v2680,
v2677,
v2679,
tloop2654delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[5] = v2680;

//TerminatorOp

//} Unrolled body 4 of loop2612.
//DEBUG: /*idx2612=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop2612.
//DEBUG: /*idx2612=*/ 3'd5, expected 5
//printTimeOffset
reg tloop2668delay[3:0] = '{default:0} ;
always@(*) tloop2668delay[0] <= tloop2668;
generate
genvar i2683;

for(i2683 = 1; i2683<= 3; i2683= i2683 + 1) begin
always@(posedge clk) begin
tloop2668delay[i2683] <= tloop2668delay[i2683-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2682 = tloop2668delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2685[/*idx2612=*/ 5:0] = '{default:0};
always@(*) shiftreg2685[0] <= idx11;
always@(posedge clk) shiftreg2685[/*idx2612=*/ 5:1] <= shiftreg2685[/*idx2612=*/ 4:0];
wire [31:0] v2684 = shiftreg2685[/*idx2612=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 5][11] = tloop11delay[5];
assign v0_addr_input[/*idx2612=*/ 5][11] = {v2684[3:0]};
wire[31:0] v2686 = v0_rd_data[/*idx2612=*/ 5];
assign v0_rd_en_input[/*idx2612=*/ 5][11] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2687 = /*idx2612=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2689[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2689[0] <= v2686;
always@(posedge clk) shiftreg2689[/*idx13=*/ 11:1] <= shiftreg2689[/*idx13=*/ 10:0];
wire [31:0] v2688 = shiftreg2689[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2690 = v1_rd_data[/*idx2612=*/ 5][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 5][/*idx13=*/ 11][0] = tloop2668delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2691;
mult mult2692(v2691,
v2688,
v2690,
tloop2668delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2693 = v2611[/*idx2612=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2694;
add add2695(v2694,
v2691,
v2693,
tloop2668delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[6] = v2694;

//TerminatorOp

//} Unrolled body 5 of loop2612.
//DEBUG: /*idx2612=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop2612.
//DEBUG: /*idx2612=*/ 3'd6, expected 6
//printTimeOffset
reg tloop2682delay[3:0] = '{default:0} ;
always@(*) tloop2682delay[0] <= tloop2682;
generate
genvar i2697;

for(i2697 = 1; i2697<= 3; i2697= i2697 + 1) begin
always@(posedge clk) begin
tloop2682delay[i2697] <= tloop2682delay[i2697-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2696 = tloop2682delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2699[/*idx2612=*/ 6:0] = '{default:0};
always@(*) shiftreg2699[0] <= idx11;
always@(posedge clk) shiftreg2699[/*idx2612=*/ 6:1] <= shiftreg2699[/*idx2612=*/ 5:0];
wire [31:0] v2698 = shiftreg2699[/*idx2612=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 6][11] = tloop11delay[6];
assign v0_addr_input[/*idx2612=*/ 6][11] = {v2698[3:0]};
wire[31:0] v2700 = v0_rd_data[/*idx2612=*/ 6];
assign v0_rd_en_input[/*idx2612=*/ 6][11] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2701 = /*idx2612=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2703[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2703[0] <= v2700;
always@(posedge clk) shiftreg2703[/*idx13=*/ 11:1] <= shiftreg2703[/*idx13=*/ 10:0];
wire [31:0] v2702 = shiftreg2703[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2704 = v1_rd_data[/*idx2612=*/ 6][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 6][/*idx13=*/ 11][0] = tloop2682delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2705;
mult mult2706(v2705,
v2702,
v2704,
tloop2682delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2707 = v2611[/*idx2612=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2708;
add add2709(v2708,
v2705,
v2707,
tloop2682delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[7] = v2708;

//TerminatorOp

//} Unrolled body 6 of loop2612.
//DEBUG: /*idx2612=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop2612.
//DEBUG: /*idx2612=*/ 3'd7, expected 7
//printTimeOffset
reg tloop2696delay[3:0] = '{default:0} ;
always@(*) tloop2696delay[0] <= tloop2696;
generate
genvar i2711;

for(i2711 = 1; i2711<= 3; i2711= i2711 + 1) begin
always@(posedge clk) begin
tloop2696delay[i2711] <= tloop2696delay[i2711-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2710 = tloop2696delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2713[/*idx2612=*/ 7:0] = '{default:0};
always@(*) shiftreg2713[0] <= idx11;
always@(posedge clk) shiftreg2713[/*idx2612=*/ 7:1] <= shiftreg2713[/*idx2612=*/ 6:0];
wire [31:0] v2712 = shiftreg2713[/*idx2612=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 7][11] = tloop11delay[7];
assign v0_addr_input[/*idx2612=*/ 7][11] = {v2712[3:0]};
wire[31:0] v2714 = v0_rd_data[/*idx2612=*/ 7];
assign v0_rd_en_input[/*idx2612=*/ 7][11] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2715 = /*idx2612=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2717[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2717[0] <= v2714;
always@(posedge clk) shiftreg2717[/*idx13=*/ 11:1] <= shiftreg2717[/*idx13=*/ 10:0];
wire [31:0] v2716 = shiftreg2717[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2718 = v1_rd_data[/*idx2612=*/ 7][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 7][/*idx13=*/ 11][0] = tloop2696delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2719;
mult mult2720(v2719,
v2716,
v2718,
tloop2696delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2721 = v2611[/*idx2612=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2722;
add add2723(v2722,
v2719,
v2721,
tloop2696delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[8] = v2722;

//TerminatorOp

//} Unrolled body 7 of loop2612.
//DEBUG: /*idx2612=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop2612.
//DEBUG: /*idx2612=*/ 4'd8, expected 8
//printTimeOffset
reg tloop2710delay[3:0] = '{default:0} ;
always@(*) tloop2710delay[0] <= tloop2710;
generate
genvar i2725;

for(i2725 = 1; i2725<= 3; i2725= i2725 + 1) begin
always@(posedge clk) begin
tloop2710delay[i2725] <= tloop2710delay[i2725-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2724 = tloop2710delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2727[/*idx2612=*/ 8:0] = '{default:0};
always@(*) shiftreg2727[0] <= idx11;
always@(posedge clk) shiftreg2727[/*idx2612=*/ 8:1] <= shiftreg2727[/*idx2612=*/ 7:0];
wire [31:0] v2726 = shiftreg2727[/*idx2612=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 8][11] = tloop11delay[8];
assign v0_addr_input[/*idx2612=*/ 8][11] = {v2726[3:0]};
wire[31:0] v2728 = v0_rd_data[/*idx2612=*/ 8];
assign v0_rd_en_input[/*idx2612=*/ 8][11] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2729 = /*idx2612=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2731[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2731[0] <= v2728;
always@(posedge clk) shiftreg2731[/*idx13=*/ 11:1] <= shiftreg2731[/*idx13=*/ 10:0];
wire [31:0] v2730 = shiftreg2731[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2732 = v1_rd_data[/*idx2612=*/ 8][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 8][/*idx13=*/ 11][0] = tloop2710delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2733;
mult mult2734(v2733,
v2730,
v2732,
tloop2710delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2735 = v2611[/*idx2612=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2736;
add add2737(v2736,
v2733,
v2735,
tloop2710delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[9] = v2736;

//TerminatorOp

//} Unrolled body 8 of loop2612.
//DEBUG: /*idx2612=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop2612.
//DEBUG: /*idx2612=*/ 4'd9, expected 9
//printTimeOffset
reg tloop2724delay[3:0] = '{default:0} ;
always@(*) tloop2724delay[0] <= tloop2724;
generate
genvar i2739;

for(i2739 = 1; i2739<= 3; i2739= i2739 + 1) begin
always@(posedge clk) begin
tloop2724delay[i2739] <= tloop2724delay[i2739-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2738 = tloop2724delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2741[/*idx2612=*/ 9:0] = '{default:0};
always@(*) shiftreg2741[0] <= idx11;
always@(posedge clk) shiftreg2741[/*idx2612=*/ 9:1] <= shiftreg2741[/*idx2612=*/ 8:0];
wire [31:0] v2740 = shiftreg2741[/*idx2612=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 9][11] = tloop11delay[9];
assign v0_addr_input[/*idx2612=*/ 9][11] = {v2740[3:0]};
wire[31:0] v2742 = v0_rd_data[/*idx2612=*/ 9];
assign v0_rd_en_input[/*idx2612=*/ 9][11] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2743 = /*idx2612=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2745[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2745[0] <= v2742;
always@(posedge clk) shiftreg2745[/*idx13=*/ 11:1] <= shiftreg2745[/*idx13=*/ 10:0];
wire [31:0] v2744 = shiftreg2745[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2746 = v1_rd_data[/*idx2612=*/ 9][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 9][/*idx13=*/ 11][0] = tloop2724delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2747;
mult mult2748(v2747,
v2744,
v2746,
tloop2724delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2749 = v2611[/*idx2612=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2750;
add add2751(v2750,
v2747,
v2749,
tloop2724delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[10] = v2750;

//TerminatorOp

//} Unrolled body 9 of loop2612.
//DEBUG: /*idx2612=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop2612.
//DEBUG: /*idx2612=*/ 4'd10, expected 10
//printTimeOffset
reg tloop2738delay[3:0] = '{default:0} ;
always@(*) tloop2738delay[0] <= tloop2738;
generate
genvar i2753;

for(i2753 = 1; i2753<= 3; i2753= i2753 + 1) begin
always@(posedge clk) begin
tloop2738delay[i2753] <= tloop2738delay[i2753-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2752 = tloop2738delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2755[/*idx2612=*/ 10:0] = '{default:0};
always@(*) shiftreg2755[0] <= idx11;
always@(posedge clk) shiftreg2755[/*idx2612=*/ 10:1] <= shiftreg2755[/*idx2612=*/ 9:0];
wire [31:0] v2754 = shiftreg2755[/*idx2612=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 10][11] = tloop11delay[10];
assign v0_addr_input[/*idx2612=*/ 10][11] = {v2754[3:0]};
wire[31:0] v2756 = v0_rd_data[/*idx2612=*/ 10];
assign v0_rd_en_input[/*idx2612=*/ 10][11] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2757 = /*idx2612=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2759[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2759[0] <= v2756;
always@(posedge clk) shiftreg2759[/*idx13=*/ 11:1] <= shiftreg2759[/*idx13=*/ 10:0];
wire [31:0] v2758 = shiftreg2759[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2760 = v1_rd_data[/*idx2612=*/ 10][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 10][/*idx13=*/ 11][0] = tloop2738delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2761;
mult mult2762(v2761,
v2758,
v2760,
tloop2738delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2763 = v2611[/*idx2612=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2764;
add add2765(v2764,
v2761,
v2763,
tloop2738delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[11] = v2764;

//TerminatorOp

//} Unrolled body 10 of loop2612.
//DEBUG: /*idx2612=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop2612.
//DEBUG: /*idx2612=*/ 4'd11, expected 11
//printTimeOffset
reg tloop2752delay[3:0] = '{default:0} ;
always@(*) tloop2752delay[0] <= tloop2752;
generate
genvar i2767;

for(i2767 = 1; i2767<= 3; i2767= i2767 + 1) begin
always@(posedge clk) begin
tloop2752delay[i2767] <= tloop2752delay[i2767-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2766 = tloop2752delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2769[/*idx2612=*/ 11:0] = '{default:0};
always@(*) shiftreg2769[0] <= idx11;
always@(posedge clk) shiftreg2769[/*idx2612=*/ 11:1] <= shiftreg2769[/*idx2612=*/ 10:0];
wire [31:0] v2768 = shiftreg2769[/*idx2612=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 11][11] = tloop11delay[11];
assign v0_addr_input[/*idx2612=*/ 11][11] = {v2768[3:0]};
wire[31:0] v2770 = v0_rd_data[/*idx2612=*/ 11];
assign v0_rd_en_input[/*idx2612=*/ 11][11] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2771 = /*idx2612=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2773[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2773[0] <= v2770;
always@(posedge clk) shiftreg2773[/*idx13=*/ 11:1] <= shiftreg2773[/*idx13=*/ 10:0];
wire [31:0] v2772 = shiftreg2773[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2774 = v1_rd_data[/*idx2612=*/ 11][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 11][/*idx13=*/ 11][0] = tloop2752delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2775;
mult mult2776(v2775,
v2772,
v2774,
tloop2752delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2777 = v2611[/*idx2612=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2778;
add add2779(v2778,
v2775,
v2777,
tloop2752delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[12] = v2778;

//TerminatorOp

//} Unrolled body 11 of loop2612.
//DEBUG: /*idx2612=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop2612.
//DEBUG: /*idx2612=*/ 4'd12, expected 12
//printTimeOffset
reg tloop2766delay[3:0] = '{default:0} ;
always@(*) tloop2766delay[0] <= tloop2766;
generate
genvar i2781;

for(i2781 = 1; i2781<= 3; i2781= i2781 + 1) begin
always@(posedge clk) begin
tloop2766delay[i2781] <= tloop2766delay[i2781-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2780 = tloop2766delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2783[/*idx2612=*/ 12:0] = '{default:0};
always@(*) shiftreg2783[0] <= idx11;
always@(posedge clk) shiftreg2783[/*idx2612=*/ 12:1] <= shiftreg2783[/*idx2612=*/ 11:0];
wire [31:0] v2782 = shiftreg2783[/*idx2612=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 12][11] = tloop11delay[12];
assign v0_addr_input[/*idx2612=*/ 12][11] = {v2782[3:0]};
wire[31:0] v2784 = v0_rd_data[/*idx2612=*/ 12];
assign v0_rd_en_input[/*idx2612=*/ 12][11] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2785 = /*idx2612=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2787[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2787[0] <= v2784;
always@(posedge clk) shiftreg2787[/*idx13=*/ 11:1] <= shiftreg2787[/*idx13=*/ 10:0];
wire [31:0] v2786 = shiftreg2787[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2788 = v1_rd_data[/*idx2612=*/ 12][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 12][/*idx13=*/ 11][0] = tloop2766delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2789;
mult mult2790(v2789,
v2786,
v2788,
tloop2766delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2791 = v2611[/*idx2612=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2792;
add add2793(v2792,
v2789,
v2791,
tloop2766delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[13] = v2792;

//TerminatorOp

//} Unrolled body 12 of loop2612.
//DEBUG: /*idx2612=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop2612.
//DEBUG: /*idx2612=*/ 4'd13, expected 13
//printTimeOffset
reg tloop2780delay[3:0] = '{default:0} ;
always@(*) tloop2780delay[0] <= tloop2780;
generate
genvar i2795;

for(i2795 = 1; i2795<= 3; i2795= i2795 + 1) begin
always@(posedge clk) begin
tloop2780delay[i2795] <= tloop2780delay[i2795-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2794 = tloop2780delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2797[/*idx2612=*/ 13:0] = '{default:0};
always@(*) shiftreg2797[0] <= idx11;
always@(posedge clk) shiftreg2797[/*idx2612=*/ 13:1] <= shiftreg2797[/*idx2612=*/ 12:0];
wire [31:0] v2796 = shiftreg2797[/*idx2612=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 13][11] = tloop11delay[13];
assign v0_addr_input[/*idx2612=*/ 13][11] = {v2796[3:0]};
wire[31:0] v2798 = v0_rd_data[/*idx2612=*/ 13];
assign v0_rd_en_input[/*idx2612=*/ 13][11] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2799 = /*idx2612=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2801[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2801[0] <= v2798;
always@(posedge clk) shiftreg2801[/*idx13=*/ 11:1] <= shiftreg2801[/*idx13=*/ 10:0];
wire [31:0] v2800 = shiftreg2801[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2802 = v1_rd_data[/*idx2612=*/ 13][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 13][/*idx13=*/ 11][0] = tloop2780delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2803;
mult mult2804(v2803,
v2800,
v2802,
tloop2780delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2805 = v2611[/*idx2612=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2806;
add add2807(v2806,
v2803,
v2805,
tloop2780delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[14] = v2806;

//TerminatorOp

//} Unrolled body 13 of loop2612.
//DEBUG: /*idx2612=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop2612.
//DEBUG: /*idx2612=*/ 4'd14, expected 14
//printTimeOffset
reg tloop2794delay[3:0] = '{default:0} ;
always@(*) tloop2794delay[0] <= tloop2794;
generate
genvar i2809;

for(i2809 = 1; i2809<= 3; i2809= i2809 + 1) begin
always@(posedge clk) begin
tloop2794delay[i2809] <= tloop2794delay[i2809-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2808 = tloop2794delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2811[/*idx2612=*/ 14:0] = '{default:0};
always@(*) shiftreg2811[0] <= idx11;
always@(posedge clk) shiftreg2811[/*idx2612=*/ 14:1] <= shiftreg2811[/*idx2612=*/ 13:0];
wire [31:0] v2810 = shiftreg2811[/*idx2612=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 14][11] = tloop11delay[14];
assign v0_addr_input[/*idx2612=*/ 14][11] = {v2810[3:0]};
wire[31:0] v2812 = v0_rd_data[/*idx2612=*/ 14];
assign v0_rd_en_input[/*idx2612=*/ 14][11] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2813 = /*idx2612=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2815[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2815[0] <= v2812;
always@(posedge clk) shiftreg2815[/*idx13=*/ 11:1] <= shiftreg2815[/*idx13=*/ 10:0];
wire [31:0] v2814 = shiftreg2815[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2816 = v1_rd_data[/*idx2612=*/ 14][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 14][/*idx13=*/ 11][0] = tloop2794delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2817;
mult mult2818(v2817,
v2814,
v2816,
tloop2794delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2819 = v2611[/*idx2612=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2820;
add add2821(v2820,
v2817,
v2819,
tloop2794delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[15] = v2820;

//TerminatorOp

//} Unrolled body 14 of loop2612.
//DEBUG: /*idx2612=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop2612.
//DEBUG: /*idx2612=*/ 4'd15, expected 15
//printTimeOffset
reg tloop2808delay[3:0] = '{default:0} ;
always@(*) tloop2808delay[0] <= tloop2808;
generate
genvar i2823;

for(i2823 = 1; i2823<= 3; i2823= i2823 + 1) begin
always@(posedge clk) begin
tloop2808delay[i2823] <= tloop2808delay[i2823-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2822 = tloop2808delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2825[/*idx2612=*/ 15:0] = '{default:0};
always@(*) shiftreg2825[0] <= idx11;
always@(posedge clk) shiftreg2825[/*idx2612=*/ 15:1] <= shiftreg2825[/*idx2612=*/ 14:0];
wire [31:0] v2824 = shiftreg2825[/*idx2612=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2612=*/ 15][11] = tloop11delay[15];
assign v0_addr_input[/*idx2612=*/ 15][11] = {v2824[3:0]};
wire[31:0] v2826 = v0_rd_data[/*idx2612=*/ 15];
assign v0_rd_en_input[/*idx2612=*/ 15][11] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2827 = /*idx2612=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2829[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2829[0] <= v2826;
always@(posedge clk) shiftreg2829[/*idx13=*/ 11:1] <= shiftreg2829[/*idx13=*/ 10:0];
wire [31:0] v2828 = shiftreg2829[/*idx13=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2830 = v1_rd_data[/*idx2612=*/ 15][/*idx13=*/ 11];
assign v1_rd_en_input[/*idx2612=*/ 15][/*idx13=*/ 11][0] = tloop2808delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2831;
mult mult2832(v2831,
v2828,
v2830,
tloop2808delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2833 = v2611[/*idx2612=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2834;
add add2835(v2834,
v2831,
v2833,
tloop2808delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2611[16] = v2834;

//TerminatorOp

//} Unrolled body 15 of loop2612.
//DEBUG: /*idx2612=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t2836;
assign t2836 = tloop2822;
//printTimeOffset
reg t2836delay[3:0] = '{default:0} ;
always@(*) t2836delay[0] <= t2836;
generate
genvar i2837;

for(i2837 = 1; i2837<= 3; i2837= i2837 + 1) begin
always@(posedge clk) begin
t2836delay[i2837] <= t2836delay[i2837-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v2838 = v2611[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg2840[/*idx13=*/ 11:0] = '{default:0};
always@(*) shiftreg2840[0] <= idx11;
always@(posedge clk) shiftreg2840[/*idx13=*/ 11:1] <= shiftreg2840[/*idx13=*/ 10:0];
wire [31:0] v2839 = shiftreg2840[/*idx13=*/ 11];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg2842[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg2842[0] <= v2839;
always@(posedge clk) shiftreg2842[/*v10=*/ 16:1] <= shiftreg2842[/*v10=*/ 15:0];
wire [31:0] v2841 = shiftreg2842[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg2844[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg2844[0] <= v2841;
always@(posedge clk) shiftreg2844[/*v8=*/ 3:1] <= shiftreg2844[/*v8=*/ 2:0];
wire [31:0] v2843 = shiftreg2844[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 11][0] = t2836delay[3];
assign v2_addr_input[/*idx13=*/ 11][0] = {v2843[3:0]};
assign v2_wr_en_input[/*idx13=*/ 11][0] = t2836delay[3];
assign v2_wr_data_valid[/*idx13=*/ 11][0] = t2836delay[3];
assign v2_wr_data_input[/*idx13=*/ 11][0] = v2838;


//TerminatorOp

//} Unrolled body 11 of loop13.
//DEBUG: /*idx13=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop13.
//DEBUG: /*idx13=*/ 4'd12, expected 12
//printTimeOffset
reg tloop2609delay[3:0] = '{default:0} ;
always@(*) tloop2609delay[0] <= tloop2609;
generate
genvar i2846;

for(i2846 = 1; i2846<= 3; i2846= i2846 + 1) begin
always@(posedge clk) begin
tloop2609delay[i2846] <= tloop2609delay[i2846-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop2845 = tloop2609delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v2847[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v2847[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop2848.
//DEBUG: /*idx2848=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2849 = tloop2609delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2851[/*idx2848=*/ 0:0] = '{default:0};
always@(*) shiftreg2851[0] <= idx11;
wire [31:0] v2850 = shiftreg2851[/*idx2848=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 0][12] = tloop11delay[0];
assign v0_addr_input[/*idx2848=*/ 0][12] = {v2850[3:0]};
wire[31:0] v2852 = v0_rd_data[/*idx2848=*/ 0];
assign v0_rd_en_input[/*idx2848=*/ 0][12] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2853 = /*idx2848=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2855[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2855[0] <= v2852;
always@(posedge clk) shiftreg2855[/*idx13=*/ 12:1] <= shiftreg2855[/*idx13=*/ 11:0];
wire [31:0] v2854 = shiftreg2855[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2856 = v1_rd_data[/*idx2848=*/ 0][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 0][/*idx13=*/ 12][0] = tloop2609delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2857;
mult mult2858(v2857,
v2854,
v2856,
tloop2609delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2859 = v2847[/*idx2848=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2860;
add add2861(v2860,
v2857,
v2859,
tloop2609delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[1] = v2860;

//TerminatorOp

//} Unrolled body 0 of loop2848.
//DEBUG: /*idx2848=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop2848.
//DEBUG: /*idx2848=*/ 1'd1, expected 1
//printTimeOffset
reg tloop2849delay[3:0] = '{default:0} ;
always@(*) tloop2849delay[0] <= tloop2849;
generate
genvar i2863;

for(i2863 = 1; i2863<= 3; i2863= i2863 + 1) begin
always@(posedge clk) begin
tloop2849delay[i2863] <= tloop2849delay[i2863-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2862 = tloop2849delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2865[/*idx2848=*/ 1:0] = '{default:0};
always@(*) shiftreg2865[0] <= idx11;
always@(posedge clk) shiftreg2865[/*idx2848=*/ 1:1] <= shiftreg2865[/*idx2848=*/ 0:0];
wire [31:0] v2864 = shiftreg2865[/*idx2848=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 1][12] = tloop11delay[1];
assign v0_addr_input[/*idx2848=*/ 1][12] = {v2864[3:0]};
wire[31:0] v2866 = v0_rd_data[/*idx2848=*/ 1];
assign v0_rd_en_input[/*idx2848=*/ 1][12] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2867 = /*idx2848=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2869[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2869[0] <= v2866;
always@(posedge clk) shiftreg2869[/*idx13=*/ 12:1] <= shiftreg2869[/*idx13=*/ 11:0];
wire [31:0] v2868 = shiftreg2869[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2870 = v1_rd_data[/*idx2848=*/ 1][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 1][/*idx13=*/ 12][0] = tloop2849delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2871;
mult mult2872(v2871,
v2868,
v2870,
tloop2849delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2873 = v2847[/*idx2848=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2874;
add add2875(v2874,
v2871,
v2873,
tloop2849delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[2] = v2874;

//TerminatorOp

//} Unrolled body 1 of loop2848.
//DEBUG: /*idx2848=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop2848.
//DEBUG: /*idx2848=*/ 2'd2, expected 2
//printTimeOffset
reg tloop2862delay[3:0] = '{default:0} ;
always@(*) tloop2862delay[0] <= tloop2862;
generate
genvar i2877;

for(i2877 = 1; i2877<= 3; i2877= i2877 + 1) begin
always@(posedge clk) begin
tloop2862delay[i2877] <= tloop2862delay[i2877-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2876 = tloop2862delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2879[/*idx2848=*/ 2:0] = '{default:0};
always@(*) shiftreg2879[0] <= idx11;
always@(posedge clk) shiftreg2879[/*idx2848=*/ 2:1] <= shiftreg2879[/*idx2848=*/ 1:0];
wire [31:0] v2878 = shiftreg2879[/*idx2848=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 2][12] = tloop11delay[2];
assign v0_addr_input[/*idx2848=*/ 2][12] = {v2878[3:0]};
wire[31:0] v2880 = v0_rd_data[/*idx2848=*/ 2];
assign v0_rd_en_input[/*idx2848=*/ 2][12] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2881 = /*idx2848=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2883[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2883[0] <= v2880;
always@(posedge clk) shiftreg2883[/*idx13=*/ 12:1] <= shiftreg2883[/*idx13=*/ 11:0];
wire [31:0] v2882 = shiftreg2883[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2884 = v1_rd_data[/*idx2848=*/ 2][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 2][/*idx13=*/ 12][0] = tloop2862delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2885;
mult mult2886(v2885,
v2882,
v2884,
tloop2862delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2887 = v2847[/*idx2848=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2888;
add add2889(v2888,
v2885,
v2887,
tloop2862delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[3] = v2888;

//TerminatorOp

//} Unrolled body 2 of loop2848.
//DEBUG: /*idx2848=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop2848.
//DEBUG: /*idx2848=*/ 2'd3, expected 3
//printTimeOffset
reg tloop2876delay[3:0] = '{default:0} ;
always@(*) tloop2876delay[0] <= tloop2876;
generate
genvar i2891;

for(i2891 = 1; i2891<= 3; i2891= i2891 + 1) begin
always@(posedge clk) begin
tloop2876delay[i2891] <= tloop2876delay[i2891-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2890 = tloop2876delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2893[/*idx2848=*/ 3:0] = '{default:0};
always@(*) shiftreg2893[0] <= idx11;
always@(posedge clk) shiftreg2893[/*idx2848=*/ 3:1] <= shiftreg2893[/*idx2848=*/ 2:0];
wire [31:0] v2892 = shiftreg2893[/*idx2848=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 3][12] = tloop11delay[3];
assign v0_addr_input[/*idx2848=*/ 3][12] = {v2892[3:0]};
wire[31:0] v2894 = v0_rd_data[/*idx2848=*/ 3];
assign v0_rd_en_input[/*idx2848=*/ 3][12] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2895 = /*idx2848=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2897[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2897[0] <= v2894;
always@(posedge clk) shiftreg2897[/*idx13=*/ 12:1] <= shiftreg2897[/*idx13=*/ 11:0];
wire [31:0] v2896 = shiftreg2897[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2898 = v1_rd_data[/*idx2848=*/ 3][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 3][/*idx13=*/ 12][0] = tloop2876delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2899;
mult mult2900(v2899,
v2896,
v2898,
tloop2876delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2901 = v2847[/*idx2848=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2902;
add add2903(v2902,
v2899,
v2901,
tloop2876delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[4] = v2902;

//TerminatorOp

//} Unrolled body 3 of loop2848.
//DEBUG: /*idx2848=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop2848.
//DEBUG: /*idx2848=*/ 3'd4, expected 4
//printTimeOffset
reg tloop2890delay[3:0] = '{default:0} ;
always@(*) tloop2890delay[0] <= tloop2890;
generate
genvar i2905;

for(i2905 = 1; i2905<= 3; i2905= i2905 + 1) begin
always@(posedge clk) begin
tloop2890delay[i2905] <= tloop2890delay[i2905-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2904 = tloop2890delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2907[/*idx2848=*/ 4:0] = '{default:0};
always@(*) shiftreg2907[0] <= idx11;
always@(posedge clk) shiftreg2907[/*idx2848=*/ 4:1] <= shiftreg2907[/*idx2848=*/ 3:0];
wire [31:0] v2906 = shiftreg2907[/*idx2848=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 4][12] = tloop11delay[4];
assign v0_addr_input[/*idx2848=*/ 4][12] = {v2906[3:0]};
wire[31:0] v2908 = v0_rd_data[/*idx2848=*/ 4];
assign v0_rd_en_input[/*idx2848=*/ 4][12] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2909 = /*idx2848=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2911[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2911[0] <= v2908;
always@(posedge clk) shiftreg2911[/*idx13=*/ 12:1] <= shiftreg2911[/*idx13=*/ 11:0];
wire [31:0] v2910 = shiftreg2911[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2912 = v1_rd_data[/*idx2848=*/ 4][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 4][/*idx13=*/ 12][0] = tloop2890delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2913;
mult mult2914(v2913,
v2910,
v2912,
tloop2890delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2915 = v2847[/*idx2848=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2916;
add add2917(v2916,
v2913,
v2915,
tloop2890delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[5] = v2916;

//TerminatorOp

//} Unrolled body 4 of loop2848.
//DEBUG: /*idx2848=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop2848.
//DEBUG: /*idx2848=*/ 3'd5, expected 5
//printTimeOffset
reg tloop2904delay[3:0] = '{default:0} ;
always@(*) tloop2904delay[0] <= tloop2904;
generate
genvar i2919;

for(i2919 = 1; i2919<= 3; i2919= i2919 + 1) begin
always@(posedge clk) begin
tloop2904delay[i2919] <= tloop2904delay[i2919-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2918 = tloop2904delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2921[/*idx2848=*/ 5:0] = '{default:0};
always@(*) shiftreg2921[0] <= idx11;
always@(posedge clk) shiftreg2921[/*idx2848=*/ 5:1] <= shiftreg2921[/*idx2848=*/ 4:0];
wire [31:0] v2920 = shiftreg2921[/*idx2848=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 5][12] = tloop11delay[5];
assign v0_addr_input[/*idx2848=*/ 5][12] = {v2920[3:0]};
wire[31:0] v2922 = v0_rd_data[/*idx2848=*/ 5];
assign v0_rd_en_input[/*idx2848=*/ 5][12] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2923 = /*idx2848=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2925[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2925[0] <= v2922;
always@(posedge clk) shiftreg2925[/*idx13=*/ 12:1] <= shiftreg2925[/*idx13=*/ 11:0];
wire [31:0] v2924 = shiftreg2925[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2926 = v1_rd_data[/*idx2848=*/ 5][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 5][/*idx13=*/ 12][0] = tloop2904delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2927;
mult mult2928(v2927,
v2924,
v2926,
tloop2904delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2929 = v2847[/*idx2848=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2930;
add add2931(v2930,
v2927,
v2929,
tloop2904delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[6] = v2930;

//TerminatorOp

//} Unrolled body 5 of loop2848.
//DEBUG: /*idx2848=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop2848.
//DEBUG: /*idx2848=*/ 3'd6, expected 6
//printTimeOffset
reg tloop2918delay[3:0] = '{default:0} ;
always@(*) tloop2918delay[0] <= tloop2918;
generate
genvar i2933;

for(i2933 = 1; i2933<= 3; i2933= i2933 + 1) begin
always@(posedge clk) begin
tloop2918delay[i2933] <= tloop2918delay[i2933-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2932 = tloop2918delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2935[/*idx2848=*/ 6:0] = '{default:0};
always@(*) shiftreg2935[0] <= idx11;
always@(posedge clk) shiftreg2935[/*idx2848=*/ 6:1] <= shiftreg2935[/*idx2848=*/ 5:0];
wire [31:0] v2934 = shiftreg2935[/*idx2848=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 6][12] = tloop11delay[6];
assign v0_addr_input[/*idx2848=*/ 6][12] = {v2934[3:0]};
wire[31:0] v2936 = v0_rd_data[/*idx2848=*/ 6];
assign v0_rd_en_input[/*idx2848=*/ 6][12] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2937 = /*idx2848=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2939[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2939[0] <= v2936;
always@(posedge clk) shiftreg2939[/*idx13=*/ 12:1] <= shiftreg2939[/*idx13=*/ 11:0];
wire [31:0] v2938 = shiftreg2939[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2940 = v1_rd_data[/*idx2848=*/ 6][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 6][/*idx13=*/ 12][0] = tloop2918delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2941;
mult mult2942(v2941,
v2938,
v2940,
tloop2918delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2943 = v2847[/*idx2848=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2944;
add add2945(v2944,
v2941,
v2943,
tloop2918delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[7] = v2944;

//TerminatorOp

//} Unrolled body 6 of loop2848.
//DEBUG: /*idx2848=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop2848.
//DEBUG: /*idx2848=*/ 3'd7, expected 7
//printTimeOffset
reg tloop2932delay[3:0] = '{default:0} ;
always@(*) tloop2932delay[0] <= tloop2932;
generate
genvar i2947;

for(i2947 = 1; i2947<= 3; i2947= i2947 + 1) begin
always@(posedge clk) begin
tloop2932delay[i2947] <= tloop2932delay[i2947-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2946 = tloop2932delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2949[/*idx2848=*/ 7:0] = '{default:0};
always@(*) shiftreg2949[0] <= idx11;
always@(posedge clk) shiftreg2949[/*idx2848=*/ 7:1] <= shiftreg2949[/*idx2848=*/ 6:0];
wire [31:0] v2948 = shiftreg2949[/*idx2848=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 7][12] = tloop11delay[7];
assign v0_addr_input[/*idx2848=*/ 7][12] = {v2948[3:0]};
wire[31:0] v2950 = v0_rd_data[/*idx2848=*/ 7];
assign v0_rd_en_input[/*idx2848=*/ 7][12] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2951 = /*idx2848=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2953[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2953[0] <= v2950;
always@(posedge clk) shiftreg2953[/*idx13=*/ 12:1] <= shiftreg2953[/*idx13=*/ 11:0];
wire [31:0] v2952 = shiftreg2953[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2954 = v1_rd_data[/*idx2848=*/ 7][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 7][/*idx13=*/ 12][0] = tloop2932delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2955;
mult mult2956(v2955,
v2952,
v2954,
tloop2932delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2957 = v2847[/*idx2848=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2958;
add add2959(v2958,
v2955,
v2957,
tloop2932delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[8] = v2958;

//TerminatorOp

//} Unrolled body 7 of loop2848.
//DEBUG: /*idx2848=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop2848.
//DEBUG: /*idx2848=*/ 4'd8, expected 8
//printTimeOffset
reg tloop2946delay[3:0] = '{default:0} ;
always@(*) tloop2946delay[0] <= tloop2946;
generate
genvar i2961;

for(i2961 = 1; i2961<= 3; i2961= i2961 + 1) begin
always@(posedge clk) begin
tloop2946delay[i2961] <= tloop2946delay[i2961-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2960 = tloop2946delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2963[/*idx2848=*/ 8:0] = '{default:0};
always@(*) shiftreg2963[0] <= idx11;
always@(posedge clk) shiftreg2963[/*idx2848=*/ 8:1] <= shiftreg2963[/*idx2848=*/ 7:0];
wire [31:0] v2962 = shiftreg2963[/*idx2848=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 8][12] = tloop11delay[8];
assign v0_addr_input[/*idx2848=*/ 8][12] = {v2962[3:0]};
wire[31:0] v2964 = v0_rd_data[/*idx2848=*/ 8];
assign v0_rd_en_input[/*idx2848=*/ 8][12] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2965 = /*idx2848=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2967[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2967[0] <= v2964;
always@(posedge clk) shiftreg2967[/*idx13=*/ 12:1] <= shiftreg2967[/*idx13=*/ 11:0];
wire [31:0] v2966 = shiftreg2967[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2968 = v1_rd_data[/*idx2848=*/ 8][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 8][/*idx13=*/ 12][0] = tloop2946delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2969;
mult mult2970(v2969,
v2966,
v2968,
tloop2946delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2971 = v2847[/*idx2848=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2972;
add add2973(v2972,
v2969,
v2971,
tloop2946delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[9] = v2972;

//TerminatorOp

//} Unrolled body 8 of loop2848.
//DEBUG: /*idx2848=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop2848.
//DEBUG: /*idx2848=*/ 4'd9, expected 9
//printTimeOffset
reg tloop2960delay[3:0] = '{default:0} ;
always@(*) tloop2960delay[0] <= tloop2960;
generate
genvar i2975;

for(i2975 = 1; i2975<= 3; i2975= i2975 + 1) begin
always@(posedge clk) begin
tloop2960delay[i2975] <= tloop2960delay[i2975-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2974 = tloop2960delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2977[/*idx2848=*/ 9:0] = '{default:0};
always@(*) shiftreg2977[0] <= idx11;
always@(posedge clk) shiftreg2977[/*idx2848=*/ 9:1] <= shiftreg2977[/*idx2848=*/ 8:0];
wire [31:0] v2976 = shiftreg2977[/*idx2848=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 9][12] = tloop11delay[9];
assign v0_addr_input[/*idx2848=*/ 9][12] = {v2976[3:0]};
wire[31:0] v2978 = v0_rd_data[/*idx2848=*/ 9];
assign v0_rd_en_input[/*idx2848=*/ 9][12] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2979 = /*idx2848=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2981[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2981[0] <= v2978;
always@(posedge clk) shiftreg2981[/*idx13=*/ 12:1] <= shiftreg2981[/*idx13=*/ 11:0];
wire [31:0] v2980 = shiftreg2981[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2982 = v1_rd_data[/*idx2848=*/ 9][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 9][/*idx13=*/ 12][0] = tloop2960delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2983;
mult mult2984(v2983,
v2980,
v2982,
tloop2960delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2985 = v2847[/*idx2848=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v2986;
add add2987(v2986,
v2983,
v2985,
tloop2960delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[10] = v2986;

//TerminatorOp

//} Unrolled body 9 of loop2848.
//DEBUG: /*idx2848=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop2848.
//DEBUG: /*idx2848=*/ 4'd10, expected 10
//printTimeOffset
reg tloop2974delay[3:0] = '{default:0} ;
always@(*) tloop2974delay[0] <= tloop2974;
generate
genvar i2989;

for(i2989 = 1; i2989<= 3; i2989= i2989 + 1) begin
always@(posedge clk) begin
tloop2974delay[i2989] <= tloop2974delay[i2989-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop2988 = tloop2974delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg2991[/*idx2848=*/ 10:0] = '{default:0};
always@(*) shiftreg2991[0] <= idx11;
always@(posedge clk) shiftreg2991[/*idx2848=*/ 10:1] <= shiftreg2991[/*idx2848=*/ 9:0];
wire [31:0] v2990 = shiftreg2991[/*idx2848=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 10][12] = tloop11delay[10];
assign v0_addr_input[/*idx2848=*/ 10][12] = {v2990[3:0]};
wire[31:0] v2992 = v0_rd_data[/*idx2848=*/ 10];
assign v0_rd_en_input[/*idx2848=*/ 10][12] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v2993 = /*idx2848=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg2995[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg2995[0] <= v2992;
always@(posedge clk) shiftreg2995[/*idx13=*/ 12:1] <= shiftreg2995[/*idx13=*/ 11:0];
wire [31:0] v2994 = shiftreg2995[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v2996 = v1_rd_data[/*idx2848=*/ 10][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 10][/*idx13=*/ 12][0] = tloop2974delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v2997;
mult mult2998(v2997,
v2994,
v2996,
tloop2974delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v2999 = v2847[/*idx2848=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3000;
add add3001(v3000,
v2997,
v2999,
tloop2974delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[11] = v3000;

//TerminatorOp

//} Unrolled body 10 of loop2848.
//DEBUG: /*idx2848=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop2848.
//DEBUG: /*idx2848=*/ 4'd11, expected 11
//printTimeOffset
reg tloop2988delay[3:0] = '{default:0} ;
always@(*) tloop2988delay[0] <= tloop2988;
generate
genvar i3003;

for(i3003 = 1; i3003<= 3; i3003= i3003 + 1) begin
always@(posedge clk) begin
tloop2988delay[i3003] <= tloop2988delay[i3003-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3002 = tloop2988delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3005[/*idx2848=*/ 11:0] = '{default:0};
always@(*) shiftreg3005[0] <= idx11;
always@(posedge clk) shiftreg3005[/*idx2848=*/ 11:1] <= shiftreg3005[/*idx2848=*/ 10:0];
wire [31:0] v3004 = shiftreg3005[/*idx2848=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 11][12] = tloop11delay[11];
assign v0_addr_input[/*idx2848=*/ 11][12] = {v3004[3:0]};
wire[31:0] v3006 = v0_rd_data[/*idx2848=*/ 11];
assign v0_rd_en_input[/*idx2848=*/ 11][12] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3007 = /*idx2848=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3009[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg3009[0] <= v3006;
always@(posedge clk) shiftreg3009[/*idx13=*/ 12:1] <= shiftreg3009[/*idx13=*/ 11:0];
wire [31:0] v3008 = shiftreg3009[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3010 = v1_rd_data[/*idx2848=*/ 11][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 11][/*idx13=*/ 12][0] = tloop2988delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3011;
mult mult3012(v3011,
v3008,
v3010,
tloop2988delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3013 = v2847[/*idx2848=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3014;
add add3015(v3014,
v3011,
v3013,
tloop2988delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[12] = v3014;

//TerminatorOp

//} Unrolled body 11 of loop2848.
//DEBUG: /*idx2848=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop2848.
//DEBUG: /*idx2848=*/ 4'd12, expected 12
//printTimeOffset
reg tloop3002delay[3:0] = '{default:0} ;
always@(*) tloop3002delay[0] <= tloop3002;
generate
genvar i3017;

for(i3017 = 1; i3017<= 3; i3017= i3017 + 1) begin
always@(posedge clk) begin
tloop3002delay[i3017] <= tloop3002delay[i3017-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3016 = tloop3002delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3019[/*idx2848=*/ 12:0] = '{default:0};
always@(*) shiftreg3019[0] <= idx11;
always@(posedge clk) shiftreg3019[/*idx2848=*/ 12:1] <= shiftreg3019[/*idx2848=*/ 11:0];
wire [31:0] v3018 = shiftreg3019[/*idx2848=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 12][12] = tloop11delay[12];
assign v0_addr_input[/*idx2848=*/ 12][12] = {v3018[3:0]};
wire[31:0] v3020 = v0_rd_data[/*idx2848=*/ 12];
assign v0_rd_en_input[/*idx2848=*/ 12][12] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3021 = /*idx2848=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3023[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg3023[0] <= v3020;
always@(posedge clk) shiftreg3023[/*idx13=*/ 12:1] <= shiftreg3023[/*idx13=*/ 11:0];
wire [31:0] v3022 = shiftreg3023[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3024 = v1_rd_data[/*idx2848=*/ 12][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 12][/*idx13=*/ 12][0] = tloop3002delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3025;
mult mult3026(v3025,
v3022,
v3024,
tloop3002delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3027 = v2847[/*idx2848=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3028;
add add3029(v3028,
v3025,
v3027,
tloop3002delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[13] = v3028;

//TerminatorOp

//} Unrolled body 12 of loop2848.
//DEBUG: /*idx2848=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop2848.
//DEBUG: /*idx2848=*/ 4'd13, expected 13
//printTimeOffset
reg tloop3016delay[3:0] = '{default:0} ;
always@(*) tloop3016delay[0] <= tloop3016;
generate
genvar i3031;

for(i3031 = 1; i3031<= 3; i3031= i3031 + 1) begin
always@(posedge clk) begin
tloop3016delay[i3031] <= tloop3016delay[i3031-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3030 = tloop3016delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3033[/*idx2848=*/ 13:0] = '{default:0};
always@(*) shiftreg3033[0] <= idx11;
always@(posedge clk) shiftreg3033[/*idx2848=*/ 13:1] <= shiftreg3033[/*idx2848=*/ 12:0];
wire [31:0] v3032 = shiftreg3033[/*idx2848=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 13][12] = tloop11delay[13];
assign v0_addr_input[/*idx2848=*/ 13][12] = {v3032[3:0]};
wire[31:0] v3034 = v0_rd_data[/*idx2848=*/ 13];
assign v0_rd_en_input[/*idx2848=*/ 13][12] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3035 = /*idx2848=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3037[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg3037[0] <= v3034;
always@(posedge clk) shiftreg3037[/*idx13=*/ 12:1] <= shiftreg3037[/*idx13=*/ 11:0];
wire [31:0] v3036 = shiftreg3037[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3038 = v1_rd_data[/*idx2848=*/ 13][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 13][/*idx13=*/ 12][0] = tloop3016delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3039;
mult mult3040(v3039,
v3036,
v3038,
tloop3016delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3041 = v2847[/*idx2848=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3042;
add add3043(v3042,
v3039,
v3041,
tloop3016delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[14] = v3042;

//TerminatorOp

//} Unrolled body 13 of loop2848.
//DEBUG: /*idx2848=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop2848.
//DEBUG: /*idx2848=*/ 4'd14, expected 14
//printTimeOffset
reg tloop3030delay[3:0] = '{default:0} ;
always@(*) tloop3030delay[0] <= tloop3030;
generate
genvar i3045;

for(i3045 = 1; i3045<= 3; i3045= i3045 + 1) begin
always@(posedge clk) begin
tloop3030delay[i3045] <= tloop3030delay[i3045-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3044 = tloop3030delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3047[/*idx2848=*/ 14:0] = '{default:0};
always@(*) shiftreg3047[0] <= idx11;
always@(posedge clk) shiftreg3047[/*idx2848=*/ 14:1] <= shiftreg3047[/*idx2848=*/ 13:0];
wire [31:0] v3046 = shiftreg3047[/*idx2848=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 14][12] = tloop11delay[14];
assign v0_addr_input[/*idx2848=*/ 14][12] = {v3046[3:0]};
wire[31:0] v3048 = v0_rd_data[/*idx2848=*/ 14];
assign v0_rd_en_input[/*idx2848=*/ 14][12] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3049 = /*idx2848=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3051[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg3051[0] <= v3048;
always@(posedge clk) shiftreg3051[/*idx13=*/ 12:1] <= shiftreg3051[/*idx13=*/ 11:0];
wire [31:0] v3050 = shiftreg3051[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3052 = v1_rd_data[/*idx2848=*/ 14][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 14][/*idx13=*/ 12][0] = tloop3030delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3053;
mult mult3054(v3053,
v3050,
v3052,
tloop3030delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3055 = v2847[/*idx2848=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3056;
add add3057(v3056,
v3053,
v3055,
tloop3030delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[15] = v3056;

//TerminatorOp

//} Unrolled body 14 of loop2848.
//DEBUG: /*idx2848=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop2848.
//DEBUG: /*idx2848=*/ 4'd15, expected 15
//printTimeOffset
reg tloop3044delay[3:0] = '{default:0} ;
always@(*) tloop3044delay[0] <= tloop3044;
generate
genvar i3059;

for(i3059 = 1; i3059<= 3; i3059= i3059 + 1) begin
always@(posedge clk) begin
tloop3044delay[i3059] <= tloop3044delay[i3059-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3058 = tloop3044delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3061[/*idx2848=*/ 15:0] = '{default:0};
always@(*) shiftreg3061[0] <= idx11;
always@(posedge clk) shiftreg3061[/*idx2848=*/ 15:1] <= shiftreg3061[/*idx2848=*/ 14:0];
wire [31:0] v3060 = shiftreg3061[/*idx2848=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx2848=*/ 15][12] = tloop11delay[15];
assign v0_addr_input[/*idx2848=*/ 15][12] = {v3060[3:0]};
wire[31:0] v3062 = v0_rd_data[/*idx2848=*/ 15];
assign v0_rd_en_input[/*idx2848=*/ 15][12] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3063 = /*idx2848=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3065[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg3065[0] <= v3062;
always@(posedge clk) shiftreg3065[/*idx13=*/ 12:1] <= shiftreg3065[/*idx13=*/ 11:0];
wire [31:0] v3064 = shiftreg3065[/*idx13=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3066 = v1_rd_data[/*idx2848=*/ 15][/*idx13=*/ 12];
assign v1_rd_en_input[/*idx2848=*/ 15][/*idx13=*/ 12][0] = tloop3044delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3067;
mult mult3068(v3067,
v3064,
v3066,
tloop3044delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3069 = v2847[/*idx2848=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3070;
add add3071(v3070,
v3067,
v3069,
tloop3044delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v2847[16] = v3070;

//TerminatorOp

//} Unrolled body 15 of loop2848.
//DEBUG: /*idx2848=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t3072;
assign t3072 = tloop3058;
//printTimeOffset
reg t3072delay[3:0] = '{default:0} ;
always@(*) t3072delay[0] <= t3072;
generate
genvar i3073;

for(i3073 = 1; i3073<= 3; i3073= i3073 + 1) begin
always@(posedge clk) begin
t3072delay[i3073] <= t3072delay[i3073-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v3074 = v2847[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg3076[/*idx13=*/ 12:0] = '{default:0};
always@(*) shiftreg3076[0] <= idx11;
always@(posedge clk) shiftreg3076[/*idx13=*/ 12:1] <= shiftreg3076[/*idx13=*/ 11:0];
wire [31:0] v3075 = shiftreg3076[/*idx13=*/ 12];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg3078[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg3078[0] <= v3075;
always@(posedge clk) shiftreg3078[/*v10=*/ 16:1] <= shiftreg3078[/*v10=*/ 15:0];
wire [31:0] v3077 = shiftreg3078[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg3080[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg3080[0] <= v3077;
always@(posedge clk) shiftreg3080[/*v8=*/ 3:1] <= shiftreg3080[/*v8=*/ 2:0];
wire [31:0] v3079 = shiftreg3080[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 12][0] = t3072delay[3];
assign v2_addr_input[/*idx13=*/ 12][0] = {v3079[3:0]};
assign v2_wr_en_input[/*idx13=*/ 12][0] = t3072delay[3];
assign v2_wr_data_valid[/*idx13=*/ 12][0] = t3072delay[3];
assign v2_wr_data_input[/*idx13=*/ 12][0] = v3074;


//TerminatorOp

//} Unrolled body 12 of loop13.
//DEBUG: /*idx13=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop13.
//DEBUG: /*idx13=*/ 4'd13, expected 13
//printTimeOffset
reg tloop2845delay[3:0] = '{default:0} ;
always@(*) tloop2845delay[0] <= tloop2845;
generate
genvar i3082;

for(i3082 = 1; i3082<= 3; i3082= i3082 + 1) begin
always@(posedge clk) begin
tloop2845delay[i3082] <= tloop2845delay[i3082-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop3081 = tloop2845delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v3083[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v3083[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop3084.
//DEBUG: /*idx3084=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3085 = tloop2845delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3087[/*idx3084=*/ 0:0] = '{default:0};
always@(*) shiftreg3087[0] <= idx11;
wire [31:0] v3086 = shiftreg3087[/*idx3084=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 0][13] = tloop11delay[0];
assign v0_addr_input[/*idx3084=*/ 0][13] = {v3086[3:0]};
wire[31:0] v3088 = v0_rd_data[/*idx3084=*/ 0];
assign v0_rd_en_input[/*idx3084=*/ 0][13] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3089 = /*idx3084=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3091[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3091[0] <= v3088;
always@(posedge clk) shiftreg3091[/*idx13=*/ 13:1] <= shiftreg3091[/*idx13=*/ 12:0];
wire [31:0] v3090 = shiftreg3091[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3092 = v1_rd_data[/*idx3084=*/ 0][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 0][/*idx13=*/ 13][0] = tloop2845delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3093;
mult mult3094(v3093,
v3090,
v3092,
tloop2845delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3095 = v3083[/*idx3084=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3096;
add add3097(v3096,
v3093,
v3095,
tloop2845delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[1] = v3096;

//TerminatorOp

//} Unrolled body 0 of loop3084.
//DEBUG: /*idx3084=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop3084.
//DEBUG: /*idx3084=*/ 1'd1, expected 1
//printTimeOffset
reg tloop3085delay[3:0] = '{default:0} ;
always@(*) tloop3085delay[0] <= tloop3085;
generate
genvar i3099;

for(i3099 = 1; i3099<= 3; i3099= i3099 + 1) begin
always@(posedge clk) begin
tloop3085delay[i3099] <= tloop3085delay[i3099-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3098 = tloop3085delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3101[/*idx3084=*/ 1:0] = '{default:0};
always@(*) shiftreg3101[0] <= idx11;
always@(posedge clk) shiftreg3101[/*idx3084=*/ 1:1] <= shiftreg3101[/*idx3084=*/ 0:0];
wire [31:0] v3100 = shiftreg3101[/*idx3084=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 1][13] = tloop11delay[1];
assign v0_addr_input[/*idx3084=*/ 1][13] = {v3100[3:0]};
wire[31:0] v3102 = v0_rd_data[/*idx3084=*/ 1];
assign v0_rd_en_input[/*idx3084=*/ 1][13] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3103 = /*idx3084=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3105[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3105[0] <= v3102;
always@(posedge clk) shiftreg3105[/*idx13=*/ 13:1] <= shiftreg3105[/*idx13=*/ 12:0];
wire [31:0] v3104 = shiftreg3105[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3106 = v1_rd_data[/*idx3084=*/ 1][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 1][/*idx13=*/ 13][0] = tloop3085delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3107;
mult mult3108(v3107,
v3104,
v3106,
tloop3085delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3109 = v3083[/*idx3084=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3110;
add add3111(v3110,
v3107,
v3109,
tloop3085delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[2] = v3110;

//TerminatorOp

//} Unrolled body 1 of loop3084.
//DEBUG: /*idx3084=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop3084.
//DEBUG: /*idx3084=*/ 2'd2, expected 2
//printTimeOffset
reg tloop3098delay[3:0] = '{default:0} ;
always@(*) tloop3098delay[0] <= tloop3098;
generate
genvar i3113;

for(i3113 = 1; i3113<= 3; i3113= i3113 + 1) begin
always@(posedge clk) begin
tloop3098delay[i3113] <= tloop3098delay[i3113-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3112 = tloop3098delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3115[/*idx3084=*/ 2:0] = '{default:0};
always@(*) shiftreg3115[0] <= idx11;
always@(posedge clk) shiftreg3115[/*idx3084=*/ 2:1] <= shiftreg3115[/*idx3084=*/ 1:0];
wire [31:0] v3114 = shiftreg3115[/*idx3084=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 2][13] = tloop11delay[2];
assign v0_addr_input[/*idx3084=*/ 2][13] = {v3114[3:0]};
wire[31:0] v3116 = v0_rd_data[/*idx3084=*/ 2];
assign v0_rd_en_input[/*idx3084=*/ 2][13] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3117 = /*idx3084=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3119[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3119[0] <= v3116;
always@(posedge clk) shiftreg3119[/*idx13=*/ 13:1] <= shiftreg3119[/*idx13=*/ 12:0];
wire [31:0] v3118 = shiftreg3119[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3120 = v1_rd_data[/*idx3084=*/ 2][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 2][/*idx13=*/ 13][0] = tloop3098delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3121;
mult mult3122(v3121,
v3118,
v3120,
tloop3098delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3123 = v3083[/*idx3084=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3124;
add add3125(v3124,
v3121,
v3123,
tloop3098delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[3] = v3124;

//TerminatorOp

//} Unrolled body 2 of loop3084.
//DEBUG: /*idx3084=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop3084.
//DEBUG: /*idx3084=*/ 2'd3, expected 3
//printTimeOffset
reg tloop3112delay[3:0] = '{default:0} ;
always@(*) tloop3112delay[0] <= tloop3112;
generate
genvar i3127;

for(i3127 = 1; i3127<= 3; i3127= i3127 + 1) begin
always@(posedge clk) begin
tloop3112delay[i3127] <= tloop3112delay[i3127-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3126 = tloop3112delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3129[/*idx3084=*/ 3:0] = '{default:0};
always@(*) shiftreg3129[0] <= idx11;
always@(posedge clk) shiftreg3129[/*idx3084=*/ 3:1] <= shiftreg3129[/*idx3084=*/ 2:0];
wire [31:0] v3128 = shiftreg3129[/*idx3084=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 3][13] = tloop11delay[3];
assign v0_addr_input[/*idx3084=*/ 3][13] = {v3128[3:0]};
wire[31:0] v3130 = v0_rd_data[/*idx3084=*/ 3];
assign v0_rd_en_input[/*idx3084=*/ 3][13] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3131 = /*idx3084=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3133[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3133[0] <= v3130;
always@(posedge clk) shiftreg3133[/*idx13=*/ 13:1] <= shiftreg3133[/*idx13=*/ 12:0];
wire [31:0] v3132 = shiftreg3133[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3134 = v1_rd_data[/*idx3084=*/ 3][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 3][/*idx13=*/ 13][0] = tloop3112delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3135;
mult mult3136(v3135,
v3132,
v3134,
tloop3112delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3137 = v3083[/*idx3084=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3138;
add add3139(v3138,
v3135,
v3137,
tloop3112delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[4] = v3138;

//TerminatorOp

//} Unrolled body 3 of loop3084.
//DEBUG: /*idx3084=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop3084.
//DEBUG: /*idx3084=*/ 3'd4, expected 4
//printTimeOffset
reg tloop3126delay[3:0] = '{default:0} ;
always@(*) tloop3126delay[0] <= tloop3126;
generate
genvar i3141;

for(i3141 = 1; i3141<= 3; i3141= i3141 + 1) begin
always@(posedge clk) begin
tloop3126delay[i3141] <= tloop3126delay[i3141-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3140 = tloop3126delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3143[/*idx3084=*/ 4:0] = '{default:0};
always@(*) shiftreg3143[0] <= idx11;
always@(posedge clk) shiftreg3143[/*idx3084=*/ 4:1] <= shiftreg3143[/*idx3084=*/ 3:0];
wire [31:0] v3142 = shiftreg3143[/*idx3084=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 4][13] = tloop11delay[4];
assign v0_addr_input[/*idx3084=*/ 4][13] = {v3142[3:0]};
wire[31:0] v3144 = v0_rd_data[/*idx3084=*/ 4];
assign v0_rd_en_input[/*idx3084=*/ 4][13] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3145 = /*idx3084=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3147[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3147[0] <= v3144;
always@(posedge clk) shiftreg3147[/*idx13=*/ 13:1] <= shiftreg3147[/*idx13=*/ 12:0];
wire [31:0] v3146 = shiftreg3147[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3148 = v1_rd_data[/*idx3084=*/ 4][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 4][/*idx13=*/ 13][0] = tloop3126delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3149;
mult mult3150(v3149,
v3146,
v3148,
tloop3126delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3151 = v3083[/*idx3084=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3152;
add add3153(v3152,
v3149,
v3151,
tloop3126delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[5] = v3152;

//TerminatorOp

//} Unrolled body 4 of loop3084.
//DEBUG: /*idx3084=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop3084.
//DEBUG: /*idx3084=*/ 3'd5, expected 5
//printTimeOffset
reg tloop3140delay[3:0] = '{default:0} ;
always@(*) tloop3140delay[0] <= tloop3140;
generate
genvar i3155;

for(i3155 = 1; i3155<= 3; i3155= i3155 + 1) begin
always@(posedge clk) begin
tloop3140delay[i3155] <= tloop3140delay[i3155-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3154 = tloop3140delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3157[/*idx3084=*/ 5:0] = '{default:0};
always@(*) shiftreg3157[0] <= idx11;
always@(posedge clk) shiftreg3157[/*idx3084=*/ 5:1] <= shiftreg3157[/*idx3084=*/ 4:0];
wire [31:0] v3156 = shiftreg3157[/*idx3084=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 5][13] = tloop11delay[5];
assign v0_addr_input[/*idx3084=*/ 5][13] = {v3156[3:0]};
wire[31:0] v3158 = v0_rd_data[/*idx3084=*/ 5];
assign v0_rd_en_input[/*idx3084=*/ 5][13] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3159 = /*idx3084=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3161[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3161[0] <= v3158;
always@(posedge clk) shiftreg3161[/*idx13=*/ 13:1] <= shiftreg3161[/*idx13=*/ 12:0];
wire [31:0] v3160 = shiftreg3161[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3162 = v1_rd_data[/*idx3084=*/ 5][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 5][/*idx13=*/ 13][0] = tloop3140delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3163;
mult mult3164(v3163,
v3160,
v3162,
tloop3140delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3165 = v3083[/*idx3084=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3166;
add add3167(v3166,
v3163,
v3165,
tloop3140delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[6] = v3166;

//TerminatorOp

//} Unrolled body 5 of loop3084.
//DEBUG: /*idx3084=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop3084.
//DEBUG: /*idx3084=*/ 3'd6, expected 6
//printTimeOffset
reg tloop3154delay[3:0] = '{default:0} ;
always@(*) tloop3154delay[0] <= tloop3154;
generate
genvar i3169;

for(i3169 = 1; i3169<= 3; i3169= i3169 + 1) begin
always@(posedge clk) begin
tloop3154delay[i3169] <= tloop3154delay[i3169-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3168 = tloop3154delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3171[/*idx3084=*/ 6:0] = '{default:0};
always@(*) shiftreg3171[0] <= idx11;
always@(posedge clk) shiftreg3171[/*idx3084=*/ 6:1] <= shiftreg3171[/*idx3084=*/ 5:0];
wire [31:0] v3170 = shiftreg3171[/*idx3084=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 6][13] = tloop11delay[6];
assign v0_addr_input[/*idx3084=*/ 6][13] = {v3170[3:0]};
wire[31:0] v3172 = v0_rd_data[/*idx3084=*/ 6];
assign v0_rd_en_input[/*idx3084=*/ 6][13] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3173 = /*idx3084=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3175[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3175[0] <= v3172;
always@(posedge clk) shiftreg3175[/*idx13=*/ 13:1] <= shiftreg3175[/*idx13=*/ 12:0];
wire [31:0] v3174 = shiftreg3175[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3176 = v1_rd_data[/*idx3084=*/ 6][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 6][/*idx13=*/ 13][0] = tloop3154delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3177;
mult mult3178(v3177,
v3174,
v3176,
tloop3154delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3179 = v3083[/*idx3084=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3180;
add add3181(v3180,
v3177,
v3179,
tloop3154delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[7] = v3180;

//TerminatorOp

//} Unrolled body 6 of loop3084.
//DEBUG: /*idx3084=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop3084.
//DEBUG: /*idx3084=*/ 3'd7, expected 7
//printTimeOffset
reg tloop3168delay[3:0] = '{default:0} ;
always@(*) tloop3168delay[0] <= tloop3168;
generate
genvar i3183;

for(i3183 = 1; i3183<= 3; i3183= i3183 + 1) begin
always@(posedge clk) begin
tloop3168delay[i3183] <= tloop3168delay[i3183-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3182 = tloop3168delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3185[/*idx3084=*/ 7:0] = '{default:0};
always@(*) shiftreg3185[0] <= idx11;
always@(posedge clk) shiftreg3185[/*idx3084=*/ 7:1] <= shiftreg3185[/*idx3084=*/ 6:0];
wire [31:0] v3184 = shiftreg3185[/*idx3084=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 7][13] = tloop11delay[7];
assign v0_addr_input[/*idx3084=*/ 7][13] = {v3184[3:0]};
wire[31:0] v3186 = v0_rd_data[/*idx3084=*/ 7];
assign v0_rd_en_input[/*idx3084=*/ 7][13] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3187 = /*idx3084=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3189[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3189[0] <= v3186;
always@(posedge clk) shiftreg3189[/*idx13=*/ 13:1] <= shiftreg3189[/*idx13=*/ 12:0];
wire [31:0] v3188 = shiftreg3189[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3190 = v1_rd_data[/*idx3084=*/ 7][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 7][/*idx13=*/ 13][0] = tloop3168delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3191;
mult mult3192(v3191,
v3188,
v3190,
tloop3168delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3193 = v3083[/*idx3084=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3194;
add add3195(v3194,
v3191,
v3193,
tloop3168delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[8] = v3194;

//TerminatorOp

//} Unrolled body 7 of loop3084.
//DEBUG: /*idx3084=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop3084.
//DEBUG: /*idx3084=*/ 4'd8, expected 8
//printTimeOffset
reg tloop3182delay[3:0] = '{default:0} ;
always@(*) tloop3182delay[0] <= tloop3182;
generate
genvar i3197;

for(i3197 = 1; i3197<= 3; i3197= i3197 + 1) begin
always@(posedge clk) begin
tloop3182delay[i3197] <= tloop3182delay[i3197-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3196 = tloop3182delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3199[/*idx3084=*/ 8:0] = '{default:0};
always@(*) shiftreg3199[0] <= idx11;
always@(posedge clk) shiftreg3199[/*idx3084=*/ 8:1] <= shiftreg3199[/*idx3084=*/ 7:0];
wire [31:0] v3198 = shiftreg3199[/*idx3084=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 8][13] = tloop11delay[8];
assign v0_addr_input[/*idx3084=*/ 8][13] = {v3198[3:0]};
wire[31:0] v3200 = v0_rd_data[/*idx3084=*/ 8];
assign v0_rd_en_input[/*idx3084=*/ 8][13] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3201 = /*idx3084=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3203[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3203[0] <= v3200;
always@(posedge clk) shiftreg3203[/*idx13=*/ 13:1] <= shiftreg3203[/*idx13=*/ 12:0];
wire [31:0] v3202 = shiftreg3203[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3204 = v1_rd_data[/*idx3084=*/ 8][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 8][/*idx13=*/ 13][0] = tloop3182delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3205;
mult mult3206(v3205,
v3202,
v3204,
tloop3182delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3207 = v3083[/*idx3084=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3208;
add add3209(v3208,
v3205,
v3207,
tloop3182delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[9] = v3208;

//TerminatorOp

//} Unrolled body 8 of loop3084.
//DEBUG: /*idx3084=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop3084.
//DEBUG: /*idx3084=*/ 4'd9, expected 9
//printTimeOffset
reg tloop3196delay[3:0] = '{default:0} ;
always@(*) tloop3196delay[0] <= tloop3196;
generate
genvar i3211;

for(i3211 = 1; i3211<= 3; i3211= i3211 + 1) begin
always@(posedge clk) begin
tloop3196delay[i3211] <= tloop3196delay[i3211-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3210 = tloop3196delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3213[/*idx3084=*/ 9:0] = '{default:0};
always@(*) shiftreg3213[0] <= idx11;
always@(posedge clk) shiftreg3213[/*idx3084=*/ 9:1] <= shiftreg3213[/*idx3084=*/ 8:0];
wire [31:0] v3212 = shiftreg3213[/*idx3084=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 9][13] = tloop11delay[9];
assign v0_addr_input[/*idx3084=*/ 9][13] = {v3212[3:0]};
wire[31:0] v3214 = v0_rd_data[/*idx3084=*/ 9];
assign v0_rd_en_input[/*idx3084=*/ 9][13] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3215 = /*idx3084=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3217[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3217[0] <= v3214;
always@(posedge clk) shiftreg3217[/*idx13=*/ 13:1] <= shiftreg3217[/*idx13=*/ 12:0];
wire [31:0] v3216 = shiftreg3217[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3218 = v1_rd_data[/*idx3084=*/ 9][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 9][/*idx13=*/ 13][0] = tloop3196delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3219;
mult mult3220(v3219,
v3216,
v3218,
tloop3196delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3221 = v3083[/*idx3084=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3222;
add add3223(v3222,
v3219,
v3221,
tloop3196delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[10] = v3222;

//TerminatorOp

//} Unrolled body 9 of loop3084.
//DEBUG: /*idx3084=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop3084.
//DEBUG: /*idx3084=*/ 4'd10, expected 10
//printTimeOffset
reg tloop3210delay[3:0] = '{default:0} ;
always@(*) tloop3210delay[0] <= tloop3210;
generate
genvar i3225;

for(i3225 = 1; i3225<= 3; i3225= i3225 + 1) begin
always@(posedge clk) begin
tloop3210delay[i3225] <= tloop3210delay[i3225-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3224 = tloop3210delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3227[/*idx3084=*/ 10:0] = '{default:0};
always@(*) shiftreg3227[0] <= idx11;
always@(posedge clk) shiftreg3227[/*idx3084=*/ 10:1] <= shiftreg3227[/*idx3084=*/ 9:0];
wire [31:0] v3226 = shiftreg3227[/*idx3084=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 10][13] = tloop11delay[10];
assign v0_addr_input[/*idx3084=*/ 10][13] = {v3226[3:0]};
wire[31:0] v3228 = v0_rd_data[/*idx3084=*/ 10];
assign v0_rd_en_input[/*idx3084=*/ 10][13] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3229 = /*idx3084=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3231[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3231[0] <= v3228;
always@(posedge clk) shiftreg3231[/*idx13=*/ 13:1] <= shiftreg3231[/*idx13=*/ 12:0];
wire [31:0] v3230 = shiftreg3231[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3232 = v1_rd_data[/*idx3084=*/ 10][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 10][/*idx13=*/ 13][0] = tloop3210delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3233;
mult mult3234(v3233,
v3230,
v3232,
tloop3210delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3235 = v3083[/*idx3084=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3236;
add add3237(v3236,
v3233,
v3235,
tloop3210delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[11] = v3236;

//TerminatorOp

//} Unrolled body 10 of loop3084.
//DEBUG: /*idx3084=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop3084.
//DEBUG: /*idx3084=*/ 4'd11, expected 11
//printTimeOffset
reg tloop3224delay[3:0] = '{default:0} ;
always@(*) tloop3224delay[0] <= tloop3224;
generate
genvar i3239;

for(i3239 = 1; i3239<= 3; i3239= i3239 + 1) begin
always@(posedge clk) begin
tloop3224delay[i3239] <= tloop3224delay[i3239-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3238 = tloop3224delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3241[/*idx3084=*/ 11:0] = '{default:0};
always@(*) shiftreg3241[0] <= idx11;
always@(posedge clk) shiftreg3241[/*idx3084=*/ 11:1] <= shiftreg3241[/*idx3084=*/ 10:0];
wire [31:0] v3240 = shiftreg3241[/*idx3084=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 11][13] = tloop11delay[11];
assign v0_addr_input[/*idx3084=*/ 11][13] = {v3240[3:0]};
wire[31:0] v3242 = v0_rd_data[/*idx3084=*/ 11];
assign v0_rd_en_input[/*idx3084=*/ 11][13] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3243 = /*idx3084=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3245[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3245[0] <= v3242;
always@(posedge clk) shiftreg3245[/*idx13=*/ 13:1] <= shiftreg3245[/*idx13=*/ 12:0];
wire [31:0] v3244 = shiftreg3245[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3246 = v1_rd_data[/*idx3084=*/ 11][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 11][/*idx13=*/ 13][0] = tloop3224delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3247;
mult mult3248(v3247,
v3244,
v3246,
tloop3224delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3249 = v3083[/*idx3084=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3250;
add add3251(v3250,
v3247,
v3249,
tloop3224delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[12] = v3250;

//TerminatorOp

//} Unrolled body 11 of loop3084.
//DEBUG: /*idx3084=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop3084.
//DEBUG: /*idx3084=*/ 4'd12, expected 12
//printTimeOffset
reg tloop3238delay[3:0] = '{default:0} ;
always@(*) tloop3238delay[0] <= tloop3238;
generate
genvar i3253;

for(i3253 = 1; i3253<= 3; i3253= i3253 + 1) begin
always@(posedge clk) begin
tloop3238delay[i3253] <= tloop3238delay[i3253-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3252 = tloop3238delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3255[/*idx3084=*/ 12:0] = '{default:0};
always@(*) shiftreg3255[0] <= idx11;
always@(posedge clk) shiftreg3255[/*idx3084=*/ 12:1] <= shiftreg3255[/*idx3084=*/ 11:0];
wire [31:0] v3254 = shiftreg3255[/*idx3084=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 12][13] = tloop11delay[12];
assign v0_addr_input[/*idx3084=*/ 12][13] = {v3254[3:0]};
wire[31:0] v3256 = v0_rd_data[/*idx3084=*/ 12];
assign v0_rd_en_input[/*idx3084=*/ 12][13] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3257 = /*idx3084=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3259[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3259[0] <= v3256;
always@(posedge clk) shiftreg3259[/*idx13=*/ 13:1] <= shiftreg3259[/*idx13=*/ 12:0];
wire [31:0] v3258 = shiftreg3259[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3260 = v1_rd_data[/*idx3084=*/ 12][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 12][/*idx13=*/ 13][0] = tloop3238delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3261;
mult mult3262(v3261,
v3258,
v3260,
tloop3238delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3263 = v3083[/*idx3084=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3264;
add add3265(v3264,
v3261,
v3263,
tloop3238delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[13] = v3264;

//TerminatorOp

//} Unrolled body 12 of loop3084.
//DEBUG: /*idx3084=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop3084.
//DEBUG: /*idx3084=*/ 4'd13, expected 13
//printTimeOffset
reg tloop3252delay[3:0] = '{default:0} ;
always@(*) tloop3252delay[0] <= tloop3252;
generate
genvar i3267;

for(i3267 = 1; i3267<= 3; i3267= i3267 + 1) begin
always@(posedge clk) begin
tloop3252delay[i3267] <= tloop3252delay[i3267-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3266 = tloop3252delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3269[/*idx3084=*/ 13:0] = '{default:0};
always@(*) shiftreg3269[0] <= idx11;
always@(posedge clk) shiftreg3269[/*idx3084=*/ 13:1] <= shiftreg3269[/*idx3084=*/ 12:0];
wire [31:0] v3268 = shiftreg3269[/*idx3084=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 13][13] = tloop11delay[13];
assign v0_addr_input[/*idx3084=*/ 13][13] = {v3268[3:0]};
wire[31:0] v3270 = v0_rd_data[/*idx3084=*/ 13];
assign v0_rd_en_input[/*idx3084=*/ 13][13] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3271 = /*idx3084=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3273[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3273[0] <= v3270;
always@(posedge clk) shiftreg3273[/*idx13=*/ 13:1] <= shiftreg3273[/*idx13=*/ 12:0];
wire [31:0] v3272 = shiftreg3273[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3274 = v1_rd_data[/*idx3084=*/ 13][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 13][/*idx13=*/ 13][0] = tloop3252delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3275;
mult mult3276(v3275,
v3272,
v3274,
tloop3252delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3277 = v3083[/*idx3084=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3278;
add add3279(v3278,
v3275,
v3277,
tloop3252delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[14] = v3278;

//TerminatorOp

//} Unrolled body 13 of loop3084.
//DEBUG: /*idx3084=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop3084.
//DEBUG: /*idx3084=*/ 4'd14, expected 14
//printTimeOffset
reg tloop3266delay[3:0] = '{default:0} ;
always@(*) tloop3266delay[0] <= tloop3266;
generate
genvar i3281;

for(i3281 = 1; i3281<= 3; i3281= i3281 + 1) begin
always@(posedge clk) begin
tloop3266delay[i3281] <= tloop3266delay[i3281-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3280 = tloop3266delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3283[/*idx3084=*/ 14:0] = '{default:0};
always@(*) shiftreg3283[0] <= idx11;
always@(posedge clk) shiftreg3283[/*idx3084=*/ 14:1] <= shiftreg3283[/*idx3084=*/ 13:0];
wire [31:0] v3282 = shiftreg3283[/*idx3084=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 14][13] = tloop11delay[14];
assign v0_addr_input[/*idx3084=*/ 14][13] = {v3282[3:0]};
wire[31:0] v3284 = v0_rd_data[/*idx3084=*/ 14];
assign v0_rd_en_input[/*idx3084=*/ 14][13] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3285 = /*idx3084=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3287[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3287[0] <= v3284;
always@(posedge clk) shiftreg3287[/*idx13=*/ 13:1] <= shiftreg3287[/*idx13=*/ 12:0];
wire [31:0] v3286 = shiftreg3287[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3288 = v1_rd_data[/*idx3084=*/ 14][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 14][/*idx13=*/ 13][0] = tloop3266delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3289;
mult mult3290(v3289,
v3286,
v3288,
tloop3266delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3291 = v3083[/*idx3084=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3292;
add add3293(v3292,
v3289,
v3291,
tloop3266delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[15] = v3292;

//TerminatorOp

//} Unrolled body 14 of loop3084.
//DEBUG: /*idx3084=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop3084.
//DEBUG: /*idx3084=*/ 4'd15, expected 15
//printTimeOffset
reg tloop3280delay[3:0] = '{default:0} ;
always@(*) tloop3280delay[0] <= tloop3280;
generate
genvar i3295;

for(i3295 = 1; i3295<= 3; i3295= i3295 + 1) begin
always@(posedge clk) begin
tloop3280delay[i3295] <= tloop3280delay[i3295-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3294 = tloop3280delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3297[/*idx3084=*/ 15:0] = '{default:0};
always@(*) shiftreg3297[0] <= idx11;
always@(posedge clk) shiftreg3297[/*idx3084=*/ 15:1] <= shiftreg3297[/*idx3084=*/ 14:0];
wire [31:0] v3296 = shiftreg3297[/*idx3084=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3084=*/ 15][13] = tloop11delay[15];
assign v0_addr_input[/*idx3084=*/ 15][13] = {v3296[3:0]};
wire[31:0] v3298 = v0_rd_data[/*idx3084=*/ 15];
assign v0_rd_en_input[/*idx3084=*/ 15][13] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3299 = /*idx3084=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3301[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3301[0] <= v3298;
always@(posedge clk) shiftreg3301[/*idx13=*/ 13:1] <= shiftreg3301[/*idx13=*/ 12:0];
wire [31:0] v3300 = shiftreg3301[/*idx13=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3302 = v1_rd_data[/*idx3084=*/ 15][/*idx13=*/ 13];
assign v1_rd_en_input[/*idx3084=*/ 15][/*idx13=*/ 13][0] = tloop3280delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3303;
mult mult3304(v3303,
v3300,
v3302,
tloop3280delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3305 = v3083[/*idx3084=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3306;
add add3307(v3306,
v3303,
v3305,
tloop3280delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3083[16] = v3306;

//TerminatorOp

//} Unrolled body 15 of loop3084.
//DEBUG: /*idx3084=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t3308;
assign t3308 = tloop3294;
//printTimeOffset
reg t3308delay[3:0] = '{default:0} ;
always@(*) t3308delay[0] <= t3308;
generate
genvar i3309;

for(i3309 = 1; i3309<= 3; i3309= i3309 + 1) begin
always@(posedge clk) begin
t3308delay[i3309] <= t3308delay[i3309-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v3310 = v3083[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg3312[/*idx13=*/ 13:0] = '{default:0};
always@(*) shiftreg3312[0] <= idx11;
always@(posedge clk) shiftreg3312[/*idx13=*/ 13:1] <= shiftreg3312[/*idx13=*/ 12:0];
wire [31:0] v3311 = shiftreg3312[/*idx13=*/ 13];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg3314[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg3314[0] <= v3311;
always@(posedge clk) shiftreg3314[/*v10=*/ 16:1] <= shiftreg3314[/*v10=*/ 15:0];
wire [31:0] v3313 = shiftreg3314[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg3316[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg3316[0] <= v3313;
always@(posedge clk) shiftreg3316[/*v8=*/ 3:1] <= shiftreg3316[/*v8=*/ 2:0];
wire [31:0] v3315 = shiftreg3316[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 13][0] = t3308delay[3];
assign v2_addr_input[/*idx13=*/ 13][0] = {v3315[3:0]};
assign v2_wr_en_input[/*idx13=*/ 13][0] = t3308delay[3];
assign v2_wr_data_valid[/*idx13=*/ 13][0] = t3308delay[3];
assign v2_wr_data_input[/*idx13=*/ 13][0] = v3310;


//TerminatorOp

//} Unrolled body 13 of loop13.
//DEBUG: /*idx13=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop13.
//DEBUG: /*idx13=*/ 4'd14, expected 14
//printTimeOffset
reg tloop3081delay[3:0] = '{default:0} ;
always@(*) tloop3081delay[0] <= tloop3081;
generate
genvar i3318;

for(i3318 = 1; i3318<= 3; i3318= i3318 + 1) begin
always@(posedge clk) begin
tloop3081delay[i3318] <= tloop3081delay[i3318-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop3317 = tloop3081delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v3319[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v3319[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop3320.
//DEBUG: /*idx3320=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3321 = tloop3081delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3323[/*idx3320=*/ 0:0] = '{default:0};
always@(*) shiftreg3323[0] <= idx11;
wire [31:0] v3322 = shiftreg3323[/*idx3320=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 0][14] = tloop11delay[0];
assign v0_addr_input[/*idx3320=*/ 0][14] = {v3322[3:0]};
wire[31:0] v3324 = v0_rd_data[/*idx3320=*/ 0];
assign v0_rd_en_input[/*idx3320=*/ 0][14] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3325 = /*idx3320=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3327[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3327[0] <= v3324;
always@(posedge clk) shiftreg3327[/*idx13=*/ 14:1] <= shiftreg3327[/*idx13=*/ 13:0];
wire [31:0] v3326 = shiftreg3327[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3328 = v1_rd_data[/*idx3320=*/ 0][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 0][/*idx13=*/ 14][0] = tloop3081delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3329;
mult mult3330(v3329,
v3326,
v3328,
tloop3081delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3331 = v3319[/*idx3320=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3332;
add add3333(v3332,
v3329,
v3331,
tloop3081delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[1] = v3332;

//TerminatorOp

//} Unrolled body 0 of loop3320.
//DEBUG: /*idx3320=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop3320.
//DEBUG: /*idx3320=*/ 1'd1, expected 1
//printTimeOffset
reg tloop3321delay[3:0] = '{default:0} ;
always@(*) tloop3321delay[0] <= tloop3321;
generate
genvar i3335;

for(i3335 = 1; i3335<= 3; i3335= i3335 + 1) begin
always@(posedge clk) begin
tloop3321delay[i3335] <= tloop3321delay[i3335-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3334 = tloop3321delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3337[/*idx3320=*/ 1:0] = '{default:0};
always@(*) shiftreg3337[0] <= idx11;
always@(posedge clk) shiftreg3337[/*idx3320=*/ 1:1] <= shiftreg3337[/*idx3320=*/ 0:0];
wire [31:0] v3336 = shiftreg3337[/*idx3320=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 1][14] = tloop11delay[1];
assign v0_addr_input[/*idx3320=*/ 1][14] = {v3336[3:0]};
wire[31:0] v3338 = v0_rd_data[/*idx3320=*/ 1];
assign v0_rd_en_input[/*idx3320=*/ 1][14] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3339 = /*idx3320=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3341[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3341[0] <= v3338;
always@(posedge clk) shiftreg3341[/*idx13=*/ 14:1] <= shiftreg3341[/*idx13=*/ 13:0];
wire [31:0] v3340 = shiftreg3341[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3342 = v1_rd_data[/*idx3320=*/ 1][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 1][/*idx13=*/ 14][0] = tloop3321delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3343;
mult mult3344(v3343,
v3340,
v3342,
tloop3321delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3345 = v3319[/*idx3320=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3346;
add add3347(v3346,
v3343,
v3345,
tloop3321delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[2] = v3346;

//TerminatorOp

//} Unrolled body 1 of loop3320.
//DEBUG: /*idx3320=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop3320.
//DEBUG: /*idx3320=*/ 2'd2, expected 2
//printTimeOffset
reg tloop3334delay[3:0] = '{default:0} ;
always@(*) tloop3334delay[0] <= tloop3334;
generate
genvar i3349;

for(i3349 = 1; i3349<= 3; i3349= i3349 + 1) begin
always@(posedge clk) begin
tloop3334delay[i3349] <= tloop3334delay[i3349-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3348 = tloop3334delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3351[/*idx3320=*/ 2:0] = '{default:0};
always@(*) shiftreg3351[0] <= idx11;
always@(posedge clk) shiftreg3351[/*idx3320=*/ 2:1] <= shiftreg3351[/*idx3320=*/ 1:0];
wire [31:0] v3350 = shiftreg3351[/*idx3320=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 2][14] = tloop11delay[2];
assign v0_addr_input[/*idx3320=*/ 2][14] = {v3350[3:0]};
wire[31:0] v3352 = v0_rd_data[/*idx3320=*/ 2];
assign v0_rd_en_input[/*idx3320=*/ 2][14] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3353 = /*idx3320=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3355[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3355[0] <= v3352;
always@(posedge clk) shiftreg3355[/*idx13=*/ 14:1] <= shiftreg3355[/*idx13=*/ 13:0];
wire [31:0] v3354 = shiftreg3355[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3356 = v1_rd_data[/*idx3320=*/ 2][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 2][/*idx13=*/ 14][0] = tloop3334delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3357;
mult mult3358(v3357,
v3354,
v3356,
tloop3334delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3359 = v3319[/*idx3320=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3360;
add add3361(v3360,
v3357,
v3359,
tloop3334delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[3] = v3360;

//TerminatorOp

//} Unrolled body 2 of loop3320.
//DEBUG: /*idx3320=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop3320.
//DEBUG: /*idx3320=*/ 2'd3, expected 3
//printTimeOffset
reg tloop3348delay[3:0] = '{default:0} ;
always@(*) tloop3348delay[0] <= tloop3348;
generate
genvar i3363;

for(i3363 = 1; i3363<= 3; i3363= i3363 + 1) begin
always@(posedge clk) begin
tloop3348delay[i3363] <= tloop3348delay[i3363-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3362 = tloop3348delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3365[/*idx3320=*/ 3:0] = '{default:0};
always@(*) shiftreg3365[0] <= idx11;
always@(posedge clk) shiftreg3365[/*idx3320=*/ 3:1] <= shiftreg3365[/*idx3320=*/ 2:0];
wire [31:0] v3364 = shiftreg3365[/*idx3320=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 3][14] = tloop11delay[3];
assign v0_addr_input[/*idx3320=*/ 3][14] = {v3364[3:0]};
wire[31:0] v3366 = v0_rd_data[/*idx3320=*/ 3];
assign v0_rd_en_input[/*idx3320=*/ 3][14] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3367 = /*idx3320=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3369[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3369[0] <= v3366;
always@(posedge clk) shiftreg3369[/*idx13=*/ 14:1] <= shiftreg3369[/*idx13=*/ 13:0];
wire [31:0] v3368 = shiftreg3369[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3370 = v1_rd_data[/*idx3320=*/ 3][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 3][/*idx13=*/ 14][0] = tloop3348delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3371;
mult mult3372(v3371,
v3368,
v3370,
tloop3348delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3373 = v3319[/*idx3320=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3374;
add add3375(v3374,
v3371,
v3373,
tloop3348delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[4] = v3374;

//TerminatorOp

//} Unrolled body 3 of loop3320.
//DEBUG: /*idx3320=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop3320.
//DEBUG: /*idx3320=*/ 3'd4, expected 4
//printTimeOffset
reg tloop3362delay[3:0] = '{default:0} ;
always@(*) tloop3362delay[0] <= tloop3362;
generate
genvar i3377;

for(i3377 = 1; i3377<= 3; i3377= i3377 + 1) begin
always@(posedge clk) begin
tloop3362delay[i3377] <= tloop3362delay[i3377-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3376 = tloop3362delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3379[/*idx3320=*/ 4:0] = '{default:0};
always@(*) shiftreg3379[0] <= idx11;
always@(posedge clk) shiftreg3379[/*idx3320=*/ 4:1] <= shiftreg3379[/*idx3320=*/ 3:0];
wire [31:0] v3378 = shiftreg3379[/*idx3320=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 4][14] = tloop11delay[4];
assign v0_addr_input[/*idx3320=*/ 4][14] = {v3378[3:0]};
wire[31:0] v3380 = v0_rd_data[/*idx3320=*/ 4];
assign v0_rd_en_input[/*idx3320=*/ 4][14] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3381 = /*idx3320=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3383[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3383[0] <= v3380;
always@(posedge clk) shiftreg3383[/*idx13=*/ 14:1] <= shiftreg3383[/*idx13=*/ 13:0];
wire [31:0] v3382 = shiftreg3383[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3384 = v1_rd_data[/*idx3320=*/ 4][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 4][/*idx13=*/ 14][0] = tloop3362delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3385;
mult mult3386(v3385,
v3382,
v3384,
tloop3362delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3387 = v3319[/*idx3320=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3388;
add add3389(v3388,
v3385,
v3387,
tloop3362delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[5] = v3388;

//TerminatorOp

//} Unrolled body 4 of loop3320.
//DEBUG: /*idx3320=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop3320.
//DEBUG: /*idx3320=*/ 3'd5, expected 5
//printTimeOffset
reg tloop3376delay[3:0] = '{default:0} ;
always@(*) tloop3376delay[0] <= tloop3376;
generate
genvar i3391;

for(i3391 = 1; i3391<= 3; i3391= i3391 + 1) begin
always@(posedge clk) begin
tloop3376delay[i3391] <= tloop3376delay[i3391-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3390 = tloop3376delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3393[/*idx3320=*/ 5:0] = '{default:0};
always@(*) shiftreg3393[0] <= idx11;
always@(posedge clk) shiftreg3393[/*idx3320=*/ 5:1] <= shiftreg3393[/*idx3320=*/ 4:0];
wire [31:0] v3392 = shiftreg3393[/*idx3320=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 5][14] = tloop11delay[5];
assign v0_addr_input[/*idx3320=*/ 5][14] = {v3392[3:0]};
wire[31:0] v3394 = v0_rd_data[/*idx3320=*/ 5];
assign v0_rd_en_input[/*idx3320=*/ 5][14] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3395 = /*idx3320=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3397[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3397[0] <= v3394;
always@(posedge clk) shiftreg3397[/*idx13=*/ 14:1] <= shiftreg3397[/*idx13=*/ 13:0];
wire [31:0] v3396 = shiftreg3397[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3398 = v1_rd_data[/*idx3320=*/ 5][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 5][/*idx13=*/ 14][0] = tloop3376delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3399;
mult mult3400(v3399,
v3396,
v3398,
tloop3376delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3401 = v3319[/*idx3320=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3402;
add add3403(v3402,
v3399,
v3401,
tloop3376delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[6] = v3402;

//TerminatorOp

//} Unrolled body 5 of loop3320.
//DEBUG: /*idx3320=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop3320.
//DEBUG: /*idx3320=*/ 3'd6, expected 6
//printTimeOffset
reg tloop3390delay[3:0] = '{default:0} ;
always@(*) tloop3390delay[0] <= tloop3390;
generate
genvar i3405;

for(i3405 = 1; i3405<= 3; i3405= i3405 + 1) begin
always@(posedge clk) begin
tloop3390delay[i3405] <= tloop3390delay[i3405-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3404 = tloop3390delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3407[/*idx3320=*/ 6:0] = '{default:0};
always@(*) shiftreg3407[0] <= idx11;
always@(posedge clk) shiftreg3407[/*idx3320=*/ 6:1] <= shiftreg3407[/*idx3320=*/ 5:0];
wire [31:0] v3406 = shiftreg3407[/*idx3320=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 6][14] = tloop11delay[6];
assign v0_addr_input[/*idx3320=*/ 6][14] = {v3406[3:0]};
wire[31:0] v3408 = v0_rd_data[/*idx3320=*/ 6];
assign v0_rd_en_input[/*idx3320=*/ 6][14] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3409 = /*idx3320=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3411[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3411[0] <= v3408;
always@(posedge clk) shiftreg3411[/*idx13=*/ 14:1] <= shiftreg3411[/*idx13=*/ 13:0];
wire [31:0] v3410 = shiftreg3411[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3412 = v1_rd_data[/*idx3320=*/ 6][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 6][/*idx13=*/ 14][0] = tloop3390delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3413;
mult mult3414(v3413,
v3410,
v3412,
tloop3390delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3415 = v3319[/*idx3320=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3416;
add add3417(v3416,
v3413,
v3415,
tloop3390delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[7] = v3416;

//TerminatorOp

//} Unrolled body 6 of loop3320.
//DEBUG: /*idx3320=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop3320.
//DEBUG: /*idx3320=*/ 3'd7, expected 7
//printTimeOffset
reg tloop3404delay[3:0] = '{default:0} ;
always@(*) tloop3404delay[0] <= tloop3404;
generate
genvar i3419;

for(i3419 = 1; i3419<= 3; i3419= i3419 + 1) begin
always@(posedge clk) begin
tloop3404delay[i3419] <= tloop3404delay[i3419-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3418 = tloop3404delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3421[/*idx3320=*/ 7:0] = '{default:0};
always@(*) shiftreg3421[0] <= idx11;
always@(posedge clk) shiftreg3421[/*idx3320=*/ 7:1] <= shiftreg3421[/*idx3320=*/ 6:0];
wire [31:0] v3420 = shiftreg3421[/*idx3320=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 7][14] = tloop11delay[7];
assign v0_addr_input[/*idx3320=*/ 7][14] = {v3420[3:0]};
wire[31:0] v3422 = v0_rd_data[/*idx3320=*/ 7];
assign v0_rd_en_input[/*idx3320=*/ 7][14] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3423 = /*idx3320=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3425[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3425[0] <= v3422;
always@(posedge clk) shiftreg3425[/*idx13=*/ 14:1] <= shiftreg3425[/*idx13=*/ 13:0];
wire [31:0] v3424 = shiftreg3425[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3426 = v1_rd_data[/*idx3320=*/ 7][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 7][/*idx13=*/ 14][0] = tloop3404delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3427;
mult mult3428(v3427,
v3424,
v3426,
tloop3404delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3429 = v3319[/*idx3320=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3430;
add add3431(v3430,
v3427,
v3429,
tloop3404delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[8] = v3430;

//TerminatorOp

//} Unrolled body 7 of loop3320.
//DEBUG: /*idx3320=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop3320.
//DEBUG: /*idx3320=*/ 4'd8, expected 8
//printTimeOffset
reg tloop3418delay[3:0] = '{default:0} ;
always@(*) tloop3418delay[0] <= tloop3418;
generate
genvar i3433;

for(i3433 = 1; i3433<= 3; i3433= i3433 + 1) begin
always@(posedge clk) begin
tloop3418delay[i3433] <= tloop3418delay[i3433-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3432 = tloop3418delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3435[/*idx3320=*/ 8:0] = '{default:0};
always@(*) shiftreg3435[0] <= idx11;
always@(posedge clk) shiftreg3435[/*idx3320=*/ 8:1] <= shiftreg3435[/*idx3320=*/ 7:0];
wire [31:0] v3434 = shiftreg3435[/*idx3320=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 8][14] = tloop11delay[8];
assign v0_addr_input[/*idx3320=*/ 8][14] = {v3434[3:0]};
wire[31:0] v3436 = v0_rd_data[/*idx3320=*/ 8];
assign v0_rd_en_input[/*idx3320=*/ 8][14] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3437 = /*idx3320=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3439[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3439[0] <= v3436;
always@(posedge clk) shiftreg3439[/*idx13=*/ 14:1] <= shiftreg3439[/*idx13=*/ 13:0];
wire [31:0] v3438 = shiftreg3439[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3440 = v1_rd_data[/*idx3320=*/ 8][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 8][/*idx13=*/ 14][0] = tloop3418delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3441;
mult mult3442(v3441,
v3438,
v3440,
tloop3418delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3443 = v3319[/*idx3320=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3444;
add add3445(v3444,
v3441,
v3443,
tloop3418delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[9] = v3444;

//TerminatorOp

//} Unrolled body 8 of loop3320.
//DEBUG: /*idx3320=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop3320.
//DEBUG: /*idx3320=*/ 4'd9, expected 9
//printTimeOffset
reg tloop3432delay[3:0] = '{default:0} ;
always@(*) tloop3432delay[0] <= tloop3432;
generate
genvar i3447;

for(i3447 = 1; i3447<= 3; i3447= i3447 + 1) begin
always@(posedge clk) begin
tloop3432delay[i3447] <= tloop3432delay[i3447-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3446 = tloop3432delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3449[/*idx3320=*/ 9:0] = '{default:0};
always@(*) shiftreg3449[0] <= idx11;
always@(posedge clk) shiftreg3449[/*idx3320=*/ 9:1] <= shiftreg3449[/*idx3320=*/ 8:0];
wire [31:0] v3448 = shiftreg3449[/*idx3320=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 9][14] = tloop11delay[9];
assign v0_addr_input[/*idx3320=*/ 9][14] = {v3448[3:0]};
wire[31:0] v3450 = v0_rd_data[/*idx3320=*/ 9];
assign v0_rd_en_input[/*idx3320=*/ 9][14] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3451 = /*idx3320=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3453[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3453[0] <= v3450;
always@(posedge clk) shiftreg3453[/*idx13=*/ 14:1] <= shiftreg3453[/*idx13=*/ 13:0];
wire [31:0] v3452 = shiftreg3453[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3454 = v1_rd_data[/*idx3320=*/ 9][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 9][/*idx13=*/ 14][0] = tloop3432delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3455;
mult mult3456(v3455,
v3452,
v3454,
tloop3432delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3457 = v3319[/*idx3320=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3458;
add add3459(v3458,
v3455,
v3457,
tloop3432delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[10] = v3458;

//TerminatorOp

//} Unrolled body 9 of loop3320.
//DEBUG: /*idx3320=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop3320.
//DEBUG: /*idx3320=*/ 4'd10, expected 10
//printTimeOffset
reg tloop3446delay[3:0] = '{default:0} ;
always@(*) tloop3446delay[0] <= tloop3446;
generate
genvar i3461;

for(i3461 = 1; i3461<= 3; i3461= i3461 + 1) begin
always@(posedge clk) begin
tloop3446delay[i3461] <= tloop3446delay[i3461-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3460 = tloop3446delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3463[/*idx3320=*/ 10:0] = '{default:0};
always@(*) shiftreg3463[0] <= idx11;
always@(posedge clk) shiftreg3463[/*idx3320=*/ 10:1] <= shiftreg3463[/*idx3320=*/ 9:0];
wire [31:0] v3462 = shiftreg3463[/*idx3320=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 10][14] = tloop11delay[10];
assign v0_addr_input[/*idx3320=*/ 10][14] = {v3462[3:0]};
wire[31:0] v3464 = v0_rd_data[/*idx3320=*/ 10];
assign v0_rd_en_input[/*idx3320=*/ 10][14] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3465 = /*idx3320=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3467[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3467[0] <= v3464;
always@(posedge clk) shiftreg3467[/*idx13=*/ 14:1] <= shiftreg3467[/*idx13=*/ 13:0];
wire [31:0] v3466 = shiftreg3467[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3468 = v1_rd_data[/*idx3320=*/ 10][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 10][/*idx13=*/ 14][0] = tloop3446delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3469;
mult mult3470(v3469,
v3466,
v3468,
tloop3446delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3471 = v3319[/*idx3320=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3472;
add add3473(v3472,
v3469,
v3471,
tloop3446delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[11] = v3472;

//TerminatorOp

//} Unrolled body 10 of loop3320.
//DEBUG: /*idx3320=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop3320.
//DEBUG: /*idx3320=*/ 4'd11, expected 11
//printTimeOffset
reg tloop3460delay[3:0] = '{default:0} ;
always@(*) tloop3460delay[0] <= tloop3460;
generate
genvar i3475;

for(i3475 = 1; i3475<= 3; i3475= i3475 + 1) begin
always@(posedge clk) begin
tloop3460delay[i3475] <= tloop3460delay[i3475-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3474 = tloop3460delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3477[/*idx3320=*/ 11:0] = '{default:0};
always@(*) shiftreg3477[0] <= idx11;
always@(posedge clk) shiftreg3477[/*idx3320=*/ 11:1] <= shiftreg3477[/*idx3320=*/ 10:0];
wire [31:0] v3476 = shiftreg3477[/*idx3320=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 11][14] = tloop11delay[11];
assign v0_addr_input[/*idx3320=*/ 11][14] = {v3476[3:0]};
wire[31:0] v3478 = v0_rd_data[/*idx3320=*/ 11];
assign v0_rd_en_input[/*idx3320=*/ 11][14] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3479 = /*idx3320=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3481[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3481[0] <= v3478;
always@(posedge clk) shiftreg3481[/*idx13=*/ 14:1] <= shiftreg3481[/*idx13=*/ 13:0];
wire [31:0] v3480 = shiftreg3481[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3482 = v1_rd_data[/*idx3320=*/ 11][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 11][/*idx13=*/ 14][0] = tloop3460delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3483;
mult mult3484(v3483,
v3480,
v3482,
tloop3460delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3485 = v3319[/*idx3320=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3486;
add add3487(v3486,
v3483,
v3485,
tloop3460delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[12] = v3486;

//TerminatorOp

//} Unrolled body 11 of loop3320.
//DEBUG: /*idx3320=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop3320.
//DEBUG: /*idx3320=*/ 4'd12, expected 12
//printTimeOffset
reg tloop3474delay[3:0] = '{default:0} ;
always@(*) tloop3474delay[0] <= tloop3474;
generate
genvar i3489;

for(i3489 = 1; i3489<= 3; i3489= i3489 + 1) begin
always@(posedge clk) begin
tloop3474delay[i3489] <= tloop3474delay[i3489-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3488 = tloop3474delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3491[/*idx3320=*/ 12:0] = '{default:0};
always@(*) shiftreg3491[0] <= idx11;
always@(posedge clk) shiftreg3491[/*idx3320=*/ 12:1] <= shiftreg3491[/*idx3320=*/ 11:0];
wire [31:0] v3490 = shiftreg3491[/*idx3320=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 12][14] = tloop11delay[12];
assign v0_addr_input[/*idx3320=*/ 12][14] = {v3490[3:0]};
wire[31:0] v3492 = v0_rd_data[/*idx3320=*/ 12];
assign v0_rd_en_input[/*idx3320=*/ 12][14] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3493 = /*idx3320=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3495[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3495[0] <= v3492;
always@(posedge clk) shiftreg3495[/*idx13=*/ 14:1] <= shiftreg3495[/*idx13=*/ 13:0];
wire [31:0] v3494 = shiftreg3495[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3496 = v1_rd_data[/*idx3320=*/ 12][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 12][/*idx13=*/ 14][0] = tloop3474delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3497;
mult mult3498(v3497,
v3494,
v3496,
tloop3474delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3499 = v3319[/*idx3320=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3500;
add add3501(v3500,
v3497,
v3499,
tloop3474delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[13] = v3500;

//TerminatorOp

//} Unrolled body 12 of loop3320.
//DEBUG: /*idx3320=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop3320.
//DEBUG: /*idx3320=*/ 4'd13, expected 13
//printTimeOffset
reg tloop3488delay[3:0] = '{default:0} ;
always@(*) tloop3488delay[0] <= tloop3488;
generate
genvar i3503;

for(i3503 = 1; i3503<= 3; i3503= i3503 + 1) begin
always@(posedge clk) begin
tloop3488delay[i3503] <= tloop3488delay[i3503-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3502 = tloop3488delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3505[/*idx3320=*/ 13:0] = '{default:0};
always@(*) shiftreg3505[0] <= idx11;
always@(posedge clk) shiftreg3505[/*idx3320=*/ 13:1] <= shiftreg3505[/*idx3320=*/ 12:0];
wire [31:0] v3504 = shiftreg3505[/*idx3320=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 13][14] = tloop11delay[13];
assign v0_addr_input[/*idx3320=*/ 13][14] = {v3504[3:0]};
wire[31:0] v3506 = v0_rd_data[/*idx3320=*/ 13];
assign v0_rd_en_input[/*idx3320=*/ 13][14] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3507 = /*idx3320=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3509[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3509[0] <= v3506;
always@(posedge clk) shiftreg3509[/*idx13=*/ 14:1] <= shiftreg3509[/*idx13=*/ 13:0];
wire [31:0] v3508 = shiftreg3509[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3510 = v1_rd_data[/*idx3320=*/ 13][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 13][/*idx13=*/ 14][0] = tloop3488delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3511;
mult mult3512(v3511,
v3508,
v3510,
tloop3488delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3513 = v3319[/*idx3320=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3514;
add add3515(v3514,
v3511,
v3513,
tloop3488delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[14] = v3514;

//TerminatorOp

//} Unrolled body 13 of loop3320.
//DEBUG: /*idx3320=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop3320.
//DEBUG: /*idx3320=*/ 4'd14, expected 14
//printTimeOffset
reg tloop3502delay[3:0] = '{default:0} ;
always@(*) tloop3502delay[0] <= tloop3502;
generate
genvar i3517;

for(i3517 = 1; i3517<= 3; i3517= i3517 + 1) begin
always@(posedge clk) begin
tloop3502delay[i3517] <= tloop3502delay[i3517-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3516 = tloop3502delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3519[/*idx3320=*/ 14:0] = '{default:0};
always@(*) shiftreg3519[0] <= idx11;
always@(posedge clk) shiftreg3519[/*idx3320=*/ 14:1] <= shiftreg3519[/*idx3320=*/ 13:0];
wire [31:0] v3518 = shiftreg3519[/*idx3320=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 14][14] = tloop11delay[14];
assign v0_addr_input[/*idx3320=*/ 14][14] = {v3518[3:0]};
wire[31:0] v3520 = v0_rd_data[/*idx3320=*/ 14];
assign v0_rd_en_input[/*idx3320=*/ 14][14] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3521 = /*idx3320=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3523[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3523[0] <= v3520;
always@(posedge clk) shiftreg3523[/*idx13=*/ 14:1] <= shiftreg3523[/*idx13=*/ 13:0];
wire [31:0] v3522 = shiftreg3523[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3524 = v1_rd_data[/*idx3320=*/ 14][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 14][/*idx13=*/ 14][0] = tloop3502delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3525;
mult mult3526(v3525,
v3522,
v3524,
tloop3502delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3527 = v3319[/*idx3320=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3528;
add add3529(v3528,
v3525,
v3527,
tloop3502delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[15] = v3528;

//TerminatorOp

//} Unrolled body 14 of loop3320.
//DEBUG: /*idx3320=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop3320.
//DEBUG: /*idx3320=*/ 4'd15, expected 15
//printTimeOffset
reg tloop3516delay[3:0] = '{default:0} ;
always@(*) tloop3516delay[0] <= tloop3516;
generate
genvar i3531;

for(i3531 = 1; i3531<= 3; i3531= i3531 + 1) begin
always@(posedge clk) begin
tloop3516delay[i3531] <= tloop3516delay[i3531-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3530 = tloop3516delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3533[/*idx3320=*/ 15:0] = '{default:0};
always@(*) shiftreg3533[0] <= idx11;
always@(posedge clk) shiftreg3533[/*idx3320=*/ 15:1] <= shiftreg3533[/*idx3320=*/ 14:0];
wire [31:0] v3532 = shiftreg3533[/*idx3320=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3320=*/ 15][14] = tloop11delay[15];
assign v0_addr_input[/*idx3320=*/ 15][14] = {v3532[3:0]};
wire[31:0] v3534 = v0_rd_data[/*idx3320=*/ 15];
assign v0_rd_en_input[/*idx3320=*/ 15][14] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3535 = /*idx3320=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3537[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3537[0] <= v3534;
always@(posedge clk) shiftreg3537[/*idx13=*/ 14:1] <= shiftreg3537[/*idx13=*/ 13:0];
wire [31:0] v3536 = shiftreg3537[/*idx13=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3538 = v1_rd_data[/*idx3320=*/ 15][/*idx13=*/ 14];
assign v1_rd_en_input[/*idx3320=*/ 15][/*idx13=*/ 14][0] = tloop3516delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3539;
mult mult3540(v3539,
v3536,
v3538,
tloop3516delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3541 = v3319[/*idx3320=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3542;
add add3543(v3542,
v3539,
v3541,
tloop3516delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3319[16] = v3542;

//TerminatorOp

//} Unrolled body 15 of loop3320.
//DEBUG: /*idx3320=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t3544;
assign t3544 = tloop3530;
//printTimeOffset
reg t3544delay[3:0] = '{default:0} ;
always@(*) t3544delay[0] <= t3544;
generate
genvar i3545;

for(i3545 = 1; i3545<= 3; i3545= i3545 + 1) begin
always@(posedge clk) begin
t3544delay[i3545] <= t3544delay[i3545-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v3546 = v3319[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg3548[/*idx13=*/ 14:0] = '{default:0};
always@(*) shiftreg3548[0] <= idx11;
always@(posedge clk) shiftreg3548[/*idx13=*/ 14:1] <= shiftreg3548[/*idx13=*/ 13:0];
wire [31:0] v3547 = shiftreg3548[/*idx13=*/ 14];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg3550[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg3550[0] <= v3547;
always@(posedge clk) shiftreg3550[/*v10=*/ 16:1] <= shiftreg3550[/*v10=*/ 15:0];
wire [31:0] v3549 = shiftreg3550[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg3552[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg3552[0] <= v3549;
always@(posedge clk) shiftreg3552[/*v8=*/ 3:1] <= shiftreg3552[/*v8=*/ 2:0];
wire [31:0] v3551 = shiftreg3552[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 14][0] = t3544delay[3];
assign v2_addr_input[/*idx13=*/ 14][0] = {v3551[3:0]};
assign v2_wr_en_input[/*idx13=*/ 14][0] = t3544delay[3];
assign v2_wr_data_valid[/*idx13=*/ 14][0] = t3544delay[3];
assign v2_wr_data_input[/*idx13=*/ 14][0] = v3546;


//TerminatorOp

//} Unrolled body 14 of loop13.
//DEBUG: /*idx13=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop13.
//DEBUG: /*idx13=*/ 4'd15, expected 15
//printTimeOffset
reg tloop3317delay[3:0] = '{default:0} ;
always@(*) tloop3317delay[0] <= tloop3317;
generate
genvar i3554;

for(i3554 = 1; i3554<= 3; i3554= i3554 + 1) begin
always@(posedge clk) begin
tloop3317delay[i3554] <= tloop3317delay[i3554-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":76:7)
wire tloop3553 = tloop3317delay[1];

//AllocOp at loc("test/HIR/matmul.mlir":77:16)
wire [31:0] v3555[16:0];

//WireWriteOp at loc("test/HIR/matmul.mlir":78:7)
assign v3555[0] = /*v5=*/ 1'd0;

//UnrollForOp at loc("test/HIR/matmul.mlir":80:15)

//{ Unrolled body 0 of loop3556.
//DEBUG: /*idx3556=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3557 = tloop3317delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3559[/*idx3556=*/ 0:0] = '{default:0};
always@(*) shiftreg3559[0] <= idx11;
wire [31:0] v3558 = shiftreg3559[/*idx3556=*/ 0];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 0][15] = tloop11delay[0];
assign v0_addr_input[/*idx3556=*/ 0][15] = {v3558[3:0]};
wire[31:0] v3560 = v0_rd_data[/*idx3556=*/ 0];
assign v0_rd_en_input[/*idx3556=*/ 0][15] = tloop11delay[0];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3561 = /*idx3556=*/ 0 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3563[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3563[0] <= v3560;
always@(posedge clk) shiftreg3563[/*idx13=*/ 15:1] <= shiftreg3563[/*idx13=*/ 14:0];
wire [31:0] v3562 = shiftreg3563[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3564 = v1_rd_data[/*idx3556=*/ 0][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 0][/*idx13=*/ 15][0] = tloop3317delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3565;
mult mult3566(v3565,
v3562,
v3564,
tloop3317delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3567 = v3555[/*idx3556=*/ 0];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3568;
add add3569(v3568,
v3565,
v3567,
tloop3317delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[1] = v3568;

//TerminatorOp

//} Unrolled body 0 of loop3556.
//DEBUG: /*idx3556=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop3556.
//DEBUG: /*idx3556=*/ 1'd1, expected 1
//printTimeOffset
reg tloop3557delay[3:0] = '{default:0} ;
always@(*) tloop3557delay[0] <= tloop3557;
generate
genvar i3571;

for(i3571 = 1; i3571<= 3; i3571= i3571 + 1) begin
always@(posedge clk) begin
tloop3557delay[i3571] <= tloop3557delay[i3571-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3570 = tloop3557delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3573[/*idx3556=*/ 1:0] = '{default:0};
always@(*) shiftreg3573[0] <= idx11;
always@(posedge clk) shiftreg3573[/*idx3556=*/ 1:1] <= shiftreg3573[/*idx3556=*/ 0:0];
wire [31:0] v3572 = shiftreg3573[/*idx3556=*/ 1];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 1][15] = tloop11delay[1];
assign v0_addr_input[/*idx3556=*/ 1][15] = {v3572[3:0]};
wire[31:0] v3574 = v0_rd_data[/*idx3556=*/ 1];
assign v0_rd_en_input[/*idx3556=*/ 1][15] = tloop11delay[1];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3575 = /*idx3556=*/ 1 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3577[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3577[0] <= v3574;
always@(posedge clk) shiftreg3577[/*idx13=*/ 15:1] <= shiftreg3577[/*idx13=*/ 14:0];
wire [31:0] v3576 = shiftreg3577[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3578 = v1_rd_data[/*idx3556=*/ 1][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 1][/*idx13=*/ 15][0] = tloop3557delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3579;
mult mult3580(v3579,
v3576,
v3578,
tloop3557delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3581 = v3555[/*idx3556=*/ 1];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3582;
add add3583(v3582,
v3579,
v3581,
tloop3557delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[2] = v3582;

//TerminatorOp

//} Unrolled body 1 of loop3556.
//DEBUG: /*idx3556=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop3556.
//DEBUG: /*idx3556=*/ 2'd2, expected 2
//printTimeOffset
reg tloop3570delay[3:0] = '{default:0} ;
always@(*) tloop3570delay[0] <= tloop3570;
generate
genvar i3585;

for(i3585 = 1; i3585<= 3; i3585= i3585 + 1) begin
always@(posedge clk) begin
tloop3570delay[i3585] <= tloop3570delay[i3585-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3584 = tloop3570delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3587[/*idx3556=*/ 2:0] = '{default:0};
always@(*) shiftreg3587[0] <= idx11;
always@(posedge clk) shiftreg3587[/*idx3556=*/ 2:1] <= shiftreg3587[/*idx3556=*/ 1:0];
wire [31:0] v3586 = shiftreg3587[/*idx3556=*/ 2];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 2][15] = tloop11delay[2];
assign v0_addr_input[/*idx3556=*/ 2][15] = {v3586[3:0]};
wire[31:0] v3588 = v0_rd_data[/*idx3556=*/ 2];
assign v0_rd_en_input[/*idx3556=*/ 2][15] = tloop11delay[2];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3589 = /*idx3556=*/ 2 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3591[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3591[0] <= v3588;
always@(posedge clk) shiftreg3591[/*idx13=*/ 15:1] <= shiftreg3591[/*idx13=*/ 14:0];
wire [31:0] v3590 = shiftreg3591[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3592 = v1_rd_data[/*idx3556=*/ 2][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 2][/*idx13=*/ 15][0] = tloop3570delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3593;
mult mult3594(v3593,
v3590,
v3592,
tloop3570delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3595 = v3555[/*idx3556=*/ 2];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3596;
add add3597(v3596,
v3593,
v3595,
tloop3570delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[3] = v3596;

//TerminatorOp

//} Unrolled body 2 of loop3556.
//DEBUG: /*idx3556=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop3556.
//DEBUG: /*idx3556=*/ 2'd3, expected 3
//printTimeOffset
reg tloop3584delay[3:0] = '{default:0} ;
always@(*) tloop3584delay[0] <= tloop3584;
generate
genvar i3599;

for(i3599 = 1; i3599<= 3; i3599= i3599 + 1) begin
always@(posedge clk) begin
tloop3584delay[i3599] <= tloop3584delay[i3599-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3598 = tloop3584delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3601[/*idx3556=*/ 3:0] = '{default:0};
always@(*) shiftreg3601[0] <= idx11;
always@(posedge clk) shiftreg3601[/*idx3556=*/ 3:1] <= shiftreg3601[/*idx3556=*/ 2:0];
wire [31:0] v3600 = shiftreg3601[/*idx3556=*/ 3];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 3][15] = tloop11delay[3];
assign v0_addr_input[/*idx3556=*/ 3][15] = {v3600[3:0]};
wire[31:0] v3602 = v0_rd_data[/*idx3556=*/ 3];
assign v0_rd_en_input[/*idx3556=*/ 3][15] = tloop11delay[3];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3603 = /*idx3556=*/ 3 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3605[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3605[0] <= v3602;
always@(posedge clk) shiftreg3605[/*idx13=*/ 15:1] <= shiftreg3605[/*idx13=*/ 14:0];
wire [31:0] v3604 = shiftreg3605[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3606 = v1_rd_data[/*idx3556=*/ 3][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 3][/*idx13=*/ 15][0] = tloop3584delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3607;
mult mult3608(v3607,
v3604,
v3606,
tloop3584delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3609 = v3555[/*idx3556=*/ 3];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3610;
add add3611(v3610,
v3607,
v3609,
tloop3584delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[4] = v3610;

//TerminatorOp

//} Unrolled body 3 of loop3556.
//DEBUG: /*idx3556=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop3556.
//DEBUG: /*idx3556=*/ 3'd4, expected 4
//printTimeOffset
reg tloop3598delay[3:0] = '{default:0} ;
always@(*) tloop3598delay[0] <= tloop3598;
generate
genvar i3613;

for(i3613 = 1; i3613<= 3; i3613= i3613 + 1) begin
always@(posedge clk) begin
tloop3598delay[i3613] <= tloop3598delay[i3613-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3612 = tloop3598delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3615[/*idx3556=*/ 4:0] = '{default:0};
always@(*) shiftreg3615[0] <= idx11;
always@(posedge clk) shiftreg3615[/*idx3556=*/ 4:1] <= shiftreg3615[/*idx3556=*/ 3:0];
wire [31:0] v3614 = shiftreg3615[/*idx3556=*/ 4];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 4][15] = tloop11delay[4];
assign v0_addr_input[/*idx3556=*/ 4][15] = {v3614[3:0]};
wire[31:0] v3616 = v0_rd_data[/*idx3556=*/ 4];
assign v0_rd_en_input[/*idx3556=*/ 4][15] = tloop11delay[4];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3617 = /*idx3556=*/ 4 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3619[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3619[0] <= v3616;
always@(posedge clk) shiftreg3619[/*idx13=*/ 15:1] <= shiftreg3619[/*idx13=*/ 14:0];
wire [31:0] v3618 = shiftreg3619[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3620 = v1_rd_data[/*idx3556=*/ 4][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 4][/*idx13=*/ 15][0] = tloop3598delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3621;
mult mult3622(v3621,
v3618,
v3620,
tloop3598delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3623 = v3555[/*idx3556=*/ 4];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3624;
add add3625(v3624,
v3621,
v3623,
tloop3598delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[5] = v3624;

//TerminatorOp

//} Unrolled body 4 of loop3556.
//DEBUG: /*idx3556=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop3556.
//DEBUG: /*idx3556=*/ 3'd5, expected 5
//printTimeOffset
reg tloop3612delay[3:0] = '{default:0} ;
always@(*) tloop3612delay[0] <= tloop3612;
generate
genvar i3627;

for(i3627 = 1; i3627<= 3; i3627= i3627 + 1) begin
always@(posedge clk) begin
tloop3612delay[i3627] <= tloop3612delay[i3627-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3626 = tloop3612delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3629[/*idx3556=*/ 5:0] = '{default:0};
always@(*) shiftreg3629[0] <= idx11;
always@(posedge clk) shiftreg3629[/*idx3556=*/ 5:1] <= shiftreg3629[/*idx3556=*/ 4:0];
wire [31:0] v3628 = shiftreg3629[/*idx3556=*/ 5];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 5][15] = tloop11delay[5];
assign v0_addr_input[/*idx3556=*/ 5][15] = {v3628[3:0]};
wire[31:0] v3630 = v0_rd_data[/*idx3556=*/ 5];
assign v0_rd_en_input[/*idx3556=*/ 5][15] = tloop11delay[5];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3631 = /*idx3556=*/ 5 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3633[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3633[0] <= v3630;
always@(posedge clk) shiftreg3633[/*idx13=*/ 15:1] <= shiftreg3633[/*idx13=*/ 14:0];
wire [31:0] v3632 = shiftreg3633[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3634 = v1_rd_data[/*idx3556=*/ 5][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 5][/*idx13=*/ 15][0] = tloop3612delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3635;
mult mult3636(v3635,
v3632,
v3634,
tloop3612delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3637 = v3555[/*idx3556=*/ 5];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3638;
add add3639(v3638,
v3635,
v3637,
tloop3612delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[6] = v3638;

//TerminatorOp

//} Unrolled body 5 of loop3556.
//DEBUG: /*idx3556=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop3556.
//DEBUG: /*idx3556=*/ 3'd6, expected 6
//printTimeOffset
reg tloop3626delay[3:0] = '{default:0} ;
always@(*) tloop3626delay[0] <= tloop3626;
generate
genvar i3641;

for(i3641 = 1; i3641<= 3; i3641= i3641 + 1) begin
always@(posedge clk) begin
tloop3626delay[i3641] <= tloop3626delay[i3641-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3640 = tloop3626delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3643[/*idx3556=*/ 6:0] = '{default:0};
always@(*) shiftreg3643[0] <= idx11;
always@(posedge clk) shiftreg3643[/*idx3556=*/ 6:1] <= shiftreg3643[/*idx3556=*/ 5:0];
wire [31:0] v3642 = shiftreg3643[/*idx3556=*/ 6];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 6][15] = tloop11delay[6];
assign v0_addr_input[/*idx3556=*/ 6][15] = {v3642[3:0]};
wire[31:0] v3644 = v0_rd_data[/*idx3556=*/ 6];
assign v0_rd_en_input[/*idx3556=*/ 6][15] = tloop11delay[6];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3645 = /*idx3556=*/ 6 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3647[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3647[0] <= v3644;
always@(posedge clk) shiftreg3647[/*idx13=*/ 15:1] <= shiftreg3647[/*idx13=*/ 14:0];
wire [31:0] v3646 = shiftreg3647[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3648 = v1_rd_data[/*idx3556=*/ 6][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 6][/*idx13=*/ 15][0] = tloop3626delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3649;
mult mult3650(v3649,
v3646,
v3648,
tloop3626delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3651 = v3555[/*idx3556=*/ 6];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3652;
add add3653(v3652,
v3649,
v3651,
tloop3626delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[7] = v3652;

//TerminatorOp

//} Unrolled body 6 of loop3556.
//DEBUG: /*idx3556=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop3556.
//DEBUG: /*idx3556=*/ 3'd7, expected 7
//printTimeOffset
reg tloop3640delay[3:0] = '{default:0} ;
always@(*) tloop3640delay[0] <= tloop3640;
generate
genvar i3655;

for(i3655 = 1; i3655<= 3; i3655= i3655 + 1) begin
always@(posedge clk) begin
tloop3640delay[i3655] <= tloop3640delay[i3655-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3654 = tloop3640delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3657[/*idx3556=*/ 7:0] = '{default:0};
always@(*) shiftreg3657[0] <= idx11;
always@(posedge clk) shiftreg3657[/*idx3556=*/ 7:1] <= shiftreg3657[/*idx3556=*/ 6:0];
wire [31:0] v3656 = shiftreg3657[/*idx3556=*/ 7];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 7][15] = tloop11delay[7];
assign v0_addr_input[/*idx3556=*/ 7][15] = {v3656[3:0]};
wire[31:0] v3658 = v0_rd_data[/*idx3556=*/ 7];
assign v0_rd_en_input[/*idx3556=*/ 7][15] = tloop11delay[7];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3659 = /*idx3556=*/ 7 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3661[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3661[0] <= v3658;
always@(posedge clk) shiftreg3661[/*idx13=*/ 15:1] <= shiftreg3661[/*idx13=*/ 14:0];
wire [31:0] v3660 = shiftreg3661[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3662 = v1_rd_data[/*idx3556=*/ 7][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 7][/*idx13=*/ 15][0] = tloop3640delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3663;
mult mult3664(v3663,
v3660,
v3662,
tloop3640delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3665 = v3555[/*idx3556=*/ 7];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3666;
add add3667(v3666,
v3663,
v3665,
tloop3640delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[8] = v3666;

//TerminatorOp

//} Unrolled body 7 of loop3556.
//DEBUG: /*idx3556=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop3556.
//DEBUG: /*idx3556=*/ 4'd8, expected 8
//printTimeOffset
reg tloop3654delay[3:0] = '{default:0} ;
always@(*) tloop3654delay[0] <= tloop3654;
generate
genvar i3669;

for(i3669 = 1; i3669<= 3; i3669= i3669 + 1) begin
always@(posedge clk) begin
tloop3654delay[i3669] <= tloop3654delay[i3669-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3668 = tloop3654delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3671[/*idx3556=*/ 8:0] = '{default:0};
always@(*) shiftreg3671[0] <= idx11;
always@(posedge clk) shiftreg3671[/*idx3556=*/ 8:1] <= shiftreg3671[/*idx3556=*/ 7:0];
wire [31:0] v3670 = shiftreg3671[/*idx3556=*/ 8];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 8][15] = tloop11delay[8];
assign v0_addr_input[/*idx3556=*/ 8][15] = {v3670[3:0]};
wire[31:0] v3672 = v0_rd_data[/*idx3556=*/ 8];
assign v0_rd_en_input[/*idx3556=*/ 8][15] = tloop11delay[8];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3673 = /*idx3556=*/ 8 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3675[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3675[0] <= v3672;
always@(posedge clk) shiftreg3675[/*idx13=*/ 15:1] <= shiftreg3675[/*idx13=*/ 14:0];
wire [31:0] v3674 = shiftreg3675[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3676 = v1_rd_data[/*idx3556=*/ 8][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 8][/*idx13=*/ 15][0] = tloop3654delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3677;
mult mult3678(v3677,
v3674,
v3676,
tloop3654delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3679 = v3555[/*idx3556=*/ 8];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3680;
add add3681(v3680,
v3677,
v3679,
tloop3654delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[9] = v3680;

//TerminatorOp

//} Unrolled body 8 of loop3556.
//DEBUG: /*idx3556=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop3556.
//DEBUG: /*idx3556=*/ 4'd9, expected 9
//printTimeOffset
reg tloop3668delay[3:0] = '{default:0} ;
always@(*) tloop3668delay[0] <= tloop3668;
generate
genvar i3683;

for(i3683 = 1; i3683<= 3; i3683= i3683 + 1) begin
always@(posedge clk) begin
tloop3668delay[i3683] <= tloop3668delay[i3683-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3682 = tloop3668delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3685[/*idx3556=*/ 9:0] = '{default:0};
always@(*) shiftreg3685[0] <= idx11;
always@(posedge clk) shiftreg3685[/*idx3556=*/ 9:1] <= shiftreg3685[/*idx3556=*/ 8:0];
wire [31:0] v3684 = shiftreg3685[/*idx3556=*/ 9];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 9][15] = tloop11delay[9];
assign v0_addr_input[/*idx3556=*/ 9][15] = {v3684[3:0]};
wire[31:0] v3686 = v0_rd_data[/*idx3556=*/ 9];
assign v0_rd_en_input[/*idx3556=*/ 9][15] = tloop11delay[9];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3687 = /*idx3556=*/ 9 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3689[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3689[0] <= v3686;
always@(posedge clk) shiftreg3689[/*idx13=*/ 15:1] <= shiftreg3689[/*idx13=*/ 14:0];
wire [31:0] v3688 = shiftreg3689[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3690 = v1_rd_data[/*idx3556=*/ 9][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 9][/*idx13=*/ 15][0] = tloop3668delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3691;
mult mult3692(v3691,
v3688,
v3690,
tloop3668delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3693 = v3555[/*idx3556=*/ 9];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3694;
add add3695(v3694,
v3691,
v3693,
tloop3668delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[10] = v3694;

//TerminatorOp

//} Unrolled body 9 of loop3556.
//DEBUG: /*idx3556=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop3556.
//DEBUG: /*idx3556=*/ 4'd10, expected 10
//printTimeOffset
reg tloop3682delay[3:0] = '{default:0} ;
always@(*) tloop3682delay[0] <= tloop3682;
generate
genvar i3697;

for(i3697 = 1; i3697<= 3; i3697= i3697 + 1) begin
always@(posedge clk) begin
tloop3682delay[i3697] <= tloop3682delay[i3697-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3696 = tloop3682delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3699[/*idx3556=*/ 10:0] = '{default:0};
always@(*) shiftreg3699[0] <= idx11;
always@(posedge clk) shiftreg3699[/*idx3556=*/ 10:1] <= shiftreg3699[/*idx3556=*/ 9:0];
wire [31:0] v3698 = shiftreg3699[/*idx3556=*/ 10];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 10][15] = tloop11delay[10];
assign v0_addr_input[/*idx3556=*/ 10][15] = {v3698[3:0]};
wire[31:0] v3700 = v0_rd_data[/*idx3556=*/ 10];
assign v0_rd_en_input[/*idx3556=*/ 10][15] = tloop11delay[10];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3701 = /*idx3556=*/ 10 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3703[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3703[0] <= v3700;
always@(posedge clk) shiftreg3703[/*idx13=*/ 15:1] <= shiftreg3703[/*idx13=*/ 14:0];
wire [31:0] v3702 = shiftreg3703[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3704 = v1_rd_data[/*idx3556=*/ 10][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 10][/*idx13=*/ 15][0] = tloop3682delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3705;
mult mult3706(v3705,
v3702,
v3704,
tloop3682delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3707 = v3555[/*idx3556=*/ 10];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3708;
add add3709(v3708,
v3705,
v3707,
tloop3682delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[11] = v3708;

//TerminatorOp

//} Unrolled body 10 of loop3556.
//DEBUG: /*idx3556=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop3556.
//DEBUG: /*idx3556=*/ 4'd11, expected 11
//printTimeOffset
reg tloop3696delay[3:0] = '{default:0} ;
always@(*) tloop3696delay[0] <= tloop3696;
generate
genvar i3711;

for(i3711 = 1; i3711<= 3; i3711= i3711 + 1) begin
always@(posedge clk) begin
tloop3696delay[i3711] <= tloop3696delay[i3711-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3710 = tloop3696delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3713[/*idx3556=*/ 11:0] = '{default:0};
always@(*) shiftreg3713[0] <= idx11;
always@(posedge clk) shiftreg3713[/*idx3556=*/ 11:1] <= shiftreg3713[/*idx3556=*/ 10:0];
wire [31:0] v3712 = shiftreg3713[/*idx3556=*/ 11];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 11][15] = tloop11delay[11];
assign v0_addr_input[/*idx3556=*/ 11][15] = {v3712[3:0]};
wire[31:0] v3714 = v0_rd_data[/*idx3556=*/ 11];
assign v0_rd_en_input[/*idx3556=*/ 11][15] = tloop11delay[11];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3715 = /*idx3556=*/ 11 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3717[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3717[0] <= v3714;
always@(posedge clk) shiftreg3717[/*idx13=*/ 15:1] <= shiftreg3717[/*idx13=*/ 14:0];
wire [31:0] v3716 = shiftreg3717[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3718 = v1_rd_data[/*idx3556=*/ 11][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 11][/*idx13=*/ 15][0] = tloop3696delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3719;
mult mult3720(v3719,
v3716,
v3718,
tloop3696delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3721 = v3555[/*idx3556=*/ 11];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3722;
add add3723(v3722,
v3719,
v3721,
tloop3696delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[12] = v3722;

//TerminatorOp

//} Unrolled body 11 of loop3556.
//DEBUG: /*idx3556=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop3556.
//DEBUG: /*idx3556=*/ 4'd12, expected 12
//printTimeOffset
reg tloop3710delay[3:0] = '{default:0} ;
always@(*) tloop3710delay[0] <= tloop3710;
generate
genvar i3725;

for(i3725 = 1; i3725<= 3; i3725= i3725 + 1) begin
always@(posedge clk) begin
tloop3710delay[i3725] <= tloop3710delay[i3725-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3724 = tloop3710delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3727[/*idx3556=*/ 12:0] = '{default:0};
always@(*) shiftreg3727[0] <= idx11;
always@(posedge clk) shiftreg3727[/*idx3556=*/ 12:1] <= shiftreg3727[/*idx3556=*/ 11:0];
wire [31:0] v3726 = shiftreg3727[/*idx3556=*/ 12];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 12][15] = tloop11delay[12];
assign v0_addr_input[/*idx3556=*/ 12][15] = {v3726[3:0]};
wire[31:0] v3728 = v0_rd_data[/*idx3556=*/ 12];
assign v0_rd_en_input[/*idx3556=*/ 12][15] = tloop11delay[12];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3729 = /*idx3556=*/ 12 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3731[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3731[0] <= v3728;
always@(posedge clk) shiftreg3731[/*idx13=*/ 15:1] <= shiftreg3731[/*idx13=*/ 14:0];
wire [31:0] v3730 = shiftreg3731[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3732 = v1_rd_data[/*idx3556=*/ 12][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 12][/*idx13=*/ 15][0] = tloop3710delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3733;
mult mult3734(v3733,
v3730,
v3732,
tloop3710delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3735 = v3555[/*idx3556=*/ 12];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3736;
add add3737(v3736,
v3733,
v3735,
tloop3710delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[13] = v3736;

//TerminatorOp

//} Unrolled body 12 of loop3556.
//DEBUG: /*idx3556=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop3556.
//DEBUG: /*idx3556=*/ 4'd13, expected 13
//printTimeOffset
reg tloop3724delay[3:0] = '{default:0} ;
always@(*) tloop3724delay[0] <= tloop3724;
generate
genvar i3739;

for(i3739 = 1; i3739<= 3; i3739= i3739 + 1) begin
always@(posedge clk) begin
tloop3724delay[i3739] <= tloop3724delay[i3739-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3738 = tloop3724delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3741[/*idx3556=*/ 13:0] = '{default:0};
always@(*) shiftreg3741[0] <= idx11;
always@(posedge clk) shiftreg3741[/*idx3556=*/ 13:1] <= shiftreg3741[/*idx3556=*/ 12:0];
wire [31:0] v3740 = shiftreg3741[/*idx3556=*/ 13];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 13][15] = tloop11delay[13];
assign v0_addr_input[/*idx3556=*/ 13][15] = {v3740[3:0]};
wire[31:0] v3742 = v0_rd_data[/*idx3556=*/ 13];
assign v0_rd_en_input[/*idx3556=*/ 13][15] = tloop11delay[13];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3743 = /*idx3556=*/ 13 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3745[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3745[0] <= v3742;
always@(posedge clk) shiftreg3745[/*idx13=*/ 15:1] <= shiftreg3745[/*idx13=*/ 14:0];
wire [31:0] v3744 = shiftreg3745[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3746 = v1_rd_data[/*idx3556=*/ 13][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 13][/*idx13=*/ 15][0] = tloop3724delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3747;
mult mult3748(v3747,
v3744,
v3746,
tloop3724delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3749 = v3555[/*idx3556=*/ 13];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3750;
add add3751(v3750,
v3747,
v3749,
tloop3724delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[14] = v3750;

//TerminatorOp

//} Unrolled body 13 of loop3556.
//DEBUG: /*idx3556=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop3556.
//DEBUG: /*idx3556=*/ 4'd14, expected 14
//printTimeOffset
reg tloop3738delay[3:0] = '{default:0} ;
always@(*) tloop3738delay[0] <= tloop3738;
generate
genvar i3753;

for(i3753 = 1; i3753<= 3; i3753= i3753 + 1) begin
always@(posedge clk) begin
tloop3738delay[i3753] <= tloop3738delay[i3753-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3752 = tloop3738delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3755[/*idx3556=*/ 14:0] = '{default:0};
always@(*) shiftreg3755[0] <= idx11;
always@(posedge clk) shiftreg3755[/*idx3556=*/ 14:1] <= shiftreg3755[/*idx3556=*/ 13:0];
wire [31:0] v3754 = shiftreg3755[/*idx3556=*/ 14];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 14][15] = tloop11delay[14];
assign v0_addr_input[/*idx3556=*/ 14][15] = {v3754[3:0]};
wire[31:0] v3756 = v0_rd_data[/*idx3556=*/ 14];
assign v0_rd_en_input[/*idx3556=*/ 14][15] = tloop11delay[14];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3757 = /*idx3556=*/ 14 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3759[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3759[0] <= v3756;
always@(posedge clk) shiftreg3759[/*idx13=*/ 15:1] <= shiftreg3759[/*idx13=*/ 14:0];
wire [31:0] v3758 = shiftreg3759[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3760 = v1_rd_data[/*idx3556=*/ 14][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 14][/*idx13=*/ 15][0] = tloop3738delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3761;
mult mult3762(v3761,
v3758,
v3760,
tloop3738delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3763 = v3555[/*idx3556=*/ 14];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3764;
add add3765(v3764,
v3761,
v3763,
tloop3738delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[15] = v3764;

//TerminatorOp

//} Unrolled body 14 of loop3556.
//DEBUG: /*idx3556=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop3556.
//DEBUG: /*idx3556=*/ 4'd15, expected 15
//printTimeOffset
reg tloop3752delay[3:0] = '{default:0} ;
always@(*) tloop3752delay[0] <= tloop3752;
generate
genvar i3767;

for(i3767 = 1; i3767<= 3; i3767= i3767 + 1) begin
always@(posedge clk) begin
tloop3752delay[i3767] <= tloop3752delay[i3767-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":81:9)
wire tloop3766 = tloop3752delay[1];

//DelayOp at loc("test/HIR/matmul.mlir":82:22)
reg[31:0]shiftreg3769[/*idx3556=*/ 15:0] = '{default:0};
always@(*) shiftreg3769[0] <= idx11;
always@(posedge clk) shiftreg3769[/*idx3556=*/ 15:1] <= shiftreg3769[/*idx3556=*/ 14:0];
wire [31:0] v3768 = shiftreg3769[/*idx3556=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":83:14)
assign v0_addr_valid[/*idx3556=*/ 15][15] = tloop11delay[15];
assign v0_addr_input[/*idx3556=*/ 15][15] = {v3768[3:0]};
wire[31:0] v3770 = v0_rd_data[/*idx3556=*/ 15];
assign v0_rd_en_input[/*idx3556=*/ 15][15] = tloop11delay[15];


//AddOp at loc("test/HIR/matmul.mlir":84:19)
//wire v3771 = /*idx3556=*/ 15 + /*v6=*/ 1;

//DelayOp at loc("test/HIR/matmul.mlir":85:22)
reg[31:0]shiftreg3773[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3773[0] <= v3770;
always@(posedge clk) shiftreg3773[/*idx13=*/ 15:1] <= shiftreg3773[/*idx13=*/ 14:0];
wire [31:0] v3772 = shiftreg3773[/*idx13=*/ 15];

//MemReadOp at loc("test/HIR/matmul.mlir":86:14)
wire[31:0] v3774 = v1_rd_data[/*idx3556=*/ 15][/*idx13=*/ 15];
assign v1_rd_en_input[/*idx3556=*/ 15][/*idx13=*/ 15][0] = tloop3752delay[0];


//CallOp at loc("test/HIR/matmul.mlir":87:15)
wire [31:0] v3775;
mult mult3776(v3775,
v3772,
v3774,
tloop3752delay[1],
clk
);

//WireReadOp at loc("test/HIR/matmul.mlir":88:19)
wire [31:0] v3777 = v3555[/*idx3556=*/ 15];

//CallOp at loc("test/HIR/matmul.mlir":89:14)
wire [31:0] v3778;
add add3779(v3778,
v3775,
v3777,
tloop3752delay[3],
clk
);

//WireWriteOp at loc("test/HIR/matmul.mlir":90:9)
assign v3555[16] = v3778;

//TerminatorOp

//} Unrolled body 15 of loop3556.
//DEBUG: /*idx3556=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t3780;
assign t3780 = tloop3766;
//printTimeOffset
reg t3780delay[3:0] = '{default:0} ;
always@(*) t3780delay[0] <= t3780;
generate
genvar i3781;

for(i3781 = 1; i3781<= 3; i3781= i3781 + 1) begin
always@(posedge clk) begin
t3780delay[i3781] <= t3780delay[i3781-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//WireReadOp at loc("test/HIR/matmul.mlir":92:14)
wire [31:0] v3782 = v3555[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":93:13)
reg[31:0]shiftreg3784[/*idx13=*/ 15:0] = '{default:0};
always@(*) shiftreg3784[0] <= idx11;
always@(posedge clk) shiftreg3784[/*idx13=*/ 15:1] <= shiftreg3784[/*idx13=*/ 14:0];
wire [31:0] v3783 = shiftreg3784[/*idx13=*/ 15];

//DelayOp at loc("test/HIR/matmul.mlir":94:13)
reg[31:0]shiftreg3786[/*v10=*/ 16:0] = '{default:0};
always@(*) shiftreg3786[0] <= v3783;
always@(posedge clk) shiftreg3786[/*v10=*/ 16:1] <= shiftreg3786[/*v10=*/ 15:0];
wire [31:0] v3785 = shiftreg3786[/*v10=*/ 16];

//DelayOp at loc("test/HIR/matmul.mlir":95:13)
reg[31:0]shiftreg3788[/*v8=*/ 3:0] = '{default:0};
always@(*) shiftreg3788[0] <= v3785;
always@(posedge clk) shiftreg3788[/*v8=*/ 3:1] <= shiftreg3788[/*v8=*/ 2:0];
wire [31:0] v3787 = shiftreg3788[/*v8=*/ 3];

//MemWriteOp at loc("test/HIR/matmul.mlir":96:7)
assign v2_addr_valid[/*idx13=*/ 15][0] = t3780delay[3];
assign v2_addr_input[/*idx13=*/ 15][0] = {v3787[3:0]};
assign v2_wr_en_input[/*idx13=*/ 15][0] = t3780delay[3];
assign v2_wr_data_valid[/*idx13=*/ 15][0] = t3780delay[3];
assign v2_wr_data_input[/*idx13=*/ 15][0] = v3782;


//TerminatorOp

//} Unrolled body 15 of loop13.
//DEBUG: /*idx13=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t3789;
assign t3789 = tloop3553;
//printTimeOffset
reg t3789delay[0:0] = '{default:0} ;
always@(*) t3789delay[0] <= t3789;
generate
genvar i3790;

for(i3790 = 1; i3790<= 0; i3790= i3790 + 1) begin
always@(posedge clk) begin
t3789delay[i3790] <= t3789delay[i3790-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//TerminatorOp

//} Loop11
//printTimeOffset
reg tfinish11delay[0:0] = '{default:0} ;
always@(*) tfinish11delay[0] <= tfinish11;
generate
genvar i3791;

for(i3791 = 1; i3791<= 0; i3791= i3791 + 1) begin
always@(posedge clk) begin
tfinish11delay[i3791] <= tfinish11delay[i3791-1];
end
end
endgenerate


//ReturnOp at loc("test/HIR/matmul.mlir":99:3)
endmodule
module writeC(
//Outputs.

//Inputs.

//MemrefType : port = r.
output reg[3:0] v0_addr[15:0],
output wire v0_rd_en[15:0],
input wire[31:0] v0_rd_data[15:0],
//MemrefType : port = w.
output reg[7:0] v1_addr,
output wire v1_wr_en,
output reg[31:0] v1_wr_data,
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

wire v0_addr_valid [15:0] [0:0] ;
wire [3:0] v0_addr_input [15:0] [0:0];
 generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
always@(*) begin
if(v0_addr_valid[i0][0] )
v0_addr[i0] = v0_addr_input[i0][0];
else
 v0_addr[i0] = 'x;
end
end
endgenerate

wire [0:0] v0_rd_en_input [15:0];
generate
for(genvar i0 = 0; i0 < 16;i0=i0 + 1) begin
assign v0_rd_en [i0] =| v0_rd_en_input [i0];
end
endgenerate


wire v1_addr_valid  [15:0] ;
wire [7:0] v1_addr_input  [15:0];
 always@(*) begin
if(v1_addr_valid[0] )
v1_addr = v1_addr_input[0];
else if (v1_addr_valid[1])
v1_addr = v1_addr_input[1];
else if (v1_addr_valid[2])
v1_addr = v1_addr_input[2];
else if (v1_addr_valid[3])
v1_addr = v1_addr_input[3];
else if (v1_addr_valid[4])
v1_addr = v1_addr_input[4];
else if (v1_addr_valid[5])
v1_addr = v1_addr_input[5];
else if (v1_addr_valid[6])
v1_addr = v1_addr_input[6];
else if (v1_addr_valid[7])
v1_addr = v1_addr_input[7];
else if (v1_addr_valid[8])
v1_addr = v1_addr_input[8];
else if (v1_addr_valid[9])
v1_addr = v1_addr_input[9];
else if (v1_addr_valid[10])
v1_addr = v1_addr_input[10];
else if (v1_addr_valid[11])
v1_addr = v1_addr_input[11];
else if (v1_addr_valid[12])
v1_addr = v1_addr_input[12];
else if (v1_addr_valid[13])
v1_addr = v1_addr_input[13];
else if (v1_addr_valid[14])
v1_addr = v1_addr_input[14];
else if (v1_addr_valid[15])
v1_addr = v1_addr_input[15];
else
 v1_addr = 'x;
end

wire [15:0] v1_wr_en_input ;
assign v1_wr_en  =| v1_wr_en_input ;
wire v1_wr_data_valid  [15:0] ;
wire [31:0] v1_wr_data_input  [15:0];
 always@(*) begin
if(v1_wr_data_valid[0] )
v1_wr_data = v1_wr_data_input[0];
else if (v1_wr_data_valid[1])
v1_wr_data = v1_wr_data_input[1];
else if (v1_wr_data_valid[2])
v1_wr_data = v1_wr_data_input[2];
else if (v1_wr_data_valid[3])
v1_wr_data = v1_wr_data_input[3];
else if (v1_wr_data_valid[4])
v1_wr_data = v1_wr_data_input[4];
else if (v1_wr_data_valid[5])
v1_wr_data = v1_wr_data_input[5];
else if (v1_wr_data_valid[6])
v1_wr_data = v1_wr_data_input[6];
else if (v1_wr_data_valid[7])
v1_wr_data = v1_wr_data_input[7];
else if (v1_wr_data_valid[8])
v1_wr_data = v1_wr_data_input[8];
else if (v1_wr_data_valid[9])
v1_wr_data = v1_wr_data_input[9];
else if (v1_wr_data_valid[10])
v1_wr_data = v1_wr_data_input[10];
else if (v1_wr_data_valid[11])
v1_wr_data = v1_wr_data_input[11];
else if (v1_wr_data_valid[12])
v1_wr_data = v1_wr_data_input[12];
else if (v1_wr_data_valid[13])
v1_wr_data = v1_wr_data_input[13];
else if (v1_wr_data_valid[14])
v1_wr_data = v1_wr_data_input[14];
else if (v1_wr_data_valid[15])
v1_wr_data = v1_wr_data_input[15];
else
 v1_wr_data = 'x;
end


//printTimeOffset
reg tstartdelay[0:0] = '{default:0} ;
always@(*) tstartdelay[0] <= tstart;
generate
genvar i3;

for(i3 = 1; i3<= 0; i3= i3 + 1) begin
always@(posedge clk) begin
tstartdelay[i3] <= tstartdelay[i3-1];
end
end
endgenerate


//ConstantOp at loc("test/HIR/matmul.mlir":106:8)
//constant v4 = 1'd0;

//ConstantOp at loc("test/HIR/matmul.mlir":107:8)
//constant v5 = 1'd1;

//ConstantOp at loc("test/HIR/matmul.mlir":108:8)
//constant [1:0] v6 = 2'd2;

//ConstantOp at loc("test/HIR/matmul.mlir":109:8)
//constant [1:0] v7 = 2'd3;

//ConstantOp at loc("test/HIR/matmul.mlir":110:8)
//constant [2:0] v8 = 3'd4;

//ConstantOp at loc("test/HIR/matmul.mlir":111:9)
//constant [4:0] v9 = 5'd16;

//ForOp at loc("test/HIR/matmul.mlir":113:3)

//{ Loop10

reg[31:0] idx10 ;
reg[4:0] ub10 ;
reg[0:0] step10 ;
wire tloop_in10;
reg tloop10;
reg tfinish10;
always@(posedge clk) begin
 if(/*tstart=*/ tstartdelay[0]) begin
   idx10 <= /*v4=*/ 1'd0; //lower bound.
   step10 <= /*v5=*/ 1'd1;
   ub10 <= /*v9=*/ 5'd16;
   tloop10 <= (/*v9=*/ 5'd16 > /*v4=*/ 1'd0);
   tfinish10 <=!(/*v9=*/ 5'd16 > /*v4=*/ 1'd0);
 end
 else if (tloop_in10) begin
   idx10 <= idx10 + step10; //increment
   tloop10 <= (idx10 + step10) < ub10;
   tfinish10 <= !((idx10 + step10) < ub10);
 end
 else begin
   tloop10 <= 1'b0;
   tfinish10 <= 1'b0;
 end
end
//Loop10 body
//printTimeOffset
reg tloop10delay[1:0] = '{default:0} ;
always@(*) tloop10delay[0] <= tloop10;
generate
genvar i11;

for(i11 = 1; i11<= 1; i11= i11 + 1) begin
always@(posedge clk) begin
tloop10delay[i11] <= tloop10delay[i11-1];
end
end
endgenerate


//UnrollForOp at loc("test/HIR/matmul.mlir":114:14)

//{ Unrolled body 0 of loop12.
//DEBUG: /*idx12=*/ 1'd0, expected 0

//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop13 = tloop10delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 0][0] = tloop10delay[0];
assign v0_addr_input[/*idx12=*/ 0][0] = {idx10[3:0]};
wire[31:0] v14 = v0_rd_data[/*idx12=*/ 0];
assign v0_rd_en_input[/*idx12=*/ 0][0] = tloop10delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[0] = tloop10delay[1];
assign v1_addr_input[0] = {idx10[3:0], /*idx12=*/ 4'd0};
assign v1_wr_en_input[0] = tloop10delay[1];
assign v1_wr_data_valid[0] = tloop10delay[1];
assign v1_wr_data_input[0] = v14;


//TerminatorOp

//} Unrolled body 0 of loop12.
//DEBUG: /*idx12=*/ 1'd0, expected 0

//{ Unrolled body 1 of loop12.
//DEBUG: /*idx12=*/ 1'd1, expected 1
//printTimeOffset
reg tloop13delay[1:0] = '{default:0} ;
always@(*) tloop13delay[0] <= tloop13;
generate
genvar i16;

for(i16 = 1; i16<= 1; i16= i16 + 1) begin
always@(posedge clk) begin
tloop13delay[i16] <= tloop13delay[i16-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop15 = tloop13delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 1][0] = tloop13delay[0];
assign v0_addr_input[/*idx12=*/ 1][0] = {idx10[3:0]};
wire[31:0] v17 = v0_rd_data[/*idx12=*/ 1];
assign v0_rd_en_input[/*idx12=*/ 1][0] = tloop13delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[1] = tloop13delay[1];
assign v1_addr_input[1] = {idx10[3:0], /*idx12=*/ 4'd1};
assign v1_wr_en_input[1] = tloop13delay[1];
assign v1_wr_data_valid[1] = tloop13delay[1];
assign v1_wr_data_input[1] = v17;


//TerminatorOp

//} Unrolled body 1 of loop12.
//DEBUG: /*idx12=*/ 1'd1, expected 1

//{ Unrolled body 2 of loop12.
//DEBUG: /*idx12=*/ 2'd2, expected 2
//printTimeOffset
reg tloop15delay[1:0] = '{default:0} ;
always@(*) tloop15delay[0] <= tloop15;
generate
genvar i19;

for(i19 = 1; i19<= 1; i19= i19 + 1) begin
always@(posedge clk) begin
tloop15delay[i19] <= tloop15delay[i19-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop18 = tloop15delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 2][0] = tloop15delay[0];
assign v0_addr_input[/*idx12=*/ 2][0] = {idx10[3:0]};
wire[31:0] v20 = v0_rd_data[/*idx12=*/ 2];
assign v0_rd_en_input[/*idx12=*/ 2][0] = tloop15delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[2] = tloop15delay[1];
assign v1_addr_input[2] = {idx10[3:0], /*idx12=*/ 4'd2};
assign v1_wr_en_input[2] = tloop15delay[1];
assign v1_wr_data_valid[2] = tloop15delay[1];
assign v1_wr_data_input[2] = v20;


//TerminatorOp

//} Unrolled body 2 of loop12.
//DEBUG: /*idx12=*/ 2'd2, expected 2

//{ Unrolled body 3 of loop12.
//DEBUG: /*idx12=*/ 2'd3, expected 3
//printTimeOffset
reg tloop18delay[1:0] = '{default:0} ;
always@(*) tloop18delay[0] <= tloop18;
generate
genvar i22;

for(i22 = 1; i22<= 1; i22= i22 + 1) begin
always@(posedge clk) begin
tloop18delay[i22] <= tloop18delay[i22-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop21 = tloop18delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 3][0] = tloop18delay[0];
assign v0_addr_input[/*idx12=*/ 3][0] = {idx10[3:0]};
wire[31:0] v23 = v0_rd_data[/*idx12=*/ 3];
assign v0_rd_en_input[/*idx12=*/ 3][0] = tloop18delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[3] = tloop18delay[1];
assign v1_addr_input[3] = {idx10[3:0], /*idx12=*/ 4'd3};
assign v1_wr_en_input[3] = tloop18delay[1];
assign v1_wr_data_valid[3] = tloop18delay[1];
assign v1_wr_data_input[3] = v23;


//TerminatorOp

//} Unrolled body 3 of loop12.
//DEBUG: /*idx12=*/ 2'd3, expected 3

//{ Unrolled body 4 of loop12.
//DEBUG: /*idx12=*/ 3'd4, expected 4
//printTimeOffset
reg tloop21delay[1:0] = '{default:0} ;
always@(*) tloop21delay[0] <= tloop21;
generate
genvar i25;

for(i25 = 1; i25<= 1; i25= i25 + 1) begin
always@(posedge clk) begin
tloop21delay[i25] <= tloop21delay[i25-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop24 = tloop21delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 4][0] = tloop21delay[0];
assign v0_addr_input[/*idx12=*/ 4][0] = {idx10[3:0]};
wire[31:0] v26 = v0_rd_data[/*idx12=*/ 4];
assign v0_rd_en_input[/*idx12=*/ 4][0] = tloop21delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[4] = tloop21delay[1];
assign v1_addr_input[4] = {idx10[3:0], /*idx12=*/ 4'd4};
assign v1_wr_en_input[4] = tloop21delay[1];
assign v1_wr_data_valid[4] = tloop21delay[1];
assign v1_wr_data_input[4] = v26;


//TerminatorOp

//} Unrolled body 4 of loop12.
//DEBUG: /*idx12=*/ 3'd4, expected 4

//{ Unrolled body 5 of loop12.
//DEBUG: /*idx12=*/ 3'd5, expected 5
//printTimeOffset
reg tloop24delay[1:0] = '{default:0} ;
always@(*) tloop24delay[0] <= tloop24;
generate
genvar i28;

for(i28 = 1; i28<= 1; i28= i28 + 1) begin
always@(posedge clk) begin
tloop24delay[i28] <= tloop24delay[i28-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop27 = tloop24delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 5][0] = tloop24delay[0];
assign v0_addr_input[/*idx12=*/ 5][0] = {idx10[3:0]};
wire[31:0] v29 = v0_rd_data[/*idx12=*/ 5];
assign v0_rd_en_input[/*idx12=*/ 5][0] = tloop24delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[5] = tloop24delay[1];
assign v1_addr_input[5] = {idx10[3:0], /*idx12=*/ 4'd5};
assign v1_wr_en_input[5] = tloop24delay[1];
assign v1_wr_data_valid[5] = tloop24delay[1];
assign v1_wr_data_input[5] = v29;


//TerminatorOp

//} Unrolled body 5 of loop12.
//DEBUG: /*idx12=*/ 3'd5, expected 5

//{ Unrolled body 6 of loop12.
//DEBUG: /*idx12=*/ 3'd6, expected 6
//printTimeOffset
reg tloop27delay[1:0] = '{default:0} ;
always@(*) tloop27delay[0] <= tloop27;
generate
genvar i31;

for(i31 = 1; i31<= 1; i31= i31 + 1) begin
always@(posedge clk) begin
tloop27delay[i31] <= tloop27delay[i31-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop30 = tloop27delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 6][0] = tloop27delay[0];
assign v0_addr_input[/*idx12=*/ 6][0] = {idx10[3:0]};
wire[31:0] v32 = v0_rd_data[/*idx12=*/ 6];
assign v0_rd_en_input[/*idx12=*/ 6][0] = tloop27delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[6] = tloop27delay[1];
assign v1_addr_input[6] = {idx10[3:0], /*idx12=*/ 4'd6};
assign v1_wr_en_input[6] = tloop27delay[1];
assign v1_wr_data_valid[6] = tloop27delay[1];
assign v1_wr_data_input[6] = v32;


//TerminatorOp

//} Unrolled body 6 of loop12.
//DEBUG: /*idx12=*/ 3'd6, expected 6

//{ Unrolled body 7 of loop12.
//DEBUG: /*idx12=*/ 3'd7, expected 7
//printTimeOffset
reg tloop30delay[1:0] = '{default:0} ;
always@(*) tloop30delay[0] <= tloop30;
generate
genvar i34;

for(i34 = 1; i34<= 1; i34= i34 + 1) begin
always@(posedge clk) begin
tloop30delay[i34] <= tloop30delay[i34-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop33 = tloop30delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 7][0] = tloop30delay[0];
assign v0_addr_input[/*idx12=*/ 7][0] = {idx10[3:0]};
wire[31:0] v35 = v0_rd_data[/*idx12=*/ 7];
assign v0_rd_en_input[/*idx12=*/ 7][0] = tloop30delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[7] = tloop30delay[1];
assign v1_addr_input[7] = {idx10[3:0], /*idx12=*/ 4'd7};
assign v1_wr_en_input[7] = tloop30delay[1];
assign v1_wr_data_valid[7] = tloop30delay[1];
assign v1_wr_data_input[7] = v35;


//TerminatorOp

//} Unrolled body 7 of loop12.
//DEBUG: /*idx12=*/ 3'd7, expected 7

//{ Unrolled body 8 of loop12.
//DEBUG: /*idx12=*/ 4'd8, expected 8
//printTimeOffset
reg tloop33delay[1:0] = '{default:0} ;
always@(*) tloop33delay[0] <= tloop33;
generate
genvar i37;

for(i37 = 1; i37<= 1; i37= i37 + 1) begin
always@(posedge clk) begin
tloop33delay[i37] <= tloop33delay[i37-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop36 = tloop33delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 8][0] = tloop33delay[0];
assign v0_addr_input[/*idx12=*/ 8][0] = {idx10[3:0]};
wire[31:0] v38 = v0_rd_data[/*idx12=*/ 8];
assign v0_rd_en_input[/*idx12=*/ 8][0] = tloop33delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[8] = tloop33delay[1];
assign v1_addr_input[8] = {idx10[3:0], /*idx12=*/ 4'd8};
assign v1_wr_en_input[8] = tloop33delay[1];
assign v1_wr_data_valid[8] = tloop33delay[1];
assign v1_wr_data_input[8] = v38;


//TerminatorOp

//} Unrolled body 8 of loop12.
//DEBUG: /*idx12=*/ 4'd8, expected 8

//{ Unrolled body 9 of loop12.
//DEBUG: /*idx12=*/ 4'd9, expected 9
//printTimeOffset
reg tloop36delay[1:0] = '{default:0} ;
always@(*) tloop36delay[0] <= tloop36;
generate
genvar i40;

for(i40 = 1; i40<= 1; i40= i40 + 1) begin
always@(posedge clk) begin
tloop36delay[i40] <= tloop36delay[i40-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop39 = tloop36delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 9][0] = tloop36delay[0];
assign v0_addr_input[/*idx12=*/ 9][0] = {idx10[3:0]};
wire[31:0] v41 = v0_rd_data[/*idx12=*/ 9];
assign v0_rd_en_input[/*idx12=*/ 9][0] = tloop36delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[9] = tloop36delay[1];
assign v1_addr_input[9] = {idx10[3:0], /*idx12=*/ 4'd9};
assign v1_wr_en_input[9] = tloop36delay[1];
assign v1_wr_data_valid[9] = tloop36delay[1];
assign v1_wr_data_input[9] = v41;


//TerminatorOp

//} Unrolled body 9 of loop12.
//DEBUG: /*idx12=*/ 4'd9, expected 9

//{ Unrolled body 10 of loop12.
//DEBUG: /*idx12=*/ 4'd10, expected 10
//printTimeOffset
reg tloop39delay[1:0] = '{default:0} ;
always@(*) tloop39delay[0] <= tloop39;
generate
genvar i43;

for(i43 = 1; i43<= 1; i43= i43 + 1) begin
always@(posedge clk) begin
tloop39delay[i43] <= tloop39delay[i43-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop42 = tloop39delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 10][0] = tloop39delay[0];
assign v0_addr_input[/*idx12=*/ 10][0] = {idx10[3:0]};
wire[31:0] v44 = v0_rd_data[/*idx12=*/ 10];
assign v0_rd_en_input[/*idx12=*/ 10][0] = tloop39delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[10] = tloop39delay[1];
assign v1_addr_input[10] = {idx10[3:0], /*idx12=*/ 4'd10};
assign v1_wr_en_input[10] = tloop39delay[1];
assign v1_wr_data_valid[10] = tloop39delay[1];
assign v1_wr_data_input[10] = v44;


//TerminatorOp

//} Unrolled body 10 of loop12.
//DEBUG: /*idx12=*/ 4'd10, expected 10

//{ Unrolled body 11 of loop12.
//DEBUG: /*idx12=*/ 4'd11, expected 11
//printTimeOffset
reg tloop42delay[1:0] = '{default:0} ;
always@(*) tloop42delay[0] <= tloop42;
generate
genvar i46;

for(i46 = 1; i46<= 1; i46= i46 + 1) begin
always@(posedge clk) begin
tloop42delay[i46] <= tloop42delay[i46-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop45 = tloop42delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 11][0] = tloop42delay[0];
assign v0_addr_input[/*idx12=*/ 11][0] = {idx10[3:0]};
wire[31:0] v47 = v0_rd_data[/*idx12=*/ 11];
assign v0_rd_en_input[/*idx12=*/ 11][0] = tloop42delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[11] = tloop42delay[1];
assign v1_addr_input[11] = {idx10[3:0], /*idx12=*/ 4'd11};
assign v1_wr_en_input[11] = tloop42delay[1];
assign v1_wr_data_valid[11] = tloop42delay[1];
assign v1_wr_data_input[11] = v47;


//TerminatorOp

//} Unrolled body 11 of loop12.
//DEBUG: /*idx12=*/ 4'd11, expected 11

//{ Unrolled body 12 of loop12.
//DEBUG: /*idx12=*/ 4'd12, expected 12
//printTimeOffset
reg tloop45delay[1:0] = '{default:0} ;
always@(*) tloop45delay[0] <= tloop45;
generate
genvar i49;

for(i49 = 1; i49<= 1; i49= i49 + 1) begin
always@(posedge clk) begin
tloop45delay[i49] <= tloop45delay[i49-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop48 = tloop45delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 12][0] = tloop45delay[0];
assign v0_addr_input[/*idx12=*/ 12][0] = {idx10[3:0]};
wire[31:0] v50 = v0_rd_data[/*idx12=*/ 12];
assign v0_rd_en_input[/*idx12=*/ 12][0] = tloop45delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[12] = tloop45delay[1];
assign v1_addr_input[12] = {idx10[3:0], /*idx12=*/ 4'd12};
assign v1_wr_en_input[12] = tloop45delay[1];
assign v1_wr_data_valid[12] = tloop45delay[1];
assign v1_wr_data_input[12] = v50;


//TerminatorOp

//} Unrolled body 12 of loop12.
//DEBUG: /*idx12=*/ 4'd12, expected 12

//{ Unrolled body 13 of loop12.
//DEBUG: /*idx12=*/ 4'd13, expected 13
//printTimeOffset
reg tloop48delay[1:0] = '{default:0} ;
always@(*) tloop48delay[0] <= tloop48;
generate
genvar i52;

for(i52 = 1; i52<= 1; i52= i52 + 1) begin
always@(posedge clk) begin
tloop48delay[i52] <= tloop48delay[i52-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop51 = tloop48delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 13][0] = tloop48delay[0];
assign v0_addr_input[/*idx12=*/ 13][0] = {idx10[3:0]};
wire[31:0] v53 = v0_rd_data[/*idx12=*/ 13];
assign v0_rd_en_input[/*idx12=*/ 13][0] = tloop48delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[13] = tloop48delay[1];
assign v1_addr_input[13] = {idx10[3:0], /*idx12=*/ 4'd13};
assign v1_wr_en_input[13] = tloop48delay[1];
assign v1_wr_data_valid[13] = tloop48delay[1];
assign v1_wr_data_input[13] = v53;


//TerminatorOp

//} Unrolled body 13 of loop12.
//DEBUG: /*idx12=*/ 4'd13, expected 13

//{ Unrolled body 14 of loop12.
//DEBUG: /*idx12=*/ 4'd14, expected 14
//printTimeOffset
reg tloop51delay[1:0] = '{default:0} ;
always@(*) tloop51delay[0] <= tloop51;
generate
genvar i55;

for(i55 = 1; i55<= 1; i55= i55 + 1) begin
always@(posedge clk) begin
tloop51delay[i55] <= tloop51delay[i55-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop54 = tloop51delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 14][0] = tloop51delay[0];
assign v0_addr_input[/*idx12=*/ 14][0] = {idx10[3:0]};
wire[31:0] v56 = v0_rd_data[/*idx12=*/ 14];
assign v0_rd_en_input[/*idx12=*/ 14][0] = tloop51delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[14] = tloop51delay[1];
assign v1_addr_input[14] = {idx10[3:0], /*idx12=*/ 4'd14};
assign v1_wr_en_input[14] = tloop51delay[1];
assign v1_wr_data_valid[14] = tloop51delay[1];
assign v1_wr_data_input[14] = v56;


//TerminatorOp

//} Unrolled body 14 of loop12.
//DEBUG: /*idx12=*/ 4'd14, expected 14

//{ Unrolled body 15 of loop12.
//DEBUG: /*idx12=*/ 4'd15, expected 15
//printTimeOffset
reg tloop54delay[1:0] = '{default:0} ;
always@(*) tloop54delay[0] <= tloop54;
generate
genvar i58;

for(i58 = 1; i58<= 1; i58= i58 + 1) begin
always@(posedge clk) begin
tloop54delay[i58] <= tloop54delay[i58-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/matmul.mlir":115:7)
wire tloop57 = tloop54delay[1];

//MemReadOp at loc("test/HIR/matmul.mlir":116:12)
assign v0_addr_valid[/*idx12=*/ 15][0] = tloop54delay[0];
assign v0_addr_input[/*idx12=*/ 15][0] = {idx10[3:0]};
wire[31:0] v59 = v0_rd_data[/*idx12=*/ 15];
assign v0_rd_en_input[/*idx12=*/ 15][0] = tloop54delay[0];


//MemWriteOp at loc("test/HIR/matmul.mlir":117:7)
assign v1_addr_valid[15] = tloop54delay[1];
assign v1_addr_input[15] = {idx10[3:0], /*idx12=*/ 4'd15};
assign v1_wr_en_input[15] = tloop54delay[1];
assign v1_wr_data_valid[15] = tloop54delay[1];
assign v1_wr_data_input[15] = v59;


//TerminatorOp

//} Unrolled body 15 of loop12.
//DEBUG: /*idx12=*/ 4'd15, expected 15

//{ Assign tlast of prev UnrollForLoop
wire t60;
assign t60 = tloop57;
//printTimeOffset
reg t60delay[0:0] = '{default:0} ;
always@(*) t60delay[0] <= t60;
generate
genvar i61;

for(i61 = 1; i61<= 0; i61= i61 + 1) begin
always@(posedge clk) begin
t60delay[i61] <= t60delay[i61-1];
end
end
endgenerate


//} Assign tlast of prev UnrollForLoop
//YieldOp at loc("test/HIR/matmul.mlir":119:5)
assign tloop_in10 = t60delay[0];

//TerminatorOp

//} Loop10
//printTimeOffset
reg tfinish10delay[0:0] = '{default:0} ;
always@(*) tfinish10delay[0] <= tfinish10;
generate
genvar i62;

for(i62 = 1; i62<= 0; i62= i62 + 1) begin
always@(posedge clk) begin
tfinish10delay[i62] <= tfinish10delay[i62-1];
end
end
endgenerate


//ReturnOp at loc("test/HIR/matmul.mlir":121:3)
endmodule
module matmul(
//Outputs.

//Inputs.

//MemrefType : port = r.
output reg[7:0] v0_addr,
output wire v0_rd_en,
input wire[31:0] v0_rd_data,
//MemrefType : port = r.
output reg[7:0] v1_addr,
output wire v1_rd_en,
input wire[31:0] v1_rd_data,
//MemrefType : port = w.
output reg[7:0] v2_addr,
output wire v2_wr_en,
output reg[31:0] v2_wr_data,
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

//Unused memref v0.

//Unused memref v1.

//Unused memref v2.

//printTimeOffset
reg tstartdelay[0:0] = '{default:0} ;
always@(*) tstartdelay[0] <= tstart;
generate
genvar i4;

for(i4 = 1; i4<= 0; i4= i4 + 1) begin
always@(posedge clk) begin
tstartdelay[i4] <= tstartdelay[i4-1];
end
end
endgenerate


//ConstantOp at loc("test/HIR/matmul.mlir":129:11)
//constant [5:0] v5 = 6'd32;

//AllocOp at loc("test/HIR/matmul.mlir":131:12)
//strMemrefInstDecl
reg[3:0] v6_addr[15:0];
wire v6_rd_en[15:0];
logic[31:0] v6_rd_data[15:0];
//strMemrefSelDecl
//Unused memref v6.

//strMemrefInstDecl
reg[3:0] v7_addr[15:0];
 wire v7_wr_en[15:0];
reg[31:0] v7_wr_data[15:0];
//strMemrefSelDecl
//Unused memref v7.


 //Instantiate Memory.
for(genvar i0= 0; i0<16;i0+=1) begin
bram_tdp_rf_rf#(.SIZE(16), .WIDTH(32))  bram_inst0(
.clka(clk),
.clkb(clk),
.ena(v6_rd_en[i0]),
.enb(v7_wr_en[i0]),
.wea(0),
.web(v7_wr_en[i0]),
.addra(v6_addr[i0]),
.addrb(v7_addr[i0]),
.dia(0),
.dib(v7_wr_data[i0]),
.doa(v6_rd_data[i0]),
.dob(/*ignored*/)
);
end

//AllocOp at loc("test/HIR/matmul.mlir":132:12)
//strMemrefInstDecl
wire v8_rd_en[15:0][15:0];
logic[31:0] v8_rd_data[15:0][15:0];
//strMemrefSelDecl
//Unused memref v8.

//strMemrefInstDecl
 wire v9_wr_en[15:0][15:0];
reg[31:0] v9_wr_data[15:0][15:0];
//strMemrefSelDecl
//Unused memref v9.


 //Instantiate Memory.
for(genvar i0= 0; i0<16;i0+=1) begin
for(genvar i1= 0; i1<16;i1+=1) begin
always@(posedge clk) begin
  if(v9_wr_en[i0][i1]) v8_rd_data[i0][i1] <= v9_wr_data[i0][i1];
end
end
end

//AllocOp at loc("test/HIR/matmul.mlir":133:12)
//strMemrefInstDecl
reg[3:0] v10_addr[15:0];
wire v10_rd_en[15:0];
logic[31:0] v10_rd_data[15:0];
//strMemrefSelDecl
//Unused memref v10.

//strMemrefInstDecl
reg[3:0] v11_addr[15:0];
 wire v11_wr_en[15:0];
reg[31:0] v11_wr_data[15:0];
//strMemrefSelDecl
//Unused memref v11.


 //Instantiate Memory.
for(genvar i0= 0; i0<16;i0+=1) begin
bram_tdp_rf_rf#(.SIZE(16), .WIDTH(32))  bram_inst1(
.clka(clk),
.clkb(clk),
.ena(v10_rd_en[i0]),
.enb(v11_wr_en[i0]),
.wea(0),
.web(v11_wr_en[i0]),
.addra(v10_addr[i0]),
.addrb(v11_addr[i0]),
.dia(0),
.dib(v11_wr_data[i0]),
.doa(v10_rd_data[i0]),
.dob(/*ignored*/)
);
end

//CallOp at loc("test/HIR/matmul.mlir":135:3)
readA readA12(v0_addr,
v0_rd_en,
v0_rd_data,
v7_addr,
v7_wr_en,
v7_wr_data,
tstartdelay[0],
clk
);

//CallOp at loc("test/HIR/matmul.mlir":138:13)
wire v13;
readB readB14(v13,
v1_addr,
v1_rd_en,
v1_rd_data,
v9_wr_en,
v9_wr_data,
tstartdelay[0],
clk
);
//printTimeOffset
reg v13delay[0:0] = '{default:0} ;
always@(*) v13delay[0] <= v13;
generate
genvar i15;

for(i15 = 1; i15<= 0; i15= i15 + 1) begin
always@(posedge clk) begin
v13delay[i15] <= v13delay[i15-1];
end
end
endgenerate


//CallOp at loc("test/HIR/matmul.mlir":141:3)
kernel kernel16(v6_addr,
v6_rd_en,
v6_rd_data,
v8_rd_en,
v8_rd_data,
v11_addr,
v11_wr_en,
v11_wr_data,
v13delay[0],
clk
);

//DelayOp at loc("test/HIR/matmul.mlir":146:11)
reg[0:0]shiftreg18[/*v5=*/ 32:0] = '{default:0};
always@(*) shiftreg18[0] <= v13;
always@(posedge clk) shiftreg18[/*v5=*/ 32:1] <= shiftreg18[/*v5=*/ 31:0];
wire v17 = shiftreg18[/*v5=*/ 32];
//printTimeOffset
reg v17delay[0:0] = '{default:0} ;
always@(*) v17delay[0] <= v17;
generate
genvar i19;

for(i19 = 1; i19<= 0; i19= i19 + 1) begin
always@(posedge clk) begin
v17delay[i19] <= v17delay[i19-1];
end
end
endgenerate


//CallOp at loc("test/HIR/matmul.mlir":147:3)
writeC writeC20(v10_addr,
v10_rd_en,
v10_rd_data,
v2_addr,
v2_wr_en,
v2_wr_data,
v17delay[0],
clk
);

//ReturnOp at loc("test/HIR/matmul.mlir":150:3)
endmodule
