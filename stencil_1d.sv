`default_nettype none
`include "helper.sv"
module stencil_1d(
//Outputs.

//Inputs.

//MemrefType : port = r.
output reg[5:0] v0_addr,
output wire v0_rd_en,
input wire[31:0] v0_rd_data,
//MemrefType : port = w.
output reg[5:0] v1_addr,
output wire v1_wr_en,
output reg[31:0] v1_wr_data,
//IntegerType.
input wire[31:0] v2,
//IntegerType.
input wire[31:0] v3,
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

wire v0_addr_valid  [2:0] ;
wire [5:0] v0_addr_input  [2:0];
 always@(*) begin
if(v0_addr_valid[0] )
v0_addr = v0_addr_input[0];
else if (v0_addr_valid[1])
v0_addr = v0_addr_input[1];
else if (v0_addr_valid[2])
v0_addr = v0_addr_input[2];
else
 v0_addr = 'x;
end

wire [2:0] v0_rd_en_input ;
assign v0_rd_en  =| v0_rd_en_input ;


wire v1_addr_valid  [0:0] ;
wire [5:0] v1_addr_input  [0:0];
 always@(*) begin
if(v1_addr_valid[0] )
v1_addr = v1_addr_input[0];
else
 v1_addr = 'x;
end

wire [0:0] v1_wr_en_input ;
assign v1_wr_en  =| v1_wr_en_input ;
wire v1_wr_data_valid  [0:0] ;
wire [31:0] v1_wr_data_input  [0:0];
 always@(*) begin
if(v1_wr_data_valid[0] )
v1_wr_data = v1_wr_data_input[0];
else
 v1_wr_data = 'x;
end


//printTimeOffset
reg tstartdelay[2:0] = '{default:0} ;
always@(*) tstartdelay[0] <= tstart;
generate
genvar i5;

for(i5 = 1; i5<= 2; i5= i5 + 1) begin
always@(posedge clk) begin
tstartdelay[i5] <= tstartdelay[i5-1];
end
end
endgenerate


//ConstantOp at loc("test/HIR/stencil_1d.mlir":6:9)
//constant v6 = 1'd0;

//ConstantOp at loc("test/HIR/stencil_1d.mlir":7:9)
//constant v7 = 1'd1;

//ConstantOp at loc("test/HIR/stencil_1d.mlir":8:9)
//constant [1:0] v8 = 2'd2;

//ConstantOp at loc("test/HIR/stencil_1d.mlir":9:9)
//constant [1:0] v9 = 2'd3;

//ConstantOp at loc("test/HIR/stencil_1d.mlir":10:9)
//constant [2:0] v10 = 3'd4;

//ConstantOp at loc("test/HIR/stencil_1d.mlir":11:9)
//constant [2:0] v11 = 3'd5;

//ConstantOp at loc("test/HIR/stencil_1d.mlir":12:9)
//constant [6:0] v12 = 7'd64;

//AllocOp at loc("test/HIR/stencil_1d.mlir":14:16)
//strMemrefInstDecl
wire v13_rd_en[1:0];
logic[31:0] v13_rd_data[1:0];
//strMemrefSelDecl
wire [0:0] v13_rd_en_input [1:0];
generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
assign v13_rd_en [i0] =| v13_rd_en_input [i0];
end
endgenerate


//strMemrefInstDecl
 wire v14_wr_en[1:0];
reg[31:0] v14_wr_data[1:0];
//strMemrefSelDecl
wire [1:0] v14_wr_en_input [1:0];
generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
assign v14_wr_en [i0] =| v14_wr_en_input [i0];
end
endgenerate
wire v14_wr_data_valid [1:0] [1:0] ;
wire [31:0] v14_wr_data_input [1:0] [1:0];
 generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
always@(*) begin
if(v14_wr_data_valid[i0][0] )
v14_wr_data[i0] = v14_wr_data_input[i0][0];
else if (v14_wr_data_valid[i0][1])
v14_wr_data[i0] = v14_wr_data_input[i0][1];
else
 v14_wr_data[i0] = 'x;
end
end
endgenerate



 //Instantiate Memory.
for(genvar i0= 0; i0<2;i0+=1) begin
always@(posedge clk) begin
  if(v14_wr_en[i0]) v13_rd_data[i0] <= v14_wr_data[i0];
end
end

//MemReadOp at loc("test/HIR/stencil_1d.mlir":18:11)
assign v0_addr_valid[0] = tstartdelay[0];
assign v0_addr_input[0] = {/*v6=*/ 6'd0};
wire[31:0] v15 = v0_rd_data;
assign v0_rd_en_input[0] = tstartdelay[0];


//DelayOp at loc("test/HIR/stencil_1d.mlir":20:12)
reg[31:0]shiftreg17[/*v7=*/ 1:0] = '{default:0};
always@(*) shiftreg17[0] <= v15;
always@(posedge clk) shiftreg17[/*v7=*/ 1:1] <= shiftreg17[/*v7=*/ 0:0];
wire [31:0] v16 = shiftreg17[/*v7=*/ 1];

//MemReadOp at loc("test/HIR/stencil_1d.mlir":22:11)
assign v0_addr_valid[1] = tstartdelay[1];
assign v0_addr_input[1] = {/*v7=*/ 6'd1};
wire[31:0] v18 = v0_rd_data;
assign v0_rd_en_input[1] = tstartdelay[1];


//MemWriteOp at loc("test/HIR/stencil_1d.mlir":25:3)
assign v14_wr_en_input[/*v6=*/ 0][0] = tstartdelay[2];
assign v14_wr_data_valid[/*v6=*/ 0][0] = tstartdelay[2];
assign v14_wr_data_input[/*v6=*/ 0][0] = v16;


//MemWriteOp at loc("test/HIR/stencil_1d.mlir":27:3)
assign v14_wr_en_input[/*v7=*/ 1][0] = tstartdelay[2];
assign v14_wr_data_valid[/*v7=*/ 1][0] = tstartdelay[2];
assign v14_wr_data_input[/*v7=*/ 1][0] = v18;


//AllocOp at loc("test/HIR/stencil_1d.mlir":30:14)
//strMemrefInstDecl
wire v19_rd_en[1:0];
logic[31:0] v19_rd_data[1:0];
//strMemrefSelDecl
wire [0:0] v19_rd_en_input [1:0];
generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
assign v19_rd_en [i0] =| v19_rd_en_input [i0];
end
endgenerate


//strMemrefInstDecl
 wire v20_wr_en[1:0];
reg[31:0] v20_wr_data[1:0];
//strMemrefSelDecl
wire [0:0] v20_wr_en_input [1:0];
generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
assign v20_wr_en [i0] =| v20_wr_en_input [i0];
end
endgenerate
wire v20_wr_data_valid [1:0] [0:0] ;
wire [31:0] v20_wr_data_input [1:0] [0:0];
 generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
always@(*) begin
if(v20_wr_data_valid[i0][0] )
v20_wr_data[i0] = v20_wr_data_input[i0][0];
else
 v20_wr_data[i0] = 'x;
end
end
endgenerate



 //Instantiate Memory.
for(genvar i0= 0; i0<2;i0+=1) begin
always@(posedge clk) begin
  if(v20_wr_en[i0]) v19_rd_data[i0] <= v20_wr_data[i0];
end
end

//MemWriteOp at loc("test/HIR/stencil_1d.mlir":32:3)
assign v20_wr_en_input[/*v6=*/ 0][0] = tstartdelay[0];
assign v20_wr_data_valid[/*v6=*/ 0][0] = tstartdelay[0];
assign v20_wr_data_input[/*v6=*/ 0][0] = v2;


//MemWriteOp at loc("test/HIR/stencil_1d.mlir":34:3)
assign v20_wr_en_input[/*v7=*/ 1][0] = tstartdelay[0];
assign v20_wr_data_valid[/*v7=*/ 1][0] = tstartdelay[0];
assign v20_wr_data_input[/*v7=*/ 1][0] = v3;


//ForOp at loc("test/HIR/stencil_1d.mlir":37:3)

//{ Loop21

reg[31:0] idx21 ;
reg[6:0] ub21 ;
reg[0:0] step21 ;
wire tloop_in21;
reg tloop21;
reg tfinish21;
always@(posedge clk) begin
 if(/*tstart=*/ tstartdelay[2]) begin
   idx21 <= /*v7=*/ 1'd1; //lower bound.
   step21 <= /*v7=*/ 1'd1;
   ub21 <= /*v12=*/ 7'd64;
   tloop21 <= (/*v12=*/ 7'd64 > /*v7=*/ 1'd1);
   tfinish21 <=!(/*v12=*/ 7'd64 > /*v7=*/ 1'd1);
 end
 else if (tloop_in21) begin
   idx21 <= idx21 + step21; //increment
   tloop21 <= (idx21 + step21) < ub21;
   tfinish21 <= !((idx21 + step21) < ub21);
 end
 else begin
   tloop21 <= 1'b0;
   tfinish21 <= 1'b0;
 end
end
//Loop21 body
//printTimeOffset
reg tloop21delay[2:0] = '{default:0} ;
always@(*) tloop21delay[0] <= tloop21;
generate
genvar i22;

for(i22 = 1; i22<= 2; i22= i22 + 1) begin
always@(posedge clk) begin
tloop21delay[i22] <= tloop21delay[i22-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/stencil_1d.mlir":39:7)
assign tloop_in21 = tloop21delay[0];

//MemReadOp at loc("test/HIR/stencil_1d.mlir":41:13)
wire[31:0] v23 = v13_rd_data[/*v6=*/ 0];
assign v13_rd_en_input[/*v6=*/ 0][0] = tloop21delay[1];


//MemReadOp at loc("test/HIR/stencil_1d.mlir":43:13)
wire[31:0] v24 = v13_rd_data[/*v7=*/ 1];
assign v13_rd_en_input[/*v7=*/ 1][0] = tloop21delay[1];


//AddOp at loc("test/HIR/stencil_1d.mlir":45:17)
wire [31:0] v25 = idx21 + /*v7=*/ 1'd1;

//MemReadOp at loc("test/HIR/stencil_1d.mlir":46:13)
assign v0_addr_valid[2] = tloop21delay[0];
assign v0_addr_input[2] = {v25[5:0]};
wire[31:0] v26 = v0_rd_data;
assign v0_rd_en_input[2] = tloop21delay[0];


//MemWriteOp at loc("test/HIR/stencil_1d.mlir":49:7)
assign v14_wr_en_input[/*v6=*/ 0][1] = tloop21delay[1];
assign v14_wr_data_valid[/*v6=*/ 0][1] = tloop21delay[1];
assign v14_wr_data_input[/*v6=*/ 0][1] = v24;


//MemWriteOp at loc("test/HIR/stencil_1d.mlir":51:7)
assign v14_wr_en_input[/*v7=*/ 1][1] = tloop21delay[1];
assign v14_wr_data_valid[/*v7=*/ 1][1] = tloop21delay[1];
assign v14_wr_data_input[/*v7=*/ 1][1] = v26;


//MemReadOp at loc("test/HIR/stencil_1d.mlir":54:15)
wire[31:0] v27 = v19_rd_data[/*v6=*/ 0];
assign v19_rd_en_input[/*v6=*/ 0][0] = tloop21delay[1];


//MemReadOp at loc("test/HIR/stencil_1d.mlir":56:15)
wire[31:0] v28 = v19_rd_data[/*v7=*/ 1];
assign v19_rd_en_input[/*v7=*/ 1][0] = tloop21delay[1];


//CallOp at loc("test/HIR/stencil_1d.mlir":59:13)
wire [31:0] v29;
weighted_sum weighted_sum30(v29,
v23,
v27,
v24,
v28,
tloop21delay[1],
clk
);

//DelayOp at loc("test/HIR/stencil_1d.mlir":62:13)
reg[31:0]shiftreg32[/*v8=*/ 2:0] = '{default:0};
always@(*) shiftreg32[0] <= idx21;
always@(posedge clk) shiftreg32[/*v8=*/ 2:1] <= shiftreg32[/*v8=*/ 1:0];
wire [31:0] v31 = shiftreg32[/*v8=*/ 2];

//MemWriteOp at loc("test/HIR/stencil_1d.mlir":63:7)
assign v1_addr_valid[0] = tloop21delay[2];
assign v1_addr_input[0] = {v31[5:0]};
assign v1_wr_en_input[0] = tloop21delay[2];
assign v1_wr_data_valid[0] = tloop21delay[2];
assign v1_wr_data_input[0] = v29;


//TerminatorOp

//} Loop21
//printTimeOffset
reg tfinish21delay[0:0] = '{default:0} ;
always@(*) tfinish21delay[0] <= tfinish21;
generate
genvar i33;

for(i33 = 1; i33<= 0; i33= i33 + 1) begin
always@(posedge clk) begin
tfinish21delay[i33] <= tfinish21delay[i33-1];
end
end
endgenerate


//ReturnOp at loc("test/HIR/stencil_1d.mlir":66:3)
endmodule
