`default_nettype none
module floyd_warshall(
//Outputs.

//Inputs.

//IntegerType.
input wire[31:0] v0,
//MemrefType : port = r.
output reg[5:0] v1_addr,
output wire v1_rd_en,
input wire[31:0] v1_rd_data,
//MemrefType : port = w.
output reg[5:0] v2_addr,
output wire v2_wr_en,
output reg[31:0] v2_wr_data,
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

wire v1_addr_valid  [2:0] ;
wire [5:0] v1_addr_input  [2:0];
 wire v1_addr_valid_if [2:0]   ;
wire [5:0] v1_addr_input_if [2:0]  ;
 always@(*) begin
if(v1_addr_valid[0] )
v1_addr = v1_addr_input[0];
else if (v1_addr_valid[1])
v1_addr = v1_addr_input[1];
else if (v1_addr_valid[2])
v1_addr = v1_addr_input[2];
else
 v1_addr = 'x;
end
assign v1_addr_valid_if[0]  = v1_addr_valid  [0];
assign v1_addr_input_if[0]  = v1_addr_input  [0];
assign v1_addr_valid_if[1]  = v1_addr_valid  [1];
assign v1_addr_input_if[1]  = v1_addr_input  [1];
assign v1_addr_valid_if[2]  = v1_addr_valid  [2];
assign v1_addr_input_if[2]  = v1_addr_input  [2];

wire [2:0] v1_rd_en_input ;
wire  v1_rd_en_input_if [2:0] ;
assign v1_rd_en  =| v1_rd_en_input ;
assign v1_rd_en_input_if[0] = v1_rd_en_input  [0];
assign v1_rd_en_input_if[1] = v1_rd_en_input  [1];
assign v1_rd_en_input_if[2] = v1_rd_en_input  [2];


wire v2_addr_valid  [0:0] ;
wire [5:0] v2_addr_input  [0:0];
 wire v2_addr_valid_if [0:0]   ;
wire [5:0] v2_addr_input_if [0:0]  ;
 always@(*) begin
if(v2_addr_valid[0] )
v2_addr = v2_addr_input[0];
else
 v2_addr = 'x;
end
assign v2_addr_valid_if[0]  = v2_addr_valid  [0];
assign v2_addr_input_if[0]  = v2_addr_input  [0];

wire [0:0] v2_wr_en_input ;
wire  v2_wr_en_input_if [0:0] ;
assign v2_wr_en  =| v2_wr_en_input ;
assign v2_wr_en_input_if[0] = v2_wr_en_input  [0];
wire v2_wr_data_valid  [0:0] ;
wire [31:0] v2_wr_data_input  [0:0];
 wire v2_wr_data_valid_if [0:0]   ;
wire [31:0] v2_wr_data_input_if [0:0]  ;
 always@(*) begin
if(v2_wr_data_valid[0] )
v2_wr_data = v2_wr_data_input[0];
else
 v2_wr_data = 'x;
end
assign v2_wr_data_valid_if[0]  = v2_wr_data_valid  [0];
assign v2_wr_data_input_if[0]  = v2_wr_data_input  [0];


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


//ConstantOp at loc("floyd-warshall.mlir":12:8)
//constant v4 = 1'd0;

//ConstantOp at loc("floyd-warshall.mlir":13:8)
//constant v5 = 1'd1;

//ConstantOp at loc("floyd-warshall.mlir":14:8)
//constant [1:0] v6 = 2'd2;

//ConstantOp at loc("floyd-warshall.mlir":15:8)
//constant [1:0] v7 = 2'd3;

//ConstantOp at loc("floyd-warshall.mlir":16:8)
//constant [2:0] v8 = 3'd4;

//ConstantOp at loc("floyd-warshall.mlir":17:8)
//constant [2:0] v9 = 3'd5;

//ConstantOp at loc("floyd-warshall.mlir":18:8)
//constant [2:0] v10 = 3'd6;

//ConstantOp at loc("floyd-warshall.mlir":19:8)
//constant [3:0] v11 = 4'd8;

//ConstantOp at loc("floyd-warshall.mlir":20:8)
//constant [3:0] v12 = 4'd9;

//ForOp at loc("floyd-warshall.mlir":23:3)

//{ Loop13

reg[31:0] idx13 ;
reg[3:0] ub13 ;
reg[0:0] step13 ;
wire tloop_in13;
reg tloop13;
reg tfinish13;
always@(posedge clk) begin
 if(/*tstart=*/ tstart) begin
   idx13 <= /*v4=*/ 1'd0; //lower bound.
   step13 <= /*v5=*/ 1'd1;
   ub13 <= /*v11=*/ 4'd8;
   tloop13 <= (/*v11=*/ 4'd8 > /*v4=*/ 1'd0);
   tfinish13 <=!(/*v11=*/ 4'd8 > /*v4=*/ 1'd0);
 end
 else if (tloop_in13) begin
   idx13 <= idx13 + step13; //increment
   tloop13 <= (idx13 + step13) < ub13;
   tfinish13 <= !((idx13 + step13) < ub13);
 end
 else begin
   tloop13 <= 1'b0;
   tfinish13 <= 1'b0;
 end
end
//Loop13 body
//printTimeOffset
reg tloop13delay[0:0] = '{default:0} ;
always@(*) tloop13delay[0] <= tloop13;
generate
genvar i14;

for(i14 = 1; i14<= 0; i14= i14 + 1) begin
always@(posedge clk) begin
tloop13delay[i14] <= tloop13delay[i14-1];
end
end
endgenerate


//ForOp at loc("floyd-warshall.mlir":26:12)

//{ Loop15

reg[31:0] idx15 ;
reg[3:0] ub15 ;
reg[0:0] step15 ;
wire tloop_in15;
reg tloop15;
reg tfinish15;
always@(posedge clk) begin
 if(/*tstart=*/ tloop13) begin
   idx15 <= /*v4=*/ 1'd0; //lower bound.
   step15 <= /*v5=*/ 1'd1;
   ub15 <= /*v11=*/ 4'd8;
   tloop15 <= (/*v11=*/ 4'd8 > /*v4=*/ 1'd0);
   tfinish15 <=!(/*v11=*/ 4'd8 > /*v4=*/ 1'd0);
 end
 else if (tloop_in15) begin
   idx15 <= idx15 + step15; //increment
   tloop15 <= (idx15 + step15) < ub15;
   tfinish15 <= !((idx15 + step15) < ub15);
 end
 else begin
   tloop15 <= 1'b0;
   tfinish15 <= 1'b0;
 end
end
//Loop15 body
//printTimeOffset
reg tloop15delay[0:0] = '{default:0} ;
always@(*) tloop15delay[0] <= tloop15;
generate
genvar i16;

for(i16 = 1; i16<= 0; i16= i16 + 1) begin
always@(posedge clk) begin
tloop15delay[i16] <= tloop15delay[i16-1];
end
end
endgenerate


//ForOp at loc("floyd-warshall.mlir":29:12)

//{ Loop17

reg[31:0] idx17 ;
reg[3:0] ub17 ;
reg[0:0] step17 ;
wire tloop_in17;
reg tloop17;
reg tfinish17;
always@(posedge clk) begin
 if(/*tstart=*/ tloop15) begin
   idx17 <= /*v4=*/ 1'd0; //lower bound.
   step17 <= /*v5=*/ 1'd1;
   ub17 <= /*v11=*/ 4'd8;
   tloop17 <= (/*v11=*/ 4'd8 > /*v4=*/ 1'd0);
   tfinish17 <=!(/*v11=*/ 4'd8 > /*v4=*/ 1'd0);
 end
 else if (tloop_in17) begin
   idx17 <= idx17 + step17; //increment
   tloop17 <= (idx17 + step17) < ub17;
   tfinish17 <= !((idx17 + step17) < ub17);
 end
 else begin
   tloop17 <= 1'b0;
   tfinish17 <= 1'b0;
 end
end
//Loop17 body
//printTimeOffset
reg tloop17delay[3:0] = '{default:0} ;
always@(*) tloop17delay[0] <= tloop17;
generate
genvar i18;

for(i18 = 1; i18<= 3; i18= i18 + 1) begin
always@(posedge clk) begin
tloop17delay[i18] <= tloop17delay[i18-1];
end
end
endgenerate


//LoadOp at loc("floyd-warshall.mlir":33:18)
assign v1_addr_valid[0] = tloop17;
assign v1_addr_input[0] = {idx15[2:0], idx17[2:0]};
wire[31:0] v19 = v1_rd_data;
assign v1_rd_en_input[0] = tloop17;


//DelayOp at loc("floyd-warshall.mlir":36:19)
reg[31:0]shiftreg21[/*v6=*/ 2:0] = '{default:0};
always@(*) shiftreg21[0] <= v19;
always@(posedge clk) shiftreg21[/*v6=*/ 2:1] <= shiftreg21[/*v6=*/ 1:0];
wire [31:0] v20 = shiftreg21[/*v6=*/ 2];

//LoadOp at loc("floyd-warshall.mlir":39:18)
assign v1_addr_valid[1] = tloop17delay[1];
assign v1_addr_input[1] = {idx15[2:0], idx13[2:0]};
wire[31:0] v22 = v1_rd_data;
assign v1_rd_en_input[1] = tloop17delay[1];


//DelayOp at loc("floyd-warshall.mlir":42:19)
reg[31:0]shiftreg24[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg24[0] <= v22;
always@(posedge clk) shiftreg24[/*v5=*/ 1:1] <= shiftreg24[/*v5=*/ 0:0];
wire [31:0] v23 = shiftreg24[/*v5=*/ 1];

//LoadOp at loc("floyd-warshall.mlir":45:18)
assign v1_addr_valid[2] = tloop17delay[2];
assign v1_addr_input[2] = {idx13[2:0], idx17[2:0]};
wire[31:0] v25 = v1_rd_data;
assign v1_rd_en_input[2] = tloop17delay[2];


//AddOp at loc("floyd-warshall.mlir":49:16)
wire [31:0] v26 = v23 + v25;

//LTOp at loc("floyd-warshall.mlir":52:17)
wire v27 = v20 < v26;

//CallOp at loc("floyd-warshall.mlir":55:16)
wire [31:0] v28;
mux mux29(v28,
v27,
v20,
v26,
tloop17delay[3],
clk
);

//StoreOp at loc("floyd-warshall.mlir":59:9)
assign v2_addr_valid[0] = tloop17delay[3];
assign v2_addr_input[0] = {idx15[2:0], idx17[2:0]};
assign v2_wr_en_input[0] = tloop17delay[3];
assign v2_wr_data_valid[0] = tloop17delay[3];
assign v2_wr_data_input[0] = v28;


//YieldOp at loc("floyd-warshall.mlir":62:9)
assign tloop_in17 = tloop17delay[3];

//TerminatorOp

//} Loop17
//printTimeOffset
reg tfinish17delay[0:0] = '{default:0} ;
always@(*) tfinish17delay[0] <= tfinish17;
generate
genvar i30;

for(i30 = 1; i30<= 0; i30= i30 + 1) begin
always@(posedge clk) begin
tfinish17delay[i30] <= tfinish17delay[i30-1];
end
end
endgenerate


//YieldOp at loc("floyd-warshall.mlir":64:7)
assign tloop_in15 = tfinish17;

//TerminatorOp

//} Loop15
//printTimeOffset
reg tfinish15delay[0:0] = '{default:0} ;
always@(*) tfinish15delay[0] <= tfinish15;
generate
genvar i31;

for(i31 = 1; i31<= 0; i31= i31 + 1) begin
always@(posedge clk) begin
tfinish15delay[i31] <= tfinish15delay[i31-1];
end
end
endgenerate


//YieldOp at loc("floyd-warshall.mlir":66:5)
assign tloop_in13 = tfinish15;

//TerminatorOp

//} Loop13
//printTimeOffset
reg tfinish13delay[0:0] = '{default:0} ;
always@(*) tfinish13delay[0] <= tfinish13;
generate
genvar i32;

for(i32 = 1; i32<= 0; i32= i32 + 1) begin
always@(posedge clk) begin
tfinish13delay[i32] <= tfinish13delay[i32-1];
end
end
endgenerate


//ReturnOp at loc("floyd-warshall.mlir":68:3)
endmodule
