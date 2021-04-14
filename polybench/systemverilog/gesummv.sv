`default_nettype none
module gesummv(
//Outputs.

//Inputs.

//IntegerType.
input wire[31:0] v0,
//IntegerType.
input wire[31:0] v1,
//MemrefType : port = w.
output reg[2:0] v2_addr,
output wire v2_wr_en,
output reg[31:0] v2_wr_data,
//MemrefType : port = r.
output reg[5:0] v3_addr,
output wire v3_rd_en,
input wire[31:0] v3_rd_data,
//MemrefType : port = r.
output reg[5:0] v4_addr,
output wire v4_rd_en,
input wire[31:0] v4_rd_data,
//MemrefType : port = r.
output reg[2:0] v5_addr,
output wire v5_rd_en,
input wire[31:0] v5_rd_data,
//MemrefType : port = w.
output reg[2:0] v6_addr,
output wire v6_wr_en,
output reg[31:0] v6_wr_data,
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

wire v2_addr_valid  [0:0] ;
wire [2:0] v2_addr_input  [0:0];
 wire v2_addr_valid_if [0:0]   ;
wire [2:0] v2_addr_input_if [0:0]  ;
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


wire v3_addr_valid  [0:0] ;
wire [5:0] v3_addr_input  [0:0];
 wire v3_addr_valid_if [0:0]   ;
wire [5:0] v3_addr_input_if [0:0]  ;
 always@(*) begin
if(v3_addr_valid[0] )
v3_addr = v3_addr_input[0];
else
 v3_addr = 'x;
end
assign v3_addr_valid_if[0]  = v3_addr_valid  [0];
assign v3_addr_input_if[0]  = v3_addr_input  [0];

wire [0:0] v3_rd_en_input ;
wire  v3_rd_en_input_if [0:0] ;
assign v3_rd_en  =| v3_rd_en_input ;
assign v3_rd_en_input_if[0] = v3_rd_en_input  [0];


wire v4_addr_valid  [0:0] ;
wire [5:0] v4_addr_input  [0:0];
 wire v4_addr_valid_if [0:0]   ;
wire [5:0] v4_addr_input_if [0:0]  ;
 always@(*) begin
if(v4_addr_valid[0] )
v4_addr = v4_addr_input[0];
else
 v4_addr = 'x;
end
assign v4_addr_valid_if[0]  = v4_addr_valid  [0];
assign v4_addr_input_if[0]  = v4_addr_input  [0];

wire [0:0] v4_rd_en_input ;
wire  v4_rd_en_input_if [0:0] ;
assign v4_rd_en  =| v4_rd_en_input ;
assign v4_rd_en_input_if[0] = v4_rd_en_input  [0];


wire v5_addr_valid  [0:0] ;
wire [2:0] v5_addr_input  [0:0];
 wire v5_addr_valid_if [0:0]   ;
wire [2:0] v5_addr_input_if [0:0]  ;
 always@(*) begin
if(v5_addr_valid[0] )
v5_addr = v5_addr_input[0];
else
 v5_addr = 'x;
end
assign v5_addr_valid_if[0]  = v5_addr_valid  [0];
assign v5_addr_input_if[0]  = v5_addr_input  [0];

wire [0:0] v5_rd_en_input ;
wire  v5_rd_en_input_if [0:0] ;
assign v5_rd_en  =| v5_rd_en_input ;
assign v5_rd_en_input_if[0] = v5_rd_en_input  [0];


wire v6_addr_valid  [0:0] ;
wire [2:0] v6_addr_input  [0:0];
 wire v6_addr_valid_if [0:0]   ;
wire [2:0] v6_addr_input_if [0:0]  ;
 always@(*) begin
if(v6_addr_valid[0] )
v6_addr = v6_addr_input[0];
else
 v6_addr = 'x;
end
assign v6_addr_valid_if[0]  = v6_addr_valid  [0];
assign v6_addr_input_if[0]  = v6_addr_input  [0];

wire [0:0] v6_wr_en_input ;
wire  v6_wr_en_input_if [0:0] ;
assign v6_wr_en  =| v6_wr_en_input ;
assign v6_wr_en_input_if[0] = v6_wr_en_input  [0];
wire v6_wr_data_valid  [0:0] ;
wire [31:0] v6_wr_data_input  [0:0];
 wire v6_wr_data_valid_if [0:0]   ;
wire [31:0] v6_wr_data_input_if [0:0]  ;
 always@(*) begin
if(v6_wr_data_valid[0] )
v6_wr_data = v6_wr_data_input[0];
else
 v6_wr_data = 'x;
end
assign v6_wr_data_valid_if[0]  = v6_wr_data_valid  [0];
assign v6_wr_data_input_if[0]  = v6_wr_data_input  [0];


//printTimeOffset
reg tstartdelay[0:0] = '{default:0} ;
always@(*) tstartdelay[0] <= tstart;
generate
genvar i7;

for(i7 = 1; i7<= 0; i7= i7 + 1) begin
always@(posedge clk) begin
tstartdelay[i7] <= tstartdelay[i7-1];
end
end
endgenerate


//AllocaOp at loc("gesummv.mlir":16:21)
//strMemrefInstDecl
reg[3:0] v8_addr[1:0];
wire v8_rd_en[1:0];
logic[31:0] v8_rd_data[1:0];
//strMemrefSelDecl
//Unused memref v8.

//strMemrefInstDecl
reg[3:0] v9_addr[1:0];
 wire v9_wr_en[1:0];
reg[31:0] v9_wr_data[1:0];
//strMemrefSelDecl
//Unused memref v9.


 //Instantiate Memory.
for(genvar i0= 0; i0<2;i0+=1) begin
bram_tdp_rf_rf#(.SIZE(16), .WIDTH(32))  bram_inst0(
.clka(clk),
.clkb(clk),
.ena(v8_rd_en[i0]),
.enb(v9_wr_en[i0]),
.wea(0),
.web(v9_wr_en[i0]),
.addra(v8_addr[i0]),
.addrb(v9_addr[i0]),
.dia(0),
.dib(v9_wr_data[i0]),
.doa(v8_rd_data[i0]),
.dob(/*ignored*/)
);
end

//AllocaOp at loc("gesummv.mlir":19:21)
//strMemrefInstDecl
wire v10_rd_en[1:0][1:0];
logic[31:0] v10_rd_data[1:0][1:0];
//strMemrefSelDecl
//Unused memref v10.

//strMemrefInstDecl
 wire v11_wr_en[1:0][1:0];
reg[31:0] v11_wr_data[1:0][1:0];
//strMemrefSelDecl
//Unused memref v11.


 //Instantiate Memory.
for(genvar i0= 0; i0<2;i0+=1) begin
for(genvar i1= 0; i1<2;i1+=1) begin
always@(posedge clk) begin
  if(v11_wr_en[i0][i1]) v10_rd_data[i0][i1] <= v11_wr_data[i0][i1];
end
end
end

//ConstantOp at loc("gesummv.mlir":22:8)
//constant v12 = 1'd0;

//ConstantOp at loc("gesummv.mlir":23:8)
//constant v13 = 1'd1;

//ConstantOp at loc("gesummv.mlir":24:8)
//constant [2:0] v14 = 3'd4;

//ConstantOp at loc("gesummv.mlir":25:8)
//constant [2:0] v15 = 3'd5;

//ConstantOp at loc("gesummv.mlir":26:8)
//constant [2:0] v16 = 3'd6;

//ConstantOp at loc("gesummv.mlir":27:8)
//constant [3:0] v17 = 4'd8;

//ConstantOp at loc("gesummv.mlir":28:8)
//constant [3:0] v18 = 4'd9;

//ForOp at loc("gesummv.mlir":31:3)

//{ Loop19

reg[31:0] idx19 ;
reg[3:0] ub19 ;
reg[0:0] step19 ;
wire tloop_in19;
reg tloop19;
reg tfinish19;
always@(posedge clk) begin
 if(/*tstart=*/ tstart) begin
   idx19 <= /*v12=*/ 1'd0; //lower bound.
   step19 <= /*v13=*/ 1'd1;
   ub19 <= /*v17=*/ 4'd8;
   tloop19 <= (/*v17=*/ 4'd8 > /*v12=*/ 1'd0);
   tfinish19 <=!(/*v17=*/ 4'd8 > /*v12=*/ 1'd0);
 end
 else if (tloop_in19) begin
   idx19 <= idx19 + step19; //increment
   tloop19 <= (idx19 + step19) < ub19;
   tfinish19 <= !((idx19 + step19) < ub19);
 end
 else begin
   tloop19 <= 1'b0;
   tfinish19 <= 1'b0;
 end
end
//Loop19 body
//printTimeOffset
reg tloop19delay[0:0] = '{default:0} ;
always@(*) tloop19delay[0] <= tloop19;
generate
genvar i20;

for(i20 = 1; i20<= 0; i20= i20 + 1) begin
always@(posedge clk) begin
tloop19delay[i20] <= tloop19delay[i20-1];
end
end
endgenerate


//AllocaOp at loc("gesummv.mlir":33:27)
//strMemrefInstDecl
wire v21_rd_en[0:0];
logic[31:0] v21_rd_data[0:0];
//strMemrefSelDecl
wire [1:0] v21_rd_en_input [0:0];
wire  v21_rd_en_input_if [1:0] [0:0];
generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
assign v21_rd_en [i0] =| v21_rd_en_input [i0];
assign v21_rd_en_input_if[0][i0] = v21_rd_en_input [i0] [0];
assign v21_rd_en_input_if[1][i0] = v21_rd_en_input [i0] [1];
end
endgenerate


//strMemrefInstDecl
 wire v22_wr_en[0:0];
reg[31:0] v22_wr_data[0:0];
//strMemrefSelDecl
wire [1:0] v22_wr_en_input [0:0];
wire  v22_wr_en_input_if [1:0] [0:0];
generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
assign v22_wr_en [i0] =| v22_wr_en_input [i0];
assign v22_wr_en_input_if[0][i0] = v22_wr_en_input [i0] [0];
assign v22_wr_en_input_if[1][i0] = v22_wr_en_input [i0] [1];
end
endgenerate
wire v22_wr_data_valid [0:0] [1:0] ;
wire [31:0] v22_wr_data_input [0:0] [1:0];
 wire v22_wr_data_valid_if [1:0] [0:0]  ;
wire [31:0] v22_wr_data_input_if [1:0] [0:0] ;
 generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
always@(*) begin
if(v22_wr_data_valid[i0][0] )
v22_wr_data[i0] = v22_wr_data_input[i0][0];
else if (v22_wr_data_valid[i0][1])
v22_wr_data[i0] = v22_wr_data_input[i0][1];
else
 v22_wr_data[i0] = 'x;
end
assign v22_wr_data_valid_if[0] [i0] = v22_wr_data_valid [i0] [0];
assign v22_wr_data_input_if[0] [i0] = v22_wr_data_input [i0] [0];
assign v22_wr_data_valid_if[1] [i0] = v22_wr_data_valid [i0] [1];
assign v22_wr_data_input_if[1] [i0] = v22_wr_data_input [i0] [1];
end
endgenerate



 //Instantiate Memory.
for(genvar i0= 0; i0<1;i0+=1) begin
always@(posedge clk) begin
  if(v22_wr_en[i0]) v21_rd_data[i0] <= v22_wr_data[i0];
end
end

//AllocaOp at loc("gesummv.mlir":35:23)
//strMemrefInstDecl
wire v23_rd_en[0:0];
logic[31:0] v23_rd_data[0:0];
//strMemrefSelDecl
wire [1:0] v23_rd_en_input [0:0];
wire  v23_rd_en_input_if [1:0] [0:0];
generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
assign v23_rd_en [i0] =| v23_rd_en_input [i0];
assign v23_rd_en_input_if[0][i0] = v23_rd_en_input [i0] [0];
assign v23_rd_en_input_if[1][i0] = v23_rd_en_input [i0] [1];
end
endgenerate


//strMemrefInstDecl
 wire v24_wr_en[0:0];
reg[31:0] v24_wr_data[0:0];
//strMemrefSelDecl
wire [1:0] v24_wr_en_input [0:0];
wire  v24_wr_en_input_if [1:0] [0:0];
generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
assign v24_wr_en [i0] =| v24_wr_en_input [i0];
assign v24_wr_en_input_if[0][i0] = v24_wr_en_input [i0] [0];
assign v24_wr_en_input_if[1][i0] = v24_wr_en_input [i0] [1];
end
endgenerate
wire v24_wr_data_valid [0:0] [1:0] ;
wire [31:0] v24_wr_data_input [0:0] [1:0];
 wire v24_wr_data_valid_if [1:0] [0:0]  ;
wire [31:0] v24_wr_data_input_if [1:0] [0:0] ;
 generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
always@(*) begin
if(v24_wr_data_valid[i0][0] )
v24_wr_data[i0] = v24_wr_data_input[i0][0];
else if (v24_wr_data_valid[i0][1])
v24_wr_data[i0] = v24_wr_data_input[i0][1];
else
 v24_wr_data[i0] = 'x;
end
assign v24_wr_data_valid_if[0] [i0] = v24_wr_data_valid [i0] [0];
assign v24_wr_data_input_if[0] [i0] = v24_wr_data_input [i0] [0];
assign v24_wr_data_valid_if[1] [i0] = v24_wr_data_valid [i0] [1];
assign v24_wr_data_input_if[1] [i0] = v24_wr_data_input [i0] [1];
end
endgenerate



 //Instantiate Memory.
for(genvar i0= 0; i0<1;i0+=1) begin
always@(posedge clk) begin
  if(v24_wr_en[i0]) v23_rd_data[i0] <= v24_wr_data[i0];
end
end

//StoreOp at loc("gesummv.mlir":38:5)
assign v22_wr_en_input[/*v12=*/ 0][0] = tloop19;
assign v22_wr_data_valid[/*v12=*/ 0][0] = tloop19;
assign v22_wr_data_input[/*v12=*/ 0][0] = /*v12=*/ 1'd0;


//StoreOp at loc("gesummv.mlir":40:5)
assign v24_wr_en_input[/*v12=*/ 0][0] = tloop19;
assign v24_wr_data_valid[/*v12=*/ 0][0] = tloop19;
assign v24_wr_data_input[/*v12=*/ 0][0] = /*v12=*/ 1'd0;


//ForOp at loc("gesummv.mlir":43:9)

//{ Loop25

reg[31:0] idx25 ;
reg[3:0] ub25 ;
reg[0:0] step25 ;
wire tloop_in25;
reg tloop25;
reg tfinish25;
always@(posedge clk) begin
 if(/*tstart=*/ tloop19) begin
   idx25 <= /*v12=*/ 1'd0; //lower bound.
   step25 <= /*v13=*/ 1'd1;
   ub25 <= /*v17=*/ 4'd8;
   tloop25 <= (/*v17=*/ 4'd8 > /*v12=*/ 1'd0);
   tfinish25 <=!(/*v17=*/ 4'd8 > /*v12=*/ 1'd0);
 end
 else if (tloop_in25) begin
   idx25 <= idx25 + step25; //increment
   tloop25 <= (idx25 + step25) < ub25;
   tfinish25 <= !((idx25 + step25) < ub25);
 end
 else begin
   tloop25 <= 1'b0;
   tfinish25 <= 1'b0;
 end
end
//Loop25 body
//printTimeOffset
reg tloop25delay[5:0] = '{default:0} ;
always@(*) tloop25delay[0] <= tloop25;
generate
genvar i26;

for(i26 = 1; i26<= 5; i26= i26 + 1) begin
always@(posedge clk) begin
tloop25delay[i26] <= tloop25delay[i26-1];
end
end
endgenerate


//LoadOp at loc("gesummv.mlir":47:18)
assign v3_addr_valid[0] = tloop25;
assign v3_addr_input[0] = {idx19[2:0], idx25[2:0]};
wire[31:0] v27 = v3_rd_data;
assign v3_rd_en_input[0] = tloop25;


//LoadOp at loc("gesummv.mlir":49:18)
assign v4_addr_valid[0] = tloop25;
assign v4_addr_input[0] = {idx19[2:0], idx25[2:0]};
wire[31:0] v28 = v4_rd_data;
assign v4_rd_en_input[0] = tloop25;


//LoadOp at loc("gesummv.mlir":51:16)
assign v5_addr_valid[0] = tloop25;
assign v5_addr_input[0] = {idx25[2:0]};
wire[31:0] v29 = v5_rd_data;
assign v5_rd_en_input[0] = tloop25;


//CallOp at loc("gesummv.mlir":54:15)
wire [31:0] v30;
i32Multiplier i32Multiplier31(v30,
v27,
v29,
tloop25delay[1],
clk
);

//LoadOp at loc("gesummv.mlir":56:19)
wire[31:0] v32 = v21_rd_data[/*v12=*/ 0];
assign v21_rd_en_input[/*v12=*/ 0][0] = tloop25delay[5];


//CallOp at loc("gesummv.mlir":58:21)
wire [31:0] v33;
i32Adder i32Adder34(v33,
v30,
v32,
tloop25delay[5],
clk
);

//StoreOp at loc("gesummv.mlir":60:9)
assign v22_wr_en_input[/*v12=*/ 0][1] = tloop25delay[5];
assign v22_wr_data_valid[/*v12=*/ 0][1] = tloop25delay[5];
assign v22_wr_data_input[/*v12=*/ 0][1] = v33;


//CallOp at loc("gesummv.mlir":63:15)
wire [31:0] v35;
i32Multiplier i32Multiplier36(v35,
v28,
v29,
tloop25delay[1],
clk
);

//LoadOp at loc("gesummv.mlir":65:14)
wire[31:0] v37 = v23_rd_data[/*v12=*/ 0];
assign v23_rd_en_input[/*v12=*/ 0][0] = tloop25delay[5];


//CallOp at loc("gesummv.mlir":67:19)
wire [31:0] v38;
i32Adder i32Adder39(v38,
v30,
v37,
tloop25delay[5],
clk
);

//StoreOp at loc("gesummv.mlir":69:9)
assign v24_wr_en_input[/*v12=*/ 0][1] = tloop25delay[5];
assign v24_wr_data_valid[/*v12=*/ 0][1] = tloop25delay[5];
assign v24_wr_data_input[/*v12=*/ 0][1] = v38;


//YieldOp at loc("gesummv.mlir":71:9)
assign tloop_in25 = tloop25;

//TerminatorOp

//} Loop25
//printTimeOffset
reg tfinish25delay[9:0] = '{default:0} ;
always@(*) tfinish25delay[0] <= tfinish25;
generate
genvar i40;

for(i40 = 1; i40<= 9; i40= i40 + 1) begin
always@(posedge clk) begin
tfinish25delay[i40] <= tfinish25delay[i40-1];
end
end
endgenerate


//LoadOp at loc("gesummv.mlir":73:15)
wire[31:0] v41 = v21_rd_data[/*v12=*/ 0];
assign v21_rd_en_input[/*v12=*/ 0][1] = tfinish25delay[5];


//StoreOp at loc("gesummv.mlir":75:5)
assign v2_addr_valid[0] = tfinish25delay[5];
assign v2_addr_input[0] = {idx19[2:0]};
assign v2_wr_en_input[0] = tfinish25delay[5];
assign v2_wr_data_valid[0] = tfinish25delay[5];
assign v2_wr_data_input[0] = v41;


//LoadOp at loc("gesummv.mlir":77:10)
wire[31:0] v42 = v23_rd_data[/*v12=*/ 0];
assign v23_rd_en_input[/*v12=*/ 0][1] = tfinish25delay[5];


//CallOp at loc("gesummv.mlir":79:18)
wire [31:0] v43;
i32Multiplier i32Multiplier44(v43,
v0,
v41,
tfinish25delay[5],
clk
);

//CallOp at loc("gesummv.mlir":81:15)
wire [31:0] v45;
i32Multiplier i32Multiplier46(v45,
v1,
v42,
tfinish25delay[5],
clk
);

//CallOp at loc("gesummv.mlir":83:15)
wire [31:0] v47;
i32Adder i32Adder48(v47,
v43,
v45,
tfinish25delay[9],
clk
);

//DelayOp at loc("gesummv.mlir":86:11)
reg[31:0]shiftreg50[/*v18=*/ 9:0] = '{default:0};
always@(*) shiftreg50[0] <= idx19;
always@(posedge clk) shiftreg50[/*v18=*/ 9:1] <= shiftreg50[/*v18=*/ 8:0];
wire [31:0] v49 = shiftreg50[/*v18=*/ 9];

//StoreOp at loc("gesummv.mlir":87:5)
assign v6_addr_valid[0] = tfinish25delay[9];
assign v6_addr_input[0] = {v49[2:0]};
assign v6_wr_en_input[0] = tfinish25delay[9];
assign v6_wr_data_valid[0] = tfinish25delay[9];
assign v6_wr_data_input[0] = v47;


//YieldOp at loc("gesummv.mlir":90:5)
assign tloop_in19 = tfinish25delay[4];

//TerminatorOp

//} Loop19
//printTimeOffset
reg tfinish19delay[0:0] = '{default:0} ;
always@(*) tfinish19delay[0] <= tfinish19;
generate
genvar i51;

for(i51 = 1; i51<= 0; i51= i51 + 1) begin
always@(posedge clk) begin
tfinish19delay[i51] <= tfinish19delay[i51-1];
end
end
endgenerate


//ReturnOp at loc("gesummv.mlir":93:3)
endmodule
