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


//AllocaOp at loc("polybench/gesummv.mlir":16:21)
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

//AllocaOp at loc("polybench/gesummv.mlir":19:21)
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

//ConstantOp at loc("polybench/gesummv.mlir":22:8)
//constant v12 = 1'd0;

//ConstantOp at loc("polybench/gesummv.mlir":23:8)
//constant v13 = 1'd1;

//ConstantOp at loc("polybench/gesummv.mlir":24:8)
//constant [2:0] v14 = 3'd4;

//ConstantOp at loc("polybench/gesummv.mlir":25:8)
//constant [2:0] v15 = 3'd5;

//ConstantOp at loc("polybench/gesummv.mlir":26:8)
//constant [3:0] v16 = 4'd8;

//ForOp at loc("polybench/gesummv.mlir":29:3)

//{ Loop17

reg[31:0] idx17 ;
reg[3:0] ub17 ;
reg[0:0] step17 ;
wire tloop_in17;
reg tloop17;
reg tfinish17;
always@(posedge clk) begin
 if(/*tstart=*/ tstart) begin
   idx17 <= /*v12=*/ 1'd0; //lower bound.
   step17 <= /*v13=*/ 1'd1;
   ub17 <= /*v16=*/ 4'd8;
   tloop17 <= (/*v16=*/ 4'd8 > /*v12=*/ 1'd0);
   tfinish17 <=!(/*v16=*/ 4'd8 > /*v12=*/ 1'd0);
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
reg tloop17delay[0:0] = '{default:0} ;
always@(*) tloop17delay[0] <= tloop17;
generate
genvar i18;

for(i18 = 1; i18<= 0; i18= i18 + 1) begin
always@(posedge clk) begin
tloop17delay[i18] <= tloop17delay[i18-1];
end
end
endgenerate


//AllocaOp at loc("polybench/gesummv.mlir":31:27)
//strMemrefInstDecl
wire v19_rd_en[0:0];
logic[31:0] v19_rd_data[0:0];
//strMemrefSelDecl
wire [1:0] v19_rd_en_input [0:0];
wire  v19_rd_en_input_if [1:0] [0:0];
generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
assign v19_rd_en [i0] =| v19_rd_en_input [i0];
assign v19_rd_en_input_if[0][i0] = v19_rd_en_input [i0] [0];
assign v19_rd_en_input_if[1][i0] = v19_rd_en_input [i0] [1];
end
endgenerate


//strMemrefInstDecl
 wire v20_wr_en[0:0];
reg[31:0] v20_wr_data[0:0];
//strMemrefSelDecl
wire [1:0] v20_wr_en_input [0:0];
wire  v20_wr_en_input_if [1:0] [0:0];
generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
assign v20_wr_en [i0] =| v20_wr_en_input [i0];
assign v20_wr_en_input_if[0][i0] = v20_wr_en_input [i0] [0];
assign v20_wr_en_input_if[1][i0] = v20_wr_en_input [i0] [1];
end
endgenerate
wire v20_wr_data_valid [0:0] [1:0] ;
wire [31:0] v20_wr_data_input [0:0] [1:0];
 wire v20_wr_data_valid_if [1:0] [0:0]  ;
wire [31:0] v20_wr_data_input_if [1:0] [0:0] ;
 generate
for(genvar i0 = 0; i0 < 1;i0=i0 + 1) begin
always@(*) begin
if(v20_wr_data_valid[i0][0] )
v20_wr_data[i0] = v20_wr_data_input[i0][0];
else if (v20_wr_data_valid[i0][1])
v20_wr_data[i0] = v20_wr_data_input[i0][1];
else
 v20_wr_data[i0] = 'x;
end
assign v20_wr_data_valid_if[0] [i0] = v20_wr_data_valid [i0] [0];
assign v20_wr_data_input_if[0] [i0] = v20_wr_data_input [i0] [0];
assign v20_wr_data_valid_if[1] [i0] = v20_wr_data_valid [i0] [1];
assign v20_wr_data_input_if[1] [i0] = v20_wr_data_input [i0] [1];
end
endgenerate



 //Instantiate Memory.
for(genvar i0= 0; i0<1;i0+=1) begin
always@(posedge clk) begin
  if(v20_wr_en[i0]) v19_rd_data[i0] <= v20_wr_data[i0];
end
end

//AllocaOp at loc("polybench/gesummv.mlir":33:23)
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

//StoreOp at loc("polybench/gesummv.mlir":36:5)
assign v20_wr_en_input[/*v12=*/ 0][0] = tloop17;
assign v20_wr_data_valid[/*v12=*/ 0][0] = tloop17;
assign v20_wr_data_input[/*v12=*/ 0][0] = /*v12=*/ 1'd0;


//StoreOp at loc("polybench/gesummv.mlir":38:5)
assign v22_wr_en_input[/*v12=*/ 0][0] = tloop17;
assign v22_wr_data_valid[/*v12=*/ 0][0] = tloop17;
assign v22_wr_data_input[/*v12=*/ 0][0] = /*v12=*/ 1'd0;


//ForOp at loc("polybench/gesummv.mlir":41:9)

//{ Loop23

reg[31:0] idx23 ;
reg[3:0] ub23 ;
reg[0:0] step23 ;
wire tloop_in23;
reg tloop23;
reg tfinish23;
always@(posedge clk) begin
 if(/*tstart=*/ tloop17) begin
   idx23 <= /*v12=*/ 1'd0; //lower bound.
   step23 <= /*v13=*/ 1'd1;
   ub23 <= /*v16=*/ 4'd8;
   tloop23 <= (/*v16=*/ 4'd8 > /*v12=*/ 1'd0);
   tfinish23 <=!(/*v16=*/ 4'd8 > /*v12=*/ 1'd0);
 end
 else if (tloop_in23) begin
   idx23 <= idx23 + step23; //increment
   tloop23 <= (idx23 + step23) < ub23;
   tfinish23 <= !((idx23 + step23) < ub23);
 end
 else begin
   tloop23 <= 1'b0;
   tfinish23 <= 1'b0;
 end
end
//Loop23 body
//printTimeOffset
reg tloop23delay[5:0] = '{default:0} ;
always@(*) tloop23delay[0] <= tloop23;
generate
genvar i24;

for(i24 = 1; i24<= 5; i24= i24 + 1) begin
always@(posedge clk) begin
tloop23delay[i24] <= tloop23delay[i24-1];
end
end
endgenerate


//LoadOp at loc("polybench/gesummv.mlir":45:18)
assign v3_addr_valid[0] = tloop23;
assign v3_addr_input[0] = {idx17[2:0], idx23[2:0]};
wire[31:0] v25 = v3_rd_data;
assign v3_rd_en_input[0] = tloop23;


//LoadOp at loc("polybench/gesummv.mlir":47:18)
assign v4_addr_valid[0] = tloop23;
assign v4_addr_input[0] = {idx17[2:0], idx23[2:0]};
wire[31:0] v26 = v4_rd_data;
assign v4_rd_en_input[0] = tloop23;


//LoadOp at loc("polybench/gesummv.mlir":49:16)
assign v5_addr_valid[0] = tloop23;
assign v5_addr_input[0] = {idx23[2:0]};
wire[31:0] v27 = v5_rd_data;
assign v5_rd_en_input[0] = tloop23;


//CallOp at loc("polybench/gesummv.mlir":52:15)
wire [31:0] v28;
multInt32 multInt3229(v28,
v25,
v27,
tloop23delay[1],
clk
);

//LoadOp at loc("polybench/gesummv.mlir":54:16)
wire[31:0] v30 = v19_rd_data[/*v12=*/ 0];
assign v19_rd_en_input[/*v12=*/ 0][0] = tloop23delay[5];


//CallOp at loc("polybench/gesummv.mlir":56:21)
wire [31:0] v31;
addInt32 addInt3232(v31,
v28,
v30,
tloop23delay[5],
clk
);

//StoreOp at loc("polybench/gesummv.mlir":58:9)
assign v20_wr_en_input[/*v12=*/ 0][1] = tloop23delay[5];
assign v20_wr_data_valid[/*v12=*/ 0][1] = tloop23delay[5];
assign v20_wr_data_input[/*v12=*/ 0][1] = v31;


//CallOp at loc("polybench/gesummv.mlir":61:15)
wire [31:0] v33;
multInt32 multInt3234(v33,
v26,
v27,
tloop23delay[1],
clk
);

//LoadOp at loc("polybench/gesummv.mlir":63:14)
wire[31:0] v35 = v21_rd_data[/*v12=*/ 0];
assign v21_rd_en_input[/*v12=*/ 0][0] = tloop23delay[5];


//CallOp at loc("polybench/gesummv.mlir":65:19)
wire [31:0] v36;
addInt32 addInt3237(v36,
v28,
v35,
tloop23delay[5],
clk
);

//StoreOp at loc("polybench/gesummv.mlir":67:9)
assign v22_wr_en_input[/*v12=*/ 0][1] = tloop23delay[5];
assign v22_wr_data_valid[/*v12=*/ 0][1] = tloop23delay[5];
assign v22_wr_data_input[/*v12=*/ 0][1] = v36;


//YieldOp at loc("polybench/gesummv.mlir":69:9)
assign tloop_in23 = tloop23;

//TerminatorOp

//} Loop23
//printTimeOffset
reg tfinish23delay[5:0] = '{default:0} ;
always@(*) tfinish23delay[0] <= tfinish23;
generate
genvar i38;

for(i38 = 1; i38<= 5; i38= i38 + 1) begin
always@(posedge clk) begin
tfinish23delay[i38] <= tfinish23delay[i38-1];
end
end
endgenerate


//LoadOp at loc("polybench/gesummv.mlir":71:12)
wire[31:0] v39 = v19_rd_data[/*v12=*/ 0];
assign v19_rd_en_input[/*v12=*/ 0][1] = tfinish23delay[5];


//StoreOp at loc("polybench/gesummv.mlir":73:5)
assign v2_addr_valid[0] = tfinish23delay[5];
assign v2_addr_input[0] = {idx17[2:0]};
assign v2_wr_en_input[0] = tfinish23delay[5];
assign v2_wr_data_valid[0] = tfinish23delay[5];
assign v2_wr_data_input[0] = v39;


//LoadOp at loc("polybench/gesummv.mlir":75:10)
wire[31:0] v40 = v21_rd_data[/*v12=*/ 0];
assign v21_rd_en_input[/*v12=*/ 0][1] = tfinish23delay[5];


//CallOp at loc("polybench/gesummv.mlir":77:18)
wire [31:0] v41;
multInt32 multInt3242(v41,
v0,
v39,
tfinish23delay[1],
clk
);

//CallOp at loc("polybench/gesummv.mlir":79:15)
wire [31:0] v43;
multInt32 multInt3244(v43,
v1,
v40,
tfinish23delay[1],
clk
);

//CallOp at loc("polybench/gesummv.mlir":81:15)
wire [31:0] v45;
addIn32 addIn3246(v45,
v41,
v43,
tfinish23delay[5],
clk
);

//StoreOp at loc("polybench/gesummv.mlir":83:5)
assign v6_addr_valid[0] = tfinish23delay[5];
assign v6_addr_input[0] = {idx17[2:0]};
assign v6_wr_en_input[0] = tfinish23delay[5];
assign v6_wr_data_valid[0] = tfinish23delay[5];
assign v6_wr_data_input[0] = v45;


//YieldOp at loc("polybench/gesummv.mlir":86:5)
assign tloop_in17 = tfinish23;

//TerminatorOp

//} Loop17
//printTimeOffset
reg tfinish17delay[0:0] = '{default:0} ;
always@(*) tfinish17delay[0] <= tfinish17;
generate
genvar i47;

for(i47 = 1; i47<= 0; i47= i47 + 1) begin
always@(posedge clk) begin
tfinish17delay[i47] <= tfinish17delay[i47-1];
end
end
endgenerate


//ReturnOp at loc("polybench/gesummv.mlir":89:3)
endmodule
