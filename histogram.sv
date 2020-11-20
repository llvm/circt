`default_nettype none
`include "helper.sv"
module histogram(
//Outputs.

//Inputs.

//MemrefType : port = r.
output reg[11:0] v0_addr,
output wire v0_rd_en,
input wire[7:0] v0_rd_data,
//MemrefType : port = w.
output reg[7:0] v1_addr,
output wire v1_wr_en,
output reg[31:0] v1_wr_data,
//TimeType.
input wire tstart,
//Clock.
input wire clk
);

wire v0_addr_valid  [0:0] ;
wire [11:0] v0_addr_input  [0:0];
 always@(*) begin
if(v0_addr_valid[0] )
v0_addr = v0_addr_input[0];
else
 v0_addr = 'x;
end

wire [0:0] v0_rd_en_input ;
assign v0_rd_en  =| v0_rd_en_input ;


wire v1_addr_valid  [0:0] ;
wire [7:0] v1_addr_input  [0:0];
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


//ConstantOp at loc("test/HIR/histogram.mlir":5:9)
//constant v4 = 1'd0;

//ConstantOp at loc("test/HIR/histogram.mlir":6:9)
//constant v5 = 1'd1;

//ConstantOp at loc("test/HIR/histogram.mlir":7:9)
//constant [1:0] v6 = 2'd2;

//ConstantOp at loc("test/HIR/histogram.mlir":8:9)
//constant [2:0] v7 = 3'd4;

//ConstantOp at loc("test/HIR/histogram.mlir":9:9)
//constant [4:0] v8 = 5'd16;

//ConstantOp at loc("test/HIR/histogram.mlir":10:9)
//constant [6:0] v9 = 7'd64;

//ConstantOp at loc("test/HIR/histogram.mlir":11:10)
//constant [8:0] v10 = 9'd256;

//AllocOp at loc("test/HIR/histogram.mlir":13:20)
//strMemrefInstDecl
reg[7:0] v11_addr;
wire v11_rd_en;
logic[31:0] v11_rd_data;
//strMemrefSelDecl
wire v11_addr_valid  [1:0] ;
wire [7:0] v11_addr_input  [1:0];
 always@(*) begin
if(v11_addr_valid[0] )
v11_addr = v11_addr_input[0];
else if (v11_addr_valid[1])
v11_addr = v11_addr_input[1];
else
 v11_addr = 'x;
end

wire [1:0] v11_rd_en_input ;
assign v11_rd_en  =| v11_rd_en_input ;


//strMemrefInstDecl
reg[7:0] v12_addr;
 wire v12_wr_en;
reg[31:0] v12_wr_data;
//strMemrefSelDecl
wire v12_addr_valid  [1:0] ;
wire [7:0] v12_addr_input  [1:0];
 always@(*) begin
if(v12_addr_valid[0] )
v12_addr = v12_addr_input[0];
else if (v12_addr_valid[1])
v12_addr = v12_addr_input[1];
else
 v12_addr = 'x;
end

wire [1:0] v12_wr_en_input ;
assign v12_wr_en  =| v12_wr_en_input ;
wire v12_wr_data_valid  [1:0] ;
wire [31:0] v12_wr_data_input  [1:0];
 always@(*) begin
if(v12_wr_data_valid[0] )
v12_wr_data = v12_wr_data_input[0];
else if (v12_wr_data_valid[1])
v12_wr_data = v12_wr_data_input[1];
else
 v12_wr_data = 'x;
end



 //Instantiate Memory.
bram_tdp_rf_rf#(.SIZE(256), .WIDTH(32))  bram_inst0(
.clka(clk),
.clkb(clk),
.ena(v11_rd_en),
.enb(v12_wr_en),
.wea(0),
.web(v12_wr_en),
.addra(v11_addr),
.addrb(v12_addr),
.dia(0),
.dib(v12_wr_data),
.doa(v11_rd_data),
.dob(/*ignored*/)
);

//ForOp at loc("test/HIR/histogram.mlir":15:9)

//{ Loop13

reg[31:0] idx13 ;
reg[8:0] ub13 ;
reg[0:0] step13 ;
wire tloop_in13;
reg tloop13;
reg tfinish13;
always@(posedge clk) begin
 if(/*tstart=*/ tstartdelay[0]) begin
   idx13 <= /*v4=*/ 1'd0; //lower bound.
   step13 <= /*v5=*/ 1'd1;
   ub13 <= /*v10=*/ 9'd256;
   tloop13 <= (/*v10=*/ 9'd256 > /*v4=*/ 1'd0);
   tfinish13 <=!(/*v10=*/ 9'd256 > /*v4=*/ 1'd0);
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


//YieldOp at loc("test/HIR/histogram.mlir":17:7)
assign tloop_in13 = tloop13delay[0];

//MemWriteOp at loc("test/HIR/histogram.mlir":18:7)
assign v12_addr_valid[0] = tloop13delay[0];
assign v12_addr_input[0] = {idx13[7:0]};
assign v12_wr_en_input[0] = tloop13delay[0];
assign v12_wr_data_valid[0] = tloop13delay[0];
assign v12_wr_data_input[0] = /*v4=*/ 1'd0;


//TerminatorOp

//} Loop13
//printTimeOffset
reg tfinish13delay[0:0] = '{default:0} ;
always@(*) tfinish13delay[0] <= tfinish13;
generate
genvar i15;

for(i15 = 1; i15<= 0; i15= i15 + 1) begin
always@(posedge clk) begin
tfinish13delay[i15] <= tfinish13delay[i15-1];
end
end
endgenerate


//ForOp at loc("test/HIR/histogram.mlir":21:7)

//{ Loop16

reg[31:0] idx16 ;
reg[4:0] ub16 ;
reg[0:0] step16 ;
wire tloop_in16;
reg tloop16;
reg tfinish16;
always@(posedge clk) begin
 if(/*tstart=*/ tfinish13delay[0]) begin
   idx16 <= /*v4=*/ 1'd0; //lower bound.
   step16 <= /*v5=*/ 1'd1;
   ub16 <= /*v8=*/ 5'd16;
   tloop16 <= (/*v8=*/ 5'd16 > /*v4=*/ 1'd0);
   tfinish16 <=!(/*v8=*/ 5'd16 > /*v4=*/ 1'd0);
 end
 else if (tloop_in16) begin
   idx16 <= idx16 + step16; //increment
   tloop16 <= (idx16 + step16) < ub16;
   tfinish16 <= !((idx16 + step16) < ub16);
 end
 else begin
   tloop16 <= 1'b0;
   tfinish16 <= 1'b0;
 end
end
//Loop16 body
//printTimeOffset
reg tloop16delay[0:0] = '{default:0} ;
always@(*) tloop16delay[0] <= tloop16;
generate
genvar i17;

for(i17 = 1; i17<= 0; i17= i17 + 1) begin
always@(posedge clk) begin
tloop16delay[i17] <= tloop16delay[i17-1];
end
end
endgenerate


//ForOp at loc("test/HIR/histogram.mlir":23:15)

//{ Loop18

reg[31:0] idx18 ;
reg[4:0] ub18 ;
reg[0:0] step18 ;
wire tloop_in18;
reg tloop18;
reg tfinish18;
always@(posedge clk) begin
 if(/*tstart=*/ tloop16delay[0]) begin
   idx18 <= /*v4=*/ 1'd0; //lower bound.
   step18 <= /*v5=*/ 1'd1;
   ub18 <= /*v8=*/ 5'd16;
   tloop18 <= (/*v8=*/ 5'd16 > /*v4=*/ 1'd0);
   tfinish18 <=!(/*v8=*/ 5'd16 > /*v4=*/ 1'd0);
 end
 else if (tloop_in18) begin
   idx18 <= idx18 + step18; //increment
   tloop18 <= (idx18 + step18) < ub18;
   tfinish18 <= !((idx18 + step18) < ub18);
 end
 else begin
   tloop18 <= 1'b0;
   tfinish18 <= 1'b0;
 end
end
//Loop18 body
//printTimeOffset
reg tloop18delay[2:0] = '{default:0} ;
always@(*) tloop18delay[0] <= tloop18;
generate
genvar i19;

for(i19 = 1; i19<= 2; i19= i19 + 1) begin
always@(posedge clk) begin
tloop18delay[i19] <= tloop18delay[i19-1];
end
end
endgenerate


//MemReadOp at loc("test/HIR/histogram.mlir":25:16)
assign v0_addr_valid[0] = tloop18delay[0];
assign v0_addr_input[0] = {idx16[5:0], idx18[5:0]};
wire[7:0] v20 = v0_rd_data;
assign v0_rd_en_input[0] = tloop18delay[0];


//MemReadOp at loc("test/HIR/histogram.mlir":27:20)
assign v11_addr_valid[0] = tloop18delay[1];
assign v11_addr_input[0] = {v20[7:0]};
wire[31:0] v21 = v11_rd_data;
assign v11_rd_en_input[0] = tloop18delay[1];


//AddOp at loc("test/HIR/histogram.mlir":29:24)
wire [31:0] v22 = v21 + /*v5=*/ 1'd1;

//MemWriteOp at loc("test/HIR/histogram.mlir":30:11)
assign v12_addr_valid[1] = tloop18delay[2];
assign v12_addr_input[1] = {v20[7:0]};
assign v12_wr_en_input[1] = tloop18delay[2];
assign v12_wr_data_valid[1] = tloop18delay[2];
assign v12_wr_data_input[1] = v22;


//YieldOp at loc("test/HIR/histogram.mlir":32:11)
assign tloop_in18 = tloop18delay[1];

//TerminatorOp

//} Loop18
//printTimeOffset
reg tfinish18delay[0:0] = '{default:0} ;
always@(*) tfinish18delay[0] <= tfinish18;
generate
genvar i23;

for(i23 = 1; i23<= 0; i23= i23 + 1) begin
always@(posedge clk) begin
tfinish18delay[i23] <= tfinish18delay[i23-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/histogram.mlir":34:7)
assign tloop_in16 = tfinish18delay[0];

//TerminatorOp

//} Loop16
//printTimeOffset
reg tfinish16delay[3:0] = '{default:0} ;
always@(*) tfinish16delay[0] <= tfinish16;
generate
genvar i24;

for(i24 = 1; i24<= 3; i24= i24 + 1) begin
always@(posedge clk) begin
tfinish16delay[i24] <= tfinish16delay[i24-1];
end
end
endgenerate


//ForOp at loc("test/HIR/histogram.mlir":36:3)

//{ Loop25

reg[31:0] idx25 ;
reg[8:0] ub25 ;
reg[0:0] step25 ;
wire tloop_in25;
reg tloop25;
reg tfinish25;
always@(posedge clk) begin
 if(/*tstart=*/ tfinish16delay[3]) begin
   idx25 <= /*v4=*/ 1'd0; //lower bound.
   step25 <= /*v5=*/ 1'd1;
   ub25 <= /*v10=*/ 9'd256;
   tloop25 <= (/*v10=*/ 9'd256 > /*v4=*/ 1'd0);
   tfinish25 <=!(/*v10=*/ 9'd256 > /*v4=*/ 1'd0);
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
reg tloop25delay[1:0] = '{default:0} ;
always@(*) tloop25delay[0] <= tloop25;
generate
genvar i26;

for(i26 = 1; i26<= 1; i26= i26 + 1) begin
always@(posedge clk) begin
tloop25delay[i26] <= tloop25delay[i26-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/histogram.mlir":38:7)
assign tloop_in25 = tloop25delay[0];

//MemReadOp at loc("test/HIR/histogram.mlir":39:16)
assign v11_addr_valid[1] = tloop25delay[0];
assign v11_addr_input[1] = {idx25[7:0]};
wire[31:0] v27 = v11_rd_data;
assign v11_rd_en_input[1] = tloop25delay[0];


//DelayOp at loc("test/HIR/histogram.mlir":41:13)
reg[31:0]shiftreg29[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg29[0] <= idx25;
always@(posedge clk) shiftreg29[/*v5=*/ 1:1] <= shiftreg29[/*v5=*/ 0:0];
wire [31:0] v28 = shiftreg29[/*v5=*/ 1];

//MemWriteOp at loc("test/HIR/histogram.mlir":42:7)
assign v1_addr_valid[0] = tloop25delay[1];
assign v1_addr_input[0] = {v28[7:0]};
assign v1_wr_en_input[0] = tloop25delay[1];
assign v1_wr_data_valid[0] = tloop25delay[1];
assign v1_wr_data_input[0] = v27;


//TerminatorOp

//} Loop25
//printTimeOffset
reg tfinish25delay[0:0] = '{default:0} ;
always@(*) tfinish25delay[0] <= tfinish25;
generate
genvar i30;

for(i30 = 1; i30<= 0; i30= i30 + 1) begin
always@(posedge clk) begin
tfinish25delay[i30] <= tfinish25delay[i30-1];
end
end
endgenerate


//ReturnOp at loc("test/HIR/histogram.mlir":45:3)
endmodule
