`default_nettype none
`include "helper.sv"
module loop_fuse(
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
genvar i3;

for(i3 = 1; i3<= 2; i3= i3 + 1) begin
always@(posedge clk) begin
tstartdelay[i3] <= tstartdelay[i3-1];
end
end
endgenerate


//ConstantOp at loc("test/HIR/loop_fuse.mlir":5:9)
//constant v4 = 1'd0;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":6:9)
//constant v5 = 1'd1;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":7:9)
//constant [1:0] v6 = 2'd2;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":8:9)
//constant [1:0] v7 = 2'd3;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":9:9)
//constant [2:0] v8 = 3'd4;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":10:9)
//constant [2:0] v9 = 3'd5;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":11:9)
//constant [3:0] v10 = 4'd8;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":12:9)
//constant [4:0] v11 = 5'd16;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":13:9)
//constant [6:0] v12 = 7'd64;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":14:9)
//constant [6:0] v13 = 7'd65;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":15:9)
//constant [6:0] v14 = 7'd66;

//ConstantOp at loc("test/HIR/loop_fuse.mlir":16:9)
//constant [6:0] v15 = 7'd66;

//AllocOp at loc("test/HIR/loop_fuse.mlir":18:16)
//strMemrefInstDecl
wire v16_rd_en[1:0];
logic[31:0] v16_rd_data[1:0];
//strMemrefSelDecl
wire [0:0] v16_rd_en_input [1:0];
generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
assign v16_rd_en [i0] =| v16_rd_en_input [i0];
end
endgenerate


//strMemrefInstDecl
 wire v17_wr_en[1:0];
reg[31:0] v17_wr_data[1:0];
//strMemrefSelDecl
wire [1:0] v17_wr_en_input [1:0];
generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
assign v17_wr_en [i0] =| v17_wr_en_input [i0];
end
endgenerate
wire v17_wr_data_valid [1:0] [1:0] ;
wire [31:0] v17_wr_data_input [1:0] [1:0];
 generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
always@(*) begin
if(v17_wr_data_valid[i0][0] )
v17_wr_data[i0] = v17_wr_data_input[i0][0];
else if (v17_wr_data_valid[i0][1])
v17_wr_data[i0] = v17_wr_data_input[i0][1];
else
 v17_wr_data[i0] = 'x;
end
end
endgenerate



 //Instantiate Memory.
for(genvar i0= 0; i0<2;i0+=1) begin
always@(posedge clk) begin
  if(v17_wr_en[i0]) v16_rd_data[i0] <= v17_wr_data[i0];
end
end

//AllocOp at loc("test/HIR/loop_fuse.mlir":20:14)
//strMemrefInstDecl
reg[5:0] v18_addr;
wire v18_rd_en;
logic[31:0] v18_rd_data;
//strMemrefSelDecl
wire v18_addr_valid  [2:0] ;
wire [5:0] v18_addr_input  [2:0];
 always@(*) begin
if(v18_addr_valid[0] )
v18_addr = v18_addr_input[0];
else if (v18_addr_valid[1])
v18_addr = v18_addr_input[1];
else if (v18_addr_valid[2])
v18_addr = v18_addr_input[2];
else
 v18_addr = 'x;
end

wire [2:0] v18_rd_en_input ;
assign v18_rd_en  =| v18_rd_en_input ;


//strMemrefInstDecl
reg[5:0] v19_addr;
 wire v19_wr_en;
reg[31:0] v19_wr_data;
//strMemrefSelDecl
wire v19_addr_valid  [0:0] ;
wire [5:0] v19_addr_input  [0:0];
 always@(*) begin
if(v19_addr_valid[0] )
v19_addr = v19_addr_input[0];
else
 v19_addr = 'x;
end

wire [0:0] v19_wr_en_input ;
assign v19_wr_en  =| v19_wr_en_input ;
wire v19_wr_data_valid  [0:0] ;
wire [31:0] v19_wr_data_input  [0:0];
 always@(*) begin
if(v19_wr_data_valid[0] )
v19_wr_data = v19_wr_data_input[0];
else
 v19_wr_data = 'x;
end



 //Instantiate Memory.
bram_tdp_rf_rf#(.SIZE(64), .WIDTH(32))  bram_inst0(
.clka(clk),
.clkb(clk),
.ena(v18_rd_en),
.enb(v19_wr_en),
.wea(0),
.web(v19_wr_en),
.addra(v18_addr),
.addrb(v19_addr),
.dia(0),
.dib(v19_wr_data),
.doa(v18_rd_data),
.dob(/*ignored*/)
);

//MemReadOp at loc("test/HIR/loop_fuse.mlir":23:11)
assign v0_addr_valid[0] = tstartdelay[0];
assign v0_addr_input[0] = {/*v4=*/ 6'd0};
wire[31:0] v20 = v0_rd_data;
assign v0_rd_en_input[0] = tstartdelay[0];


//DelayOp at loc("test/HIR/loop_fuse.mlir":25:12)
reg[31:0]shiftreg22[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg22[0] <= v20;
always@(posedge clk) shiftreg22[/*v5=*/ 1:1] <= shiftreg22[/*v5=*/ 0:0];
wire [31:0] v21 = shiftreg22[/*v5=*/ 1];

//MemReadOp at loc("test/HIR/loop_fuse.mlir":27:11)
assign v0_addr_valid[1] = tstartdelay[1];
assign v0_addr_input[1] = {/*v5=*/ 6'd1};
wire[31:0] v23 = v0_rd_data;
assign v0_rd_en_input[1] = tstartdelay[1];


//MemWriteOp at loc("test/HIR/loop_fuse.mlir":30:3)
assign v17_wr_en_input[/*v4=*/ 0][0] = tstartdelay[2];
assign v17_wr_data_valid[/*v4=*/ 0][0] = tstartdelay[2];
assign v17_wr_data_input[/*v4=*/ 0][0] = v21;


//MemWriteOp at loc("test/HIR/loop_fuse.mlir":32:3)
assign v17_wr_en_input[/*v5=*/ 1][0] = tstartdelay[2];
assign v17_wr_data_valid[/*v5=*/ 1][0] = tstartdelay[2];
assign v17_wr_data_input[/*v5=*/ 1][0] = v23;


//ForOp at loc("test/HIR/loop_fuse.mlir":35:3)

//{ Loop24

reg[31:0] idx24 ;
reg[6:0] ub24 ;
reg[0:0] step24 ;
wire tloop_in24;
reg tloop24;
reg tfinish24;
always@(posedge clk) begin
 if(/*tstart=*/ tstartdelay[2]) begin
   idx24 <= /*v5=*/ 1'd1; //lower bound.
   step24 <= /*v5=*/ 1'd1;
   ub24 <= /*v12=*/ 7'd64;
   tloop24 <= (/*v12=*/ 7'd64 > /*v5=*/ 1'd1);
   tfinish24 <=!(/*v12=*/ 7'd64 > /*v5=*/ 1'd1);
 end
 else if (tloop_in24) begin
   idx24 <= idx24 + step24; //increment
   tloop24 <= (idx24 + step24) < ub24;
   tfinish24 <= !((idx24 + step24) < ub24);
 end
 else begin
   tloop24 <= 1'b0;
   tfinish24 <= 1'b0;
 end
end
//Loop24 body
//printTimeOffset
reg tloop24delay[2:0] = '{default:0} ;
always@(*) tloop24delay[0] <= tloop24;
generate
genvar i25;

for(i25 = 1; i25<= 2; i25= i25 + 1) begin
always@(posedge clk) begin
tloop24delay[i25] <= tloop24delay[i25-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/loop_fuse.mlir":37:7)
assign tloop_in24 = tloop24delay[0];

//MemReadOp at loc("test/HIR/loop_fuse.mlir":39:13)
wire[31:0] v26 = v16_rd_data[/*v4=*/ 0];
assign v16_rd_en_input[/*v4=*/ 0][0] = tloop24delay[1];


//MemReadOp at loc("test/HIR/loop_fuse.mlir":41:13)
wire[31:0] v27 = v16_rd_data[/*v5=*/ 1];
assign v16_rd_en_input[/*v5=*/ 1][0] = tloop24delay[1];


//AddOp at loc("test/HIR/loop_fuse.mlir":43:17)
wire [31:0] v28 = idx24 + /*v5=*/ 1'd1;

//MemReadOp at loc("test/HIR/loop_fuse.mlir":44:13)
assign v0_addr_valid[2] = tloop24delay[0];
assign v0_addr_input[2] = {v28[5:0]};
wire[31:0] v29 = v0_rd_data;
assign v0_rd_en_input[2] = tloop24delay[0];


//MemWriteOp at loc("test/HIR/loop_fuse.mlir":47:7)
assign v17_wr_en_input[/*v4=*/ 0][1] = tloop24delay[1];
assign v17_wr_data_valid[/*v4=*/ 0][1] = tloop24delay[1];
assign v17_wr_data_input[/*v4=*/ 0][1] = v27;


//MemWriteOp at loc("test/HIR/loop_fuse.mlir":49:7)
assign v17_wr_en_input[/*v5=*/ 1][1] = tloop24delay[1];
assign v17_wr_data_valid[/*v5=*/ 1][1] = tloop24delay[1];
assign v17_wr_data_input[/*v5=*/ 1][1] = v29;


//CallOp at loc("test/HIR/loop_fuse.mlir":52:13)
wire [31:0] v30;
weighted_sum weighted_sum31(v30,
v26,
v27,
tloop24delay[1],
clk
);

//DelayOp at loc("test/HIR/loop_fuse.mlir":54:13)
reg[31:0]shiftreg33[/*v6=*/ 2:0] = '{default:0};
always@(*) shiftreg33[0] <= idx24;
always@(posedge clk) shiftreg33[/*v6=*/ 2:1] <= shiftreg33[/*v6=*/ 1:0];
wire [31:0] v32 = shiftreg33[/*v6=*/ 2];

//MemWriteOp at loc("test/HIR/loop_fuse.mlir":55:7)
assign v19_addr_valid[0] = tloop24delay[2];
assign v19_addr_input[0] = {v32[5:0]};
assign v19_wr_en_input[0] = tloop24delay[2];
assign v19_wr_data_valid[0] = tloop24delay[2];
assign v19_wr_data_input[0] = v30;


//TerminatorOp

//} Loop24
//printTimeOffset
reg tfinish24delay[0:0] = '{default:0} ;
always@(*) tfinish24delay[0] <= tfinish24;
generate
genvar i34;

for(i34 = 1; i34<= 0; i34= i34 + 1) begin
always@(posedge clk) begin
tfinish24delay[i34] <= tfinish24delay[i34-1];
end
end
endgenerate


//AllocOp at loc("test/HIR/loop_fuse.mlir":59:16)
//strMemrefInstDecl
wire v35_rd_en[1:0];
logic[31:0] v35_rd_data[1:0];
//strMemrefSelDecl
wire [0:0] v35_rd_en_input [1:0];
generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
assign v35_rd_en [i0] =| v35_rd_en_input [i0];
end
endgenerate


//strMemrefInstDecl
 wire v36_wr_en[1:0];
reg[31:0] v36_wr_data[1:0];
//strMemrefSelDecl
wire [1:0] v36_wr_en_input [1:0];
generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
assign v36_wr_en [i0] =| v36_wr_en_input [i0];
end
endgenerate
wire v36_wr_data_valid [1:0] [1:0] ;
wire [31:0] v36_wr_data_input [1:0] [1:0];
 generate
for(genvar i0 = 0; i0 < 2;i0=i0 + 1) begin
always@(*) begin
if(v36_wr_data_valid[i0][0] )
v36_wr_data[i0] = v36_wr_data_input[i0][0];
else if (v36_wr_data_valid[i0][1])
v36_wr_data[i0] = v36_wr_data_input[i0][1];
else
 v36_wr_data[i0] = 'x;
end
end
endgenerate



 //Instantiate Memory.
for(genvar i0= 0; i0<2;i0+=1) begin
always@(posedge clk) begin
  if(v36_wr_en[i0]) v35_rd_data[i0] <= v36_wr_data[i0];
end
end

//DelayOp at loc("test/HIR/loop_fuse.mlir":62:9)
reg[0:0]shiftreg38[/*v10=*/ 8:0] = '{default:0};
always@(*) shiftreg38[0] <= tstart;
always@(posedge clk) shiftreg38[/*v10=*/ 8:1] <= shiftreg38[/*v10=*/ 7:0];
wire v37 = shiftreg38[/*v10=*/ 8];
//printTimeOffset
reg v37delay[2:0] = '{default:0} ;
always@(*) v37delay[0] <= v37;
generate
genvar i39;

for(i39 = 1; i39<= 2; i39= i39 + 1) begin
always@(posedge clk) begin
v37delay[i39] <= v37delay[i39-1];
end
end
endgenerate


//MemReadOp at loc("test/HIR/loop_fuse.mlir":63:11)
assign v18_addr_valid[0] = v37delay[0];
assign v18_addr_input[0] = {/*v5=*/ 6'd1};
wire[31:0] v40 = v18_rd_data;
assign v18_rd_en_input[0] = v37delay[0];


//DelayOp at loc("test/HIR/loop_fuse.mlir":66:12)
reg[31:0]shiftreg42[/*v5=*/ 1:0] = '{default:0};
always@(*) shiftreg42[0] <= v40;
always@(posedge clk) shiftreg42[/*v5=*/ 1:1] <= shiftreg42[/*v5=*/ 0:0];
wire [31:0] v41 = shiftreg42[/*v5=*/ 1];

//MemReadOp at loc("test/HIR/loop_fuse.mlir":67:11)
assign v18_addr_valid[1] = v37delay[1];
assign v18_addr_input[1] = {/*v6=*/ 6'd2};
wire[31:0] v43 = v18_rd_data;
assign v18_rd_en_input[1] = v37delay[1];


//MemWriteOp at loc("test/HIR/loop_fuse.mlir":70:3)
assign v36_wr_en_input[/*v4=*/ 0][0] = v37delay[2];
assign v36_wr_data_valid[/*v4=*/ 0][0] = v37delay[2];
assign v36_wr_data_input[/*v4=*/ 0][0] = v41;


//MemWriteOp at loc("test/HIR/loop_fuse.mlir":72:3)
assign v36_wr_en_input[/*v5=*/ 1][0] = v37delay[2];
assign v36_wr_data_valid[/*v5=*/ 1][0] = v37delay[2];
assign v36_wr_data_input[/*v5=*/ 1][0] = v43;


//ForOp at loc("test/HIR/loop_fuse.mlir":75:3)

//{ Loop44

reg[31:0] idx44 ;
reg[6:0] ub44 ;
reg[0:0] step44 ;
wire tloop_in44;
reg tloop44;
reg tfinish44;
always@(posedge clk) begin
 if(/*tstart=*/ v37delay[2]) begin
   idx44 <= /*v6=*/ 2'd2; //lower bound.
   step44 <= /*v5=*/ 1'd1;
   ub44 <= /*v12=*/ 7'd64;
   tloop44 <= (/*v12=*/ 7'd64 > /*v6=*/ 2'd2);
   tfinish44 <=!(/*v12=*/ 7'd64 > /*v6=*/ 2'd2);
 end
 else if (tloop_in44) begin
   idx44 <= idx44 + step44; //increment
   tloop44 <= (idx44 + step44) < ub44;
   tfinish44 <= !((idx44 + step44) < ub44);
 end
 else begin
   tloop44 <= 1'b0;
   tfinish44 <= 1'b0;
 end
end
//Loop44 body
//printTimeOffset
reg tloop44delay[2:0] = '{default:0} ;
always@(*) tloop44delay[0] <= tloop44;
generate
genvar i45;

for(i45 = 1; i45<= 2; i45= i45 + 1) begin
always@(posedge clk) begin
tloop44delay[i45] <= tloop44delay[i45-1];
end
end
endgenerate


//YieldOp at loc("test/HIR/loop_fuse.mlir":77:7)
assign tloop_in44 = tloop44delay[0];

//MemReadOp at loc("test/HIR/loop_fuse.mlir":79:13)
wire[31:0] v46 = v35_rd_data[/*v4=*/ 0];
assign v35_rd_en_input[/*v4=*/ 0][0] = tloop44delay[1];


//MemReadOp at loc("test/HIR/loop_fuse.mlir":81:13)
wire[31:0] v47 = v35_rd_data[/*v5=*/ 1];
assign v35_rd_en_input[/*v5=*/ 1][0] = tloop44delay[1];


//AddOp at loc("test/HIR/loop_fuse.mlir":83:17)
wire [31:0] v48 = idx44 + /*v5=*/ 1'd1;

//MemReadOp at loc("test/HIR/loop_fuse.mlir":84:13)
assign v18_addr_valid[2] = tloop44delay[0];
assign v18_addr_input[2] = {v48[5:0]};
wire[31:0] v49 = v18_rd_data;
assign v18_rd_en_input[2] = tloop44delay[0];


//MemWriteOp at loc("test/HIR/loop_fuse.mlir":87:7)
assign v36_wr_en_input[/*v4=*/ 0][1] = tloop44delay[1];
assign v36_wr_data_valid[/*v4=*/ 0][1] = tloop44delay[1];
assign v36_wr_data_input[/*v4=*/ 0][1] = v47;


//MemWriteOp at loc("test/HIR/loop_fuse.mlir":89:7)
assign v36_wr_en_input[/*v5=*/ 1][1] = tloop44delay[1];
assign v36_wr_data_valid[/*v5=*/ 1][1] = tloop44delay[1];
assign v36_wr_data_input[/*v5=*/ 1][1] = v49;


//CallOp at loc("test/HIR/loop_fuse.mlir":92:13)
wire [31:0] v50;
max max51(v50,
v46,
v47,
tloop44delay[1],
clk
);

//DelayOp at loc("test/HIR/loop_fuse.mlir":95:13)
reg[31:0]shiftreg53[/*v6=*/ 2:0] = '{default:0};
always@(*) shiftreg53[0] <= idx44;
always@(posedge clk) shiftreg53[/*v6=*/ 2:1] <= shiftreg53[/*v6=*/ 1:0];
wire [31:0] v52 = shiftreg53[/*v6=*/ 2];

//MemWriteOp at loc("test/HIR/loop_fuse.mlir":96:7)
assign v1_addr_valid[0] = tloop44delay[2];
assign v1_addr_input[0] = {v52[5:0]};
assign v1_wr_en_input[0] = tloop44delay[2];
assign v1_wr_data_valid[0] = tloop44delay[2];
assign v1_wr_data_input[0] = v50;


//TerminatorOp

//} Loop44
//printTimeOffset
reg tfinish44delay[0:0] = '{default:0} ;
always@(*) tfinish44delay[0] <= tfinish44;
generate
genvar i54;

for(i54 = 1; i54<= 0; i54= i54 + 1) begin
always@(posedge clk) begin
tfinish44delay[i54] <= tfinish44delay[i54-1];
end
end
endgenerate


//ReturnOp at loc("test/HIR/loop_fuse.mlir":100:3)
endmodule
