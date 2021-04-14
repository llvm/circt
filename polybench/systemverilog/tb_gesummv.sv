`default_nettype none
`define Alpha 1
`define Beta 1
module tb_gesummv();
 tb_gesummv_mlir  inst1();
 tb_gesummv_hls inst2();
endmodule

module tb_gesummv_mlir();

  wire tstart;
  wire clk;

  wire[31:0] alpha;
  wire[31:0] beta;

  reg[31:0] tmp_mem[7:0];
  wire [2:0] tmp_addr;
  wire       tmp_wr_en;
  wire[31:0] tmp_wr_data;

  reg[31:0] A_mem[63:0];
  wire [5:0] A_addr;
  wire       A_rd_en;
  wire[31:0] A_rd_data;

  reg[31:0] B_mem[63:0];
  wire [5:0] B_addr;
  wire       B_rd_en;
  wire[31:0] B_rd_data;

  reg[31:0] X_mem[7:0];
  wire [2:0] X_addr;
  wire       X_rd_en;
  wire[31:0] X_rd_data;

  reg[31:0] Y_mem[7:0];
  wire [2:0] Y_addr;
  wire       Y_wr_en;
  wire[31:0] Y_wr_data;
  assign alpha = `Alpha;
  assign beta = `Beta;

  initial begin
    for (int i = 0; i < 64; i = i + 1) begin
      A_mem[i] = i+1;
      B_mem[i] = i+1;
    end
    for (int i = 0; i < 8; i = i + 1) begin
      tmp_mem[i] = i+1;
      X_mem[i] = i+1;
      Y_mem[i] = i+1;
    end
  end 


  generate_clk gen_clk_inst(clk);
  generate_tstart gen_tstart_inst(tstart);

  memref_wr#(.WIDTH(32),.SIZE(8)) memref_wr_tmp(tmp_mem,tmp_wr_en, tmp_addr,tmp_wr_data,clk);
  memref_rd#(.WIDTH(32),.SIZE(64)) memref_rd_A(A_mem,A_rd_en, A_addr,/*valid*/ ,A_rd_data,clk);

  memref_rd#(.WIDTH(32),.SIZE(64)) memref_rd_B(B_mem,B_rd_en, B_addr,/*valid*/ ,B_rd_data,clk);
  memref_rd#(.WIDTH(32),.SIZE(8)) memref_rd_X(X_mem,X_rd_en, X_addr,/*valid*/ ,X_rd_data,clk);
  memref_wr#(.WIDTH(32),.SIZE(8)) memref_wr_Y(Y_mem,Y_wr_en, Y_addr,Y_wr_data,clk);

  gesummv dut_inst(
    .v0        (alpha) ,
    .v1        (beta) ,
    .v2_addr   (tmp_addr) ,
    .v2_wr_en  (tmp_wr_en) ,
    .v2_wr_data(tmp_wr_data) ,
    .v3_addr   (A_addr) ,
    .v3_rd_en  (A_rd_en) ,
    .v3_rd_data(A_rd_data) ,
    .v4_addr   (B_addr) ,
    .v4_rd_en  (B_rd_en) ,
    .v4_rd_data(B_rd_data) ,
    .v5_addr   (X_addr) ,
    .v5_rd_en  (X_rd_en) ,
    .v5_rd_data(X_rd_data) ,
    .v6_addr   (Y_addr) ,
    .v6_wr_en  (Y_wr_en) ,
    .v6_wr_data(Y_wr_data) ,
    .tstart    (tstart) ,
    .clk       (clk) 
  );
endmodule

module tb_gesummv_hls();

  wire tstart;
  wire clk;

  wire[31:0] alpha;
  wire[31:0] beta;

  reg[31:0] tmp_mem[7:0];
  wire [2:0] tmp_addr;
  wire       tmp_wr_en;
  wire[31:0] tmp_wr_data;

  reg[31:0] A_mem[63:0];
  wire [5:0] A_addr;
  wire       A_rd_en;
  wire[31:0] A_rd_data;

  reg[31:0] B_mem[63:0];
  wire [5:0] B_addr;
  wire       B_rd_en;
  wire[31:0] B_rd_data;

  reg[31:0] X_mem[7:0];
  wire [2:0] X_addr;
  wire       X_rd_en;
  wire[31:0] X_rd_data;

  reg[31:0] Y_mem[7:0];
  wire [2:0] Y_addr;
  wire       Y_wr_en;
  wire[31:0] Y_wr_data;
  assign alpha = `Alpha;
  assign beta = `Beta;

  initial begin
    for (int i = 0; i < 64; i = i + 1) begin
      A_mem[i] = i+1;
      B_mem[i] = i+1;
    end
    for (int i = 0; i < 8; i = i + 1) begin
      tmp_mem[i] = i+1;
      X_mem[i] = i+1;
      Y_mem[i] = i+1;
    end
  end 


  generate_clk gen_clk_inst(clk);
  generate_tstart gen_tstart_inst(tstart);

  memref_wr#(.WIDTH(32),.SIZE(8)) memref_wr_tmp(tmp_mem,tmp_wr_en, tmp_addr,tmp_wr_data,clk);
  memref_rd#(.WIDTH(32),.SIZE(64)) memref_rd_A(A_mem,A_rd_en, A_addr,/*valid*/ ,A_rd_data,clk);

  memref_rd#(.WIDTH(32),.SIZE(64)) memref_rd_B(B_mem,B_rd_en, B_addr,/*valid*/ ,B_rd_data,clk);
  memref_rd#(.WIDTH(32),.SIZE(8)) memref_rd_X(X_mem,X_rd_en, X_addr,/*valid*/ ,X_rd_data,clk);
  memref_wr#(.WIDTH(32),.SIZE(8)) memref_wr_Y(Y_mem,Y_wr_en, Y_addr,Y_wr_data,clk);


gesummv_hls_0 your_instance_name (
  .ap_clk(clk),                  // input wire ap_clk
  .ap_rst(tstart),                  // input wire ap_rst
  .alpha_V(alpha),                // input wire [31 : 0] alpha_V
  .beta_V(beta),                  // input wire [31 : 0] beta_V
  .tmp_V_ce0(),            // output wire tmp_V_ce0
  .tmp_V_we0(tmp_wr_en),            // output wire tmp_V_we0
  .tmp_V_address0(tmp_addr),  // output wire [2 : 0] tmp_V_address0
  .tmp_V_d0(tmp_wr_data),              // output wire [31 : 0] tmp_V_d0
  .A_V_ce0(A_rd_en),                // output wire A_V_ce0
  .A_V_address0(A_addr),      // output wire [5 : 0] A_V_address0
  .A_V_q0(A_rd_data),                  // input wire [31 : 0] A_V_q0
  .B_V_ce0(B_rd_en),                // output wire B_V_ce0
  .B_V_address0(B_addr),      // output wire [5 : 0] B_V_address0
  .B_V_q0(B_rd_data),                  // input wire [31 : 0] B_V_q0
  .x_V_ce0(X_rd_en),                // output wire x_V_ce0
  .x_V_address0(X_addr),      // output wire [2 : 0] x_V_address0
  .x_V_q0(X_rd_data),                  // input wire [31 : 0] x_V_q0
  .y_V_ce0(),                // output wire y_V_ce0
  .y_V_we0(Y_wr_en),                // output wire y_V_we0
  .y_V_address0(Y_addr),      // output wire [2 : 0] y_V_address0
  .y_V_d0(Y_wr_data)                  // output wire [31 : 0] y_V_d0
);

endmodule

module generate_clk(clk);
  output reg clk;// = 1'b0;

  initial begin
    clk <= 1'b0;
  end

  always begin
    #1
    clk <= !clk;
  end
endmodule

module generate_tstart(tstart);
  output reg tstart;
  initial begin
    tstart <= 1'b0;
    #3 tstart <= 1'b1;
    #2 tstart <= 1'b0;
  end
endmodule

module memref_wr(mem,wr_en,addr,din,clk); 
  parameter WIDTH = 32;
  parameter SIZE = 1;

  ref reg[WIDTH-1:0] mem[SIZE-1:0];
  input wire wr_en;
  input wire [$clog2(SIZE)-1:0] addr;
  input reg[WIDTH-1:0] din;
  input wire clk;

  always@(posedge clk) begin
    if(wr_en) begin
      mem[addr] <= din;
    end
  end
endmodule

module  memref_rd#(WIDTH=32,SIZE=8)( 

  ref reg[WIDTH-1:0] mem[SIZE-1:0],
  input wire rd_en,
  input wire [$clog2(SIZE)-1:0] addr,
  output reg dout_valid,
  output reg[WIDTH-1:0] dout,
  input wire clk);
  always@(posedge clk) begin
    if(rd_en) begin
      dout <= mem[addr];
      dout_valid <= 1'b1;
    end
    else begin
      dout <= 'X;
      dout_valid <= 1'b0;
    end
  end
endmodule
