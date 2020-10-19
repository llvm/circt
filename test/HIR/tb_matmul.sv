`default_nettype none
`include "../../out.sv"

module matmul_tb();
wire [3:0] v0_addr[15:0];
wire v0_rd_en[15:0];
reg[31:0] v0_rd_data[15:0];

wire v1_rd_en[15:0][15:0];
reg[31:0] v1_rd_data[15:0][15:0];

wire[3:0] v2_addr[15:0];
wire v2_wr_en[15:0];
wire[31:0] v2_wr_data[15:0];

reg tstart;
reg clk = 1'b0;

hirMatmulKernel hirMatmulKernel_inst(
  v0_addr    ,
  v0_rd_en   ,
  v0_rd_data ,
  v1_rd_en   ,
  v1_rd_data ,
  v2_addr    ,
  v2_wr_en   ,
  v2_wr_data ,
  tstart     ,
  clk
);


initial begin
  tstart <= 'd0;
  #9
  tstart <= 1'b1;
  #2
  tstart <= 1'b0;
end

generate for(genvar k=0;k<16;k++) begin
  always@(posedge clk) begin
    //v0_rd_data[k] <=  {32'd0,v0_addr[k]+1};
    v0_rd_data[k] <=  {32'd0,v0_addr[k]+k+1};
  end
  for(genvar j=0;j<16;j++) begin
    always@(posedge clk) begin
      v1_rd_data[k][j]<=  v1_rd_en[k][j]?(k+j):32'd255;
    end
  end
end
endgenerate

always begin
  #1
  clk <= !clk;
end
endmodule
