`default_nettype none
`include "../../matmul.sv"

module matmul_tb();
wire [7:0] v0_addr;
wire v0_rd_en;
reg[31:0] v0_rd_data;

wire[7:0] v1_addr;
wire v1_rd_en;
reg[31:0] v1_rd_data;

wire[7:0] v2_addr;
wire v2_wr_en;
wire[31:0] v2_wr_data;

reg tstart;
reg clk = 1'b0;

matmul dut_inst(
  .v0_addr    (v0_addr    ),
  .v0_rd_en   (v0_rd_en   ),
  .v0_rd_data (v0_rd_data ),
  .v1_addr    (v1_addr    ),
  .v1_rd_en   (v1_rd_en   ),
  .v1_rd_data (v1_rd_data ),
  .v2_addr    (v2_addr    ),
  .v2_wr_en   (v2_wr_en   ),
  .v2_wr_data (v2_wr_data ),
  .tstart     (tstart     ),
  .clk        (clk        )
);


initial begin
  tstart <= 'd0;
  #9
  tstart <= 1'b1;
  #2
  tstart <= 1'b0;
end

always@(posedge clk) begin
  v0_rd_data <=  {32'd0,v0_addr};
end

always@(posedge clk) begin
  v1_rd_data <=  {32'd0,v1_addr};
end

always begin
  #1
  clk <= !clk;
end
endmodule
