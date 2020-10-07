`default_nettype none
`include "out.sv"
module Add_tb();
wire[7:0][6:0] v0_addr;
wire v0_rd_en;
reg[31:0]  v0_rd_data;
wire[6:0] v1_addr;
wire v1_rd_en;
reg[31:0]  v1_rd_data;
wire[6:0] v2_addr;
wire v2_wr_en;
wire[63:0]  v2_wr_data;
reg t3;
reg clk = 1'b0;

Add Add_inst(
  .v0_addr   (v0_addr   ) ,
  .v0_rd_en  (v0_rd_en  ) ,
  .v0_rd_data(v0_rd_data) ,
  .v1_addr   (v1_addr   ) ,
  .v1_rd_en  (v1_rd_en  ) ,
  .v1_rd_data(v1_rd_data) ,
  .v2_addr   (v2_addr   ) ,
  .v2_wr_en  (v2_wr_en  ) ,
  .v2_wr_data(v2_wr_data) ,
  .tstart    (t3        ) ,
  .clk       (clk       ) 
);

initial begin
  v0_rd_data <= 'd5;
  v1_rd_data <= 'd100;
  t3 <= 'd0;
  #9
  t3 <= 1'b1;
  #2
  t3 <= 1'b0;
end

initial begin
end
always@(posedge clk) begin
  if(v0_rd_en) begin
    v0_rd_data <=  v0_rd_data + 'd1;
  end
  if(v1_rd_en) begin
    v1_rd_data <=  v1_rd_data + 'd1;
  end
end

always begin
  #1
  clk <= !clk;
end
endmodule
