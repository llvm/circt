`default_nettype none
`include "../../histogram.sv"
module tb_stencil_1d();

wire[11:0] v0_addr     ;
wire v0_rd_en         ;
reg[7:0] v0_rd_data  ;
wire[7:0] v1_addr     ;
wire v1_wr_en         ;
wire[31:0] v1_wr_data ;
reg tstart            ;
reg clk       =0      ;

histogram dut_inst(
  .v0_addr   (v0_addr   ) ,
  .v0_rd_en  (v0_rd_en  ) ,
  .v0_rd_data(v0_rd_data) ,
  .v1_addr   (v1_addr   ) ,
  .v1_wr_en  (v1_wr_en  ) ,
  .v1_wr_data(v1_wr_data) ,
  .tstart    (tstart    ) ,
  .clk       (clk       ) 
);

initial begin
  v0_rd_data <= 'd0;
  tstart <= 'd0;
  #9
  tstart <= 1'b1;
  #2
  tstart <= 1'b0;
end

initial begin
end
always@(posedge clk) begin
  if(v0_rd_en) begin
    v0_rd_data <=  v0_addr == 3? 'd5:'d1;
  end
end

always begin
  #1
  clk <= !clk;
end
endmodule
