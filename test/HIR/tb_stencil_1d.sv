`default_nettype none
`include "../../stencil_1d.sv"
module tb_stencil_1d();

wire[5:0] v0_addr     ;
wire v0_rd_en         ;
reg[31:0] v0_rd_data  ;
wire[5:0] v1_addr     ;
wire v1_wr_en         ;
wire[31:0] v1_wr_data ;
reg tstart            ;
reg clk       =0      ;

stencil_1d dut_inst(
  .v0_addr   (v0_addr   ) ,
  .v0_rd_en  (v0_rd_en  ) ,
  .v0_rd_data(v0_rd_data) ,
  .v1_addr   (v1_addr   ) ,
  .v1_wr_en  (v1_wr_en  ) ,
  .v1_wr_data(v1_wr_data) ,
  .v2(2),
  .v3(3),
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
    v0_rd_data <=  v0_rd_data + 'd1;
  end
end

always begin
  #1
  clk <= !clk;
end
endmodule
