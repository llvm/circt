`default_nettype none
`include "out.sv"
module Add_tb();
wire[6:0] v_addr0;
wire v_rd_en0;
reg[31:0]  v_rd_data0;
wire[6:0] v_addr1;
wire v_rd_en1;
reg[31:0]  v_rd_data1;
wire[6:0] v_addr2;
wire v_wr_en2;
wire[63:0]  v_wr_data2;
reg t3;
reg clk = 1'b0;

Add Add_inst(
  .v_addr0   (v_addr0   ) ,
  .v_rd_en0  (v_rd_en0  ) ,
  .v_rd_data0(v_rd_data0) ,
  .v_addr1   (v_addr1   ) ,
  .v_rd_en1  (v_rd_en1  ) ,
  .v_rd_data1(v_rd_data1) ,
  .v_addr2   (v_addr2   ) ,
  .v_wr_en2  (v_wr_en2  ) ,
  .v_wr_data2(v_wr_data2) ,
  .t3        (t3        ) ,
  .clk       (clk       ) 
);

initial begin
  v_rd_data0 <= 'd5;
  v_rd_data1 <= 'd100;
  t3 <= 'd0;
  #9
  t3 <= 1'b1;
  #2
  t3 <= 1'b0;
end

initial begin
end
always@(posedge clk) begin
  if(v_rd_en0) begin
      v_rd_data0 <=  v_rd_data0 + 'd1;
  end
  if(v_rd_en1) begin
      v_rd_data1 <=  v_rd_data1 + 'd1;
  end
end

always begin
  #1
  clk <= !clk;
end
endmodule
