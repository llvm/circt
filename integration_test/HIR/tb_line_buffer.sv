`default_nettype none
module tb_line_buffer();

  wire tready; 
  reg[31:0] delay=32;
  reg tvalid=1'b1;
  reg [31:0]  vi = 32'b0;
  wire to;
  wire[31:0] vo[2][2];

  reg tstart            ;
  reg clk       =0      ;

  line_buffer dut_inst(
    .v0(tvalid),
    .v1(tready),
    .v2(vi),
    .v3(to),
    .v4(vo),
    .tstart(tstart),
    .clk(clk)        
  );

  initial begin
    tstart <= 'd0;
    tvalid <= 'd1;
    #9
    tstart <= 1'b1;
    #2
    tstart <= 1'b0;
    #20
    tvalid <= 'd0;
    #20
    tvalid <= 'd1;
  end

  always@(posedge clk) begin
    if(tvalid && tready) begin
      vi <=  vi + 'd1;
    end
  end

  always begin
    #1
    clk <= !clk;
  end
endmodule
