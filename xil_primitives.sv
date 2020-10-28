//Block ram True-Dual-Port, Read-First.
module bram_tdp_rf_rf#(SIZE=1024,WIDTH=256) (clka,clkb,ena,enb,wea,web,addra,addrb,dia,dib,doa,dob);
input wire clka,clkb,ena,enb,wea,web;
input wire [$clog2(SIZE)-1:0] addra,addrb;
input wire [WIDTH-1:0] dia,dib;
output reg [WIDTH-1:0] doa,dob;
reg [WIDTH-1:0] ram [SIZE-1:0];
//reg [WIDTH-1:0] doa,dob;
always @(posedge clka)
begin
  if (ena)
  begin
    if (wea)
      ram[addra] <= dia;
    doa <= ram[addra];
  end
end
always @(posedge clkb)
begin
  if (enb)
  begin
    if (web)
      ram[addrb] <= dib;
    dob <= ram[addrb];
  end
end
endmodule
