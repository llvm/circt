module driver();
  logic clk = 0;
  logic rst_n = 0;
  logic in0; // arvalid
  logic [7:0] in1; // arlen
  logic in2; // rready
  logic out0; // arready
  logic out1; // rvalid
  logic out2; // rlast

  axi_read_target dut (.*);

  always #2 clk = ~clk;

  initial begin
    in0 = 0;
    in1 = 0;
    in2 = 1;

    // Hold reset high for one clk cycle.
    rst_n = 0;
    @(posedge clk);
    rst_n = 1;

    // arready should be high in IDLE state.
    assert(out0);

    // Hold valid high for one clk cycle.
    in0 = 1;
    in1 = 7;
    @(posedge clk);
    in0 = 0;
    in1 = 0;

    // rvalid should be high in MID or END state.
    assert(out1);

    $display("Success");
    $finish();
  end

endmodule // driver
