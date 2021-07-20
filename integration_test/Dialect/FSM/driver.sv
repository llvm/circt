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

  always #5 clk = ~clk;

  initial begin
    rst_n = 0;
    in0 = 0; // arvalid
    in1 = 0; // arlen
    in2 = 1; // rready

    // arready should be high after reset.
    @(posedge clk) #1;
    rst_n = 1;
    assert (out0); // arready
    assert (~out1); // rvalid
    assert (~out2); // rlast

    // Issue a burst read request with a length of 4.
    @(posedge clk) #1;
    in0 = 1; // arvalid
    in1 = 3; // arlen

    // Only rvalid should be high for the first 3 clock cycles.
    @(posedge clk) #1;
    in0 = 0; // arvalid
    in1 = 0; // arlen
    assert (~out0); // arready
    assert (out1); // rvalid
    assert (~out2); // rlast

    @(posedge clk) #1;
    assert (~out0); // arready
    assert (out1); // rvalid
    assert (~out2); // rlast

    @(posedge clk) #1;
    assert (~out0); // arready
    assert (out1); // rvalid
    assert (~out2); // rlast

    // arready, rvalid, and rlast should be high for the last clock cycle.
    // Issue another read request with a length of 2.
    @(posedge clk) #1;
    in0 = 1; // arvalid
    in1 = 1; // arlen
    assert (out0); // arready
    assert (out1); // rvalid
    assert (out2); // rlast

    // Only rvalid should be high for the first clock cycle.
    @(posedge clk) #1;
    in0 = 0; // arvalid
    in1 = 0; // arlen
    assert (~out0); // arready
    assert (out1); // rvalid
    assert (~out2); // rlast

    // arready, rvalid, and rlast should be high for the last clock cycle.
    @(posedge clk) #1;
    assert (out0); // arready
    assert (out1); // rvalid
    assert (out2); // rlast

    // Return to IDLE state.
    @(posedge clk) #1;
    assert (out0); // arready
    assert (~out1); // rvalid
    assert (~out2); // rlast

    $display("Success");
    $finish();
  end

endmodule // driver
