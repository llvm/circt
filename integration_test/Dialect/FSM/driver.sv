// WaveDrom visulization:
// https://wavedrom.com/editor.html?%7B%20signal%3A%20%5B%0A%20%20%7B%20name%3A%20%22clk%22%2C%20wave%3A%20%27P........%27%20%7D%2C%0A%20%20%7B%7D%2C%0A%20%20%7B%20name%3A%20%22arvalid_i%22%2C%20wave%3A%20%2201x..1x0.%22%20%7D%2C%0A%20%20%7B%20name%3A%20%22arready_o%22%2C%20wave%3A%20%221.0..101.%22%20%7D%2C%0A%20%20%7B%20name%3A%20%22(araddr_i)%22%2C%20wave%3A%20%22x3xxx4xxx%22%20%7D%2C%0A%20%20%7B%20name%3A%20%22arlen_i%22%2C%20wave%3A%20%22x%3Dxxx%3Dxxx%22%2C%20data%3A%20%5B%223%22%2C%20%221%22%5D%20%7D%2C%0A%20%20%7B%7D%2C%0A%20%20%7B%20name%3A%20%22rvalid_o%22%2C%20wave%3A%20%220.1...1.0%22%20%7D%2C%0A%20%20%7B%20name%3A%20%22rready_i%22%2C%20wave%3A%20%22xx1.....x%22%20%7D%2C%0A%20%20%7B%20name%3A%20%22(rdata_o)%22%2C%20wave%3A%20%222.3456782%22%20%7D%2C%0A%20%20%7B%20name%3A%20%22rlast_o%22%2C%20wave%3A%20%220....1010%22%20%7D%0A%5D%7D

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
