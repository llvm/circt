module driver();
  logic clock = 0;
  logic reset = 0;
  logic arg0_valid, arg0_ready;
  logic out0_valid, out0_ready;
  logic [31:0] out0_data;
  logic outCtrl_valid, outCtrl_ready;

  top dut (.*);

  always begin
    // A clock period is #4.
    clock = ~clock;
    #2;
  end

  initial begin
    arg0_valid = 1;
    out0_ready = 1;
    outCtrl_ready = 1;

    reset = 1;
    // Hold reset high for one clock cycle.
    @(posedge clock);
    reset = 0;

    // Hold valid high for one clock cycle.
    @(posedge clock);
    arg0_valid = 0;

    wait(out0_valid == 1);

    $display("Result=%d", out0_data);
    $finish();
  end

endmodule // driver
