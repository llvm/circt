module top();
  logic clock = 0;
  logic reset = 0;
  logic arg0_valid, arg0_ready;
  logic arg1_valid, arg1_ready;
  logic [31:0] arg1_data;
  logic arg2_valid, arg2_ready;

  main main (.*);

  always begin
    // A clock period is #4.
    clock = ~clock;
    #2;
  end

  initial begin
    arg0_valid = 1;
    arg1_ready = 1;
    arg2_ready = 1;

    reset = 1;
    // Hold reset high for one clock cycle.
    @(posedge clock);
    reset = 0;

    wait(arg1_valid == 1);

    $display("%d", arg1_data);
    $finish();
  end

endmodule // driver
