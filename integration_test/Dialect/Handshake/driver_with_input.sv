module driver();
  logic clock = 0;
  logic reset = 0;
  logic out0_valid, out0_ready;
  logic [63:0] out0_data;
  logic inCtrl_valid, inCtrl_ready;
  logic outCtrl_valid, outCtrl_ready;

  logic in0_valid, in0_ready;
  logic [63:0] in0_data;

  top dut (.*);

  always begin
    // A clock period is #4.
    clock = ~clock;
    #2;
  end

  logic wasRdy;
  logic [15:0] clkCnt = 0;

  logic directSend;

  task send(input [63:0] data);
    in0_valid = 1;
    inCtrl_valid = 1;
    in0_data = data;

    // If both signals are already asserted, the handshake will take place in
    // the current cycles. Otherwise, we have to wait an additional cycle in
    // the end.
    directSend = (in0_ready == 1 && inCtrl_ready == 1);

    awaitHandshake();
    if(~directSend) begin
      @(posedge clock); 
    end
  endtask


  task awaitHandshake();
    // Ensure that each input is only transmitted once
    fork
      begin
        wait(in0_ready == 1);
        @(posedge clock); 
        in0_valid = 0;
      end

      begin
        wait(inCtrl_ready == 1);
        @(posedge clock); 
        inCtrl_valid = 0;
      end
    join
  endtask

  initial begin
    out0_ready = 1;
    outCtrl_ready = 1;
    in0_valid = 0;
    inCtrl_valid = 0;

    reset = 1;
    // Hold reset high for one clock cycle.
    @(posedge clock);
    reset = 0;

    // give reset some time
    @(posedge clock);

    send(0);
    send(24);
  end

  logic [15:0] resCnt = 0;
  always @(posedge clock) begin
    if(clkCnt == 10000) begin
      $finish();
    end
    if(out0_valid == 1) begin
      $display("Result=%d", out0_data);
      resCnt += 1;
    end

    if(resCnt == 2) begin
      $finish();
    end
    clkCnt += 1;
  end
endmodule // driver
