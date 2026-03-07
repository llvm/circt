// Minimal FSM: state register + counter variable register.
// In state 0: if x, go to state 1 and increment counter.
// In state 1: go back to state 0 (counter unchanged).
//
// Expected: guard for 0->1 transition checks "x == 1",
// action should be just "counter <- counter + 1".
// Suspected inefficiency: action is "counter <- x ? counter+1 : counter".

module test_action_guard(
  input  logic       clk,
  input  logic       rst,
  input  logic       x,
  output logic [1:0] count
);
  logic       state;
  logic [1:0] counter;

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      state   <= 1'b0;
      counter <= 2'b00;
    end else begin
      case (state)
        1'b0: begin
          if (x) begin
            state   <= 1'b1;
            counter <= counter + 2'b01;
          end
        end
        1'b1: begin
          state   <= 1'b0;
        end
      endcase
    end
  end

  assign count = counter;
endmodule
