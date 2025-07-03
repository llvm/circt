function logic [3:0] seeminglyExhaustiveCase(logic [1:0] a);
  case (a)
    2'd0: seeminglyExhaustiveCase = 4'b0001;
    2'd1: seeminglyExhaustiveCase = 4'b0010;
    2'd2: seeminglyExhaustiveCase = 4'b0100;
    2'd3: seeminglyExhaustiveCase = 4'b1000;
    default: seeminglyExhaustiveCase = 4'b1111;
  endcase
endfunction

// module Foo1 (
//   input logic [1:0] a,
//   output logic [3:0] z
// );
//   always_comb begin
//     case (a)
//       2'd0: z = 4'b0001;
//       2'd1: z = 4'b0010;
//       2'd2: z = 4'b0100;
//       2'd3: z = 4'b1000;
//     endcase
//   end
// endmodule

// module Foo2 (
//   input logic [1:0] a,
//   output logic [3:0] z
// );
//   always_comb begin
//     case (a)
//       2'd0: z = 4'b0001;
//       2'd1: z = 4'b0010;
//       2'd2: z = 4'b0100;
//       2'd3: z = 4'b1000;
//       default: z = 4'b1111;
//     endcase
//   end
// endmodule
