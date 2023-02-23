# EDA Tool Workarounds

This documents various bugs found in EDA tools and their workarounds in circt.
Each but will have a brief description, example code, and the mitigation added
(with links to the commit when possible).  


# Inline Array calculations can cause synthesis failures

## Example
```
module Foo (input clock, input in, output [2:0] out);
  reg [2:0] state;
  wire [7:0][2:0] array = 24'h4 << 6;
  wire [2:0] a = array[state];
  wire [2:0] b = array[state + 3'h1 + 3'h1];
  // works:      array[state + (3'h1 + 3'h1)]
  // works:      array[state + 3'h2]
  always @(posedge clock) state <= in ? a : b;
  assign out = b;
endmodule
```

## Workaround

Flag added to export verilog to force array index calculations to not be inline.

https://github.com/llvm/circt/commit/15a1f95f2d59767f20b459a12ac42338de22bc97