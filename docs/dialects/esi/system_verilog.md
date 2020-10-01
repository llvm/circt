[Back to table of contents](index.md#Table-of-contents)

# ESI System Verilog integration roadmap

Here is an example schema (in CapnProto schema format) which will be used in
the examples below.

```capnp
using ESI = import "/EsiCoreAnnotations.capnp";

struct Polynomial3 { # ax^2 + bx + c
    a @0 :UInt32 $ESI.bits(24);
    b @1 :UInt32 $ESI.bits(20);
    c @2 :UInt16;
}
```

## System Verilog types

Most of the ESI types map quite nicely to the System Verilog.

```sv
// *****
// struct Polynomial3
//
typedef struct packed
{
    logic [23:0] a;
    logic [19:0] b;
    logic [15:0] c;
} Polynomial3;

```

## System Verilog interfaces

In order to bundle both the data and signaling, System Verilog interfaces
will be used:

```sv
// *****
// 'Polynomial3' message interface with ready/valid semantics
//
interface IPolynomial3ValidReady
    (
        input wire clk,
        input wire rstn
    );

    logic valid;
    logic ready;

    Polynomial3 data;

    modport Source (
        input clk, rstn,
        output valid,
        input ready,

        output data
    );

    modport Sink (
        input clk, rstn,
        input valid,
        output ready,

        input data
    );
endinterface
```

to be used like this:

```sv
// Compute ax^2 + bx + c = y
module Polynomial3Compute (
    input logic clk,
    input logic rstn,

    // ESI connection
    IPolynomial3ValidReady.Sink abc,

    // Legacy wires
    input logic[9:0] x,
    output logic[45:0] y
);

    Polynomial3 dr;
    logic[9:0] xr;
    assign abc.ready = 1'b1;

    always_ff @(posedge clk) begin
        if (abc.valid) begin
            dr <= abc.data;
            xr <= x;
        end
    end

    always_comb begin
        y = (dr.a*xr*xr) + (dr.b*xr) + dr.c;
    end

endmodule
```

```sv
module Polynomial3Compute_tb ();
    logic clk;

    IPolynomial3ValidReady inputAbc (.clk(clk), .rstn(1'b1));
    logic [9:0] inputX;
    logic [45:0] outputY;

    Polynomial3Compute dut (
        .clk(clk),
        .rstn(1'b1),

        .abc(inputAbc.Sink),
        .x(inputX),

        .y(outputY)
    );

    initial begin
        clk = 0;
        #10
        for (int i=0; i<10; i++) begin
            #5;
            clk = !clk;
        end
    end

    initial begin
        #17
        inputAbc.valid = 1;
        inputAbc.data.a = 42;
        inputAbc.data.b = 184;
        inputAbc.data.c = 2;
        inputX = 1;
    end

endmodule
```
