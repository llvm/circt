// PY: import base_dpi_test as test
// PY: rpc = test.LoopbackTester(rpcSchemaPath)
// PY: rpc.write_read_many(5)

import Cosim_DpiPkg::*;

module main(
    `ifdef VERILATOR
    input logic clk,
    input logic rstn
    `endif
);
    localparam int TYPE_SIZE_BITS =
        (1 * 64) + // root message
        (1 * 64) + // list header
        (1 * 64);  // list of length 3 bytes, rounded up to multiples of 8 bytes
    `ifndef VERILATOR
    logic clk;
    logic rstn;
    `endif

    wire DataOutValid;
    wire DataOutReady = 1;
    wire [TYPE_SIZE_BITS-1:0] DataOut;
    
    logic DataInValid;
    logic DataInReady;
    logic [TYPE_SIZE_BITS-1:0] DataIn;

    Cosim_Endpoint #(
        .ENDPOINT_ID(1),
        .ESI_TYPE_ID(1),
        .TYPE_SIZE_BITS(TYPE_SIZE_BITS)
    ) ep (
        .*
    );

    always@(posedge clk)
    begin
        if (rstn)
        begin
            if (DataOutValid && DataOutReady)
            begin
                $display("[%d] Recv'd: %h", $time(), DataOut);
                DataIn <= DataOut;
            end
            DataInValid <= DataOutValid && DataOutReady;

            if (DataInValid && DataInReady)
            begin
                $display("[%d] Sent: %h", $time(), DataIn);
            end
        end
        else
        begin
            DataInValid <= 0;
        end
    end

    //     #10
    //     // Accept 1 token
    //     DataOutReady = 1;
    //     #1
    //     @(posedge clk && DataOutValid);
    //     #1
    //     $display("Recv'd: %h", DataOut);
    //     DataOutReady = 0;
    //     #8

    //     #10
    //     // Send a token
    //     DataIn = 1024'hDEADBEEF;
    //     DataInValid = 1;
    //     @(posedge clk && DataInReady);
    //     #1
    //     DataInValid = 0;
    //     #9
    //     $finish();
    // end

`ifndef VERILATOR
    // Clock
    initial
    begin
        clk = 1'b0;
        while (1)
        begin
            #5;
            clk = !clk;
        end
    end

    initial
    begin
        rstn = 0;
        #17
        // Run!
        rstn = 1;
    end
`endif

endmodule
