// REQUIRES: esi-cosim, rtl-sim
// RUN: esi-cosim-runner.py --exec %s.py %s


import Cosim_DpiPkg::*;

module top(
    input logic clk,
    input logic rst
);

    Cosim_Manifest #(
      .COMPRESSED_MANIFEST_SIZE(30)
    ) __manifest (
      // zlib-compressed JSON:
      // {
      //   "api_version": 1
      // }
      .compressed_manifest('{
        8'h78, 8'h9C, 8'hAB, 8'hE6, 8'h52, 8'h50, 8'h50, 8'h4A,
        8'h2C, 8'hC8, 8'h8C, 8'h2F, 8'h4B, 8'h2D, 8'h2A, 8'hCE,
        8'hCC, 8'hCF, 8'h53, 8'hB2, 8'h52, 8'h30, 8'hE4, 8'hAA,
        8'h05, 8'h00, 8'h4D, 8'h63, 8'h06, 8'hBB
      })
    );

    localparam int TYPE_SIZE_BITS =
        (1 * 64) + // root message
        (1 * 64) + // list header
        (1 * 64);  // list of length 3 bytes, rounded up to multiples of 8 bytes

    wire DataOutValid;
    wire DataOutReady = 1;
    wire [TYPE_SIZE_BITS-1:0] DataOut;
    
    logic DataInValid;
    logic DataInReady;
    logic [TYPE_SIZE_BITS-1:0] DataIn;

    Cosim_Endpoint #(
        .RECV_TYPE_ID(1),
        .RECV_TYPE_SIZE_BITS(TYPE_SIZE_BITS),
        .SEND_TYPE_ID(1),
        .SEND_TYPE_SIZE_BITS(TYPE_SIZE_BITS)
    ) ep (
        .*
    );

    always@(posedge clk)
    begin
        if (~rst)
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
endmodule
