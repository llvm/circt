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

    wire DataOutValid;
    wire DataOutReady = 1;
    wire [23:0] DataOut;
    
    logic DataInValid;
    logic DataInReady;
    logic [31:0] DataIn;

    Cosim_Endpoint_FromHost #(
        .ENDPOINT_ID("fromHost"),
        .FROM_HOST_TYPE_ID("i24"),
        .FROM_HOST_SIZE_BITS(24)
    ) fromHost (
        .*
    );

    Cosim_Endpoint_ToHost #(
        .ENDPOINT_ID("toHost"),
        .TO_HOST_TYPE_ID("i32"),
        .TO_HOST_SIZE_BITS(32)
    ) toHost (
        .*
    );

    always@(posedge clk)
    begin
        if (~rst)
        begin
            if (DataOutValid && DataOutReady)
            begin
                $display("[%d] Recv'd: %h", $time(), DataOut);
                DataIn <= {8'b0, DataOut};
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
