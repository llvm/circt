//===- Cosim_Manifest.sv - ESI manifest module -------------*- verilog -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A cosimulated design needs to instantiate this ONCE with the zlib-compressed
// JSON manifest as the parameter.
//
//===----------------------------------------------------------------------===//

import Cosim_DpiPkg::*;

module Cosim_Manifest
#(
  parameter int COMPRESSED_MANIFEST_SIZE = 0
)(
  input logic clk,
  input byte unsigned compressed_manifest[COMPRESSED_MANIFEST_SIZE]
);

  // Registration logic -- run once.
  bit Initialized = 0;
  always@(posedge clk) begin
    if (!Initialized) begin
      cosim_set_manifest(compressed_manifest);
      Initialized = 1;
    end
  end

endmodule
