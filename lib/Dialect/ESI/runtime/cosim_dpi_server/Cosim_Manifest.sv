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
  parameter int COMPRESSED_MANIFEST_SIZE = 0,
  parameter int unsigned ESI_VERSION = 0
)(
  input logic [COMPRESSED_MANIFEST_SIZE-1:0][7:0] compressed_manifest
);

  byte unsigned compressed_manifest_bytes[COMPRESSED_MANIFEST_SIZE-1:0];
  always_comb
    for (int i=0; i<COMPRESSED_MANIFEST_SIZE; i++)
      compressed_manifest_bytes[i] = compressed_manifest[i];

  always@(compressed_manifest)
    cosim_set_manifest(ESI_VERSION, compressed_manifest_bytes);

endmodule
