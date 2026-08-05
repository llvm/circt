//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the types for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4TYPES_TD
#define CIRCT_DIALECT_AXI4_AXI4TYPES_TD

include "circt/Dialect/AXI4/AXI4Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class AXI4TypeDef<string name, list<Trait> traits = []> :
    TypeDef<AXI4Dialect, name, traits>;

def PortType : AXI4TypeDef<"Port"> {
  let mnemonic = "port";
  let summary = "an AXI4 port interface";
  let description = [{
    The primary interface passed between endpoints and interconnect primitives.
    Carries the address, data, write/read ID and user widths, the access windows
    the port supports, and the maximum outstanding writes and reads, e.g.
    `!axi4.port<addr_width = 32, data_width = 64, write_id_width = 4,
    read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff,
    burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4,
    outstanding_reads = 4>`.

    The fields may be given in any order. AXI4 permits address widths up to 64
    bits, data widths that are a power of two between 8 and 1024, and ID widths
    up to 32 bits. The outstanding request counts must not exceed the number of
    distinct IDs available.
  }];

  let parameters = (ins "uint32_t":$addr_width,
                        "uint32_t":$data_width,
                        "uint32_t":$write_id_width,
                        "uint32_t":$read_id_width,
                        "uint32_t":$user_width,
                        "::circt::axi4::WindowSetAttr":$windows,
                        "uint32_t":$outstanding_writes,
                        "uint32_t":$outstanding_reads);

  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;
}

#endif // CIRCT_DIALECT_AXI4_AXI4TYPES_TD
