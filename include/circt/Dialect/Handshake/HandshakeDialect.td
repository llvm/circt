//===- HandshakeDialect.td - Handshake dialect definition --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKEDIALECT_TD
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKEDIALECT_TD

include "mlir/IR/OpBase.td"

def Handshake_Dialect : Dialect {
  let name = "handshake";
  let cppNamespace = "::circt::handshake";

  let summary = "Types and operations for the handshake dialect";
  let description = [{
    This dialect defined the `handshake` dialect, modeling dataflow circuits.
    Handshake/dataflow IR is describes independent, unsynchronized processes
    communicating data through First-in First-out (FIFO) communication channels.
  }];

  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "circt::esi::ESIDialect",
    "circt::seq::SeqDialect"
  ];
}

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEDIALECT_TD
