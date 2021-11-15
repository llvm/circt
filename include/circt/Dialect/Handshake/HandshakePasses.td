//===-- HandshakePasses.td - Handshake pass definition file *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_TD
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_TD

include "mlir/Pass/PassBase.td"

def HandshakeDotPrint : Pass<"handshake-print-dot", "mlir::ModuleOp"> {
  let summary = "Print .dot graph of a handshake function.";
  let description = [{
    This pass analyses a handshake.func operation and prints a .dot graph of the
    structure. If multiple functions are present in the IR, the top level
    function will be printed, and called functions will be subgraphs within
    the main graph.
  }];
  let constructor = "circt::handshake::createHandshakeDotPrintPass()";
}

def HandshakeOpCount : Pass<"handshake-op-count", "mlir::ModuleOp"> {
  let summary = "Count the number of operations (resources) in a handshake function.";
  let description = [{
    This pass analyses a handshake.func operation and prints the number of
    operations (resources) used the function.
  }];
  let constructor = "circt::handshake::createHandshakeOpCountPass()";
}

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_TD
