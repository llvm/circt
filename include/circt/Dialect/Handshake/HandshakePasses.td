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

def HandshakeMaterializeForksSinks : Pass<"handshake-materialize-forks-sinks", "handshake::FuncOp"> {
  let summary = "Materialize fork and sink operations.";
  let description = [{
    This pass analyses a handshake.func operation and inserts fork and sink
    operations ensuring that all values have exactly one use.
  }];
  let constructor = "circt::handshake::createHandshakeMaterializeForksSinksPass()";
}

def HandshakeDematerializeForksSinks : Pass<"handshake-dematerialize-forks-sinks", "handshake::FuncOp"> {
  let summary = "Dematerialize fork and sink operations.";
  let description = [{
    This pass analyses a handshake.func operation and removes all fork and sink
    operations.
  }];
  let constructor = "circt::handshake::createHandshakeDematerializeForksSinksPass()";
}

def HandshakeRemoveBuffers : Pass<"handshake-remove-buffers", "handshake::FuncOp"> {
  let summary = "Remove buffers from handshake functions.";
  let description = [{
    This pass analyses a handshake.func operation and removes any buffers from
    the function.
  }];
  let constructor = "circt::handshake::createHandshakeRemoveBuffersPass()";
}

def HandshakeAddIDs : Pass<"handshake-add-ids", "handshake::FuncOp"> {
  let summary = "Add an ID to each operation in a handshake function.";
  let description = [{
    This pass adds an ID to each operation in a handshake function. This id can
    be used in lowerings facilitate mapping lowered IR back to the handshake code
    which it originated from. An ID is unique with respect to other operations
    of the same type in the function. The tuple of the operation name and the
    operation ID denotes a unique identifier for the operation within the
    `handshake.func` operation.
  }];
  let constructor = "circt::handshake::createHandshakeAddIDsPass()";
}

def HandshakeInsertBuffers
  : Pass<"handshake-insert-buffers", "handshake::FuncOp"> {
  let summary = "Insert buffers to break graph cycles";
  let constructor = "circt::handshake::createHandshakeInsertBuffersPass()";
  let options = [
    Option<"strategy", "strategy", "std::string", "\"all\"",
           "Strategy to apply. Possible values are: cycles, allFIFO, all (default)">,
    Option<"bufferSize", "buffer-size", "unsigned", /*default=*/"2",
           "Number of slots in each buffer">,
  ];
}

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_TD
