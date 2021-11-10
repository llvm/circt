//===-- Passes.td - Handshake pass definition file ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_TRANSFORMS_PASSES
#define CIRCT_DIALECT_HANDSHAKE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HandshakeWrapESI : Pass<"handshake-wrap-esi", "mlir::ModuleOp"> {
  let summary = "Wrap the top-level Handshake module with ESI ports";
  let constructor = "circt::handshake::createHandshakeWrapESIPass()";
  let dependentDialects = ["circt::esi::ESIDialect"];
  let options = [
    Option<"topModule", "top-module", "std::string", "\"top\"",
           "Name of the top-level module to process">,
  ];
}

#endif // CIRCT_DIALECT_HANDSHAKE_TRANSFORMS_PASSES