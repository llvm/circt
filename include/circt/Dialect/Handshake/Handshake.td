//===- Handshake.td - Handshake dialect definition ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HANDSHAKE_TD
#define HANDSHAKE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "circt/Dialect/Handshake/HandshakeInterfaces.td"

def Handshake_Dialect : Dialect {
  let name = "handshake";
  let cppNamespace = "::circt::handshake";
}

// Base class for Handshake dialect ops.
class Handshake_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Handshake_Dialect, mnemonic,
         traits #[HasParent<"handshake::FuncOp">,
         DeclareOpInterfaceMethods<NamedIOInterface>,
         DeclareOpInterfaceMethods<ControlInterface>]> {
}

include "circt/Dialect/Handshake/HandshakeOps.td"

#endif // HANDSHAKE_TD
