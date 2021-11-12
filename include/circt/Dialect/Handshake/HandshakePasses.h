//===- HandshakePasses.h - Handshake pass entry points ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H

#include "circt/Support/LLVM.h"
#include <map>
#include <memory>
#include <set>

namespace circt {
namespace handshake {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeDotPrintPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeOpCountPass();

/// Iterates over the handshake::FuncOp's in the program to build an instance
/// graph. In doing so, we detect whether there are any cycles in this graph, as
/// well as infer a top function for the design by performing a topological sort
/// of the instance graph. The result of this sort is placed in sortedFuncs.
using InstanceGraph = std::map<std::string, std::set<std::string>>;
LogicalResult resolveInstanceGraph(ModuleOp moduleOp,
                                   InstanceGraph &instanceGraph,
                                   std::string &topLevel,
                                   SmallVectorImpl<std::string> &sortedFuncs);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"

} // namespace handshake
} // namespace circt

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
