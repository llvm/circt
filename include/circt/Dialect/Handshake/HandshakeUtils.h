//===- HandshakeUtils.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKEUTILS_H
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKEUTILS_H

#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/LLVM.h"
#include <map>

namespace circt {
namespace handshake {

/// Iterates over the handshake::FuncOp's in the program to build an instance
/// graph. In doing so, we detect whether there are any cycles in this graph, as
/// well as infer a top function for the design by performing a topological sort
/// of the instance graph. The result of this sort is placed in sortedFuncs.
using InstanceGraph = std::map<std::string, std::set<std::string>>;
LogicalResult resolveInstanceGraph(ModuleOp moduleOp,
                                   InstanceGraph &instanceGraph,
                                   std::string &topLevel,
                                   SmallVectorImpl<std::string> &sortedFuncs);

/// Checks all block arguments and values within op to ensure that all
/// values have exactly one use.
LogicalResult verifyAllValuesHasOneUse(handshake::FuncOp op);

/// Attribute name for the name of a predeclaration of the to-be-lowered
/// hw.module from a handshake function.
static constexpr const char *kPredeclarationAttr = "handshake.module_name";

/// Converts 't' into a valid HW type. This is strictly used for converting
/// 'index' types into a fixed-width type.
Type toValidType(Type t);

/// Wraps a type into an ESI ChannelType type. The inner type is converted to
/// ensure comprehensability by the RTL dialects.
esi::ChannelType esiWrapper(Type t);

/// Returns the hw::ModulePortInfo that corresponds to the given handshake
/// operation and its in- and output types.
hw::ModulePortInfo getPortInfoForOpTypes(mlir::Operation *op, TypeRange inputs,
                                         TypeRange outputs);

/// Adds fork operations to any value with multiple uses in r.
void insertFork(Value result, bool isLazy, OpBuilder &rewriter);

} // namespace handshake
} // namespace circt

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEUTILS_H
