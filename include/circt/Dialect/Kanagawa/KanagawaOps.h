//===- KanagawaOps.h - Definition of Kanagawa dialect ops -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_KANAGAWAOPS_H
#define CIRCT_DIALECT_KANAGAWA_KANAGAWAOPS_H

#include "circt/Dialect/DC/DCTypes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/Handshake/HandshakeInterfaces.h"
#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/InstanceGraphInterface.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
namespace circt {
namespace kanagawa {
class ContainerOp;

// Symbol name for the kanagawa operator library to be used during scheduling.
static constexpr const char *kKanagawaOperatorLibName =
    "kanagawa_operator_library";

namespace detail {
// Verify that `op` conforms to the ScopeOpInterface.
LogicalResult verifyScopeOpInterface(Operation *op);

} // namespace detail
} // namespace kanagawa
} // namespace circt

#include "circt/Dialect/Kanagawa/KanagawaInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Kanagawa/Kanagawa.h.inc"

#endif // CIRCT_DIALECT_KANAGAWA_KANAGAWAOPS_H
