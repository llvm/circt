//===- InteropOps.h - Declare Interop dialect operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Interop dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_INTEROP_INTEROPOPS_H
#define CIRCT_DIALECT_INTEROP_INTEROPOPS_H

#include "circt/Dialect/Interop/InteropDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

//===----------------------------------------------------------------------===//
// Interop Mechanism
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Interop/InteropEnums.h.inc"

//===----------------------------------------------------------------------===//
// Interop Traits
//===----------------------------------------------------------------------===//

namespace circt {
namespace interop {
template <typename ConcreteType>
struct HasInterop : mlir::OpTrait::TraitBase<ConcreteType, HasInterop> {};
} // namespace interop
} // namespace circt

//===----------------------------------------------------------------------===//
// Interop Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/Interop/Interop.h.inc"

#endif // CIRCT_DIALECT_INTEROP_INTEROPOPS_H
