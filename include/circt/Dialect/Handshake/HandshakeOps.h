//===- Ops.h - Handshake MLIR Operations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines convenience types for working with handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_HANDSHAKEOPS_OPS_H_
#define CIRCT_HANDSHAKEOPS_OPS_H_

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Any.h"

namespace circt {
namespace handshake {

class TerminatorOp;

#include "circt/Dialect/Handshake/HandshakeInterfaces.h.inc"

} // end namespace handshake
} // end namespace circt

namespace mlir {
namespace OpTrait {
template <typename ConcreteType>
class HasClock : public TraitBase<ConcreteType, HasClock> {};
} // namespace OpTrait
} // namespace mlir

#define GET_OP_CLASSES
#include "circt/Dialect/Handshake/Handshake.h.inc"

#endif // MLIR_HANDSHAKEOPS_OPS_H_
