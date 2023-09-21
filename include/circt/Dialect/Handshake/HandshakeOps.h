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
#include "circt/Dialect/Handshake/HandshakeInterfaces.h"
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
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Any.h"

namespace mlir {
namespace OpTrait {
template <typename ConcreteType>
class HasClock : public TraitBase<ConcreteType, HasClock> {};

template <typename InterfaceType>
class HasParentInterface {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType,
                                HasParentInterface<InterfaceType>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      if (llvm::isa_and_nonnull<InterfaceType>(op->getParentOp()))
        return success();

      // @mortbopet: What a horrible error message - however, there's no way to
      // report the interface name without going in and adjusting the tablegen
      // backend to also emit string literal names for interfaces.
      return op->emitOpError() << "expects parent op to be of the interface "
                                  "parent type required by the given op type";
    }
  };
};

} // namespace OpTrait
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Handshake/HandshakeAttributes.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Handshake/Handshake.h.inc"

#endif // MLIR_HANDSHAKEOPS_OPS_H_
