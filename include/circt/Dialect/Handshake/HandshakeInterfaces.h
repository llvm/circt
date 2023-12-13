//===- HandshakeInterfaces.h - Handshake op interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces of the handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKEINTERFACES_H
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKEINTERFACES_H

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/Any.h"

namespace circt {
namespace handshake {

struct MemLoadInterface {
  unsigned index;
  mlir::Value addressIn;
  mlir::Value dataOut;
  mlir::Value doneOut;
};

struct MemStoreInterface {
  unsigned index;
  mlir::Value addressIn;
  mlir::Value dataIn;
  mlir::Value doneOut;
};

/// Default implementation for checking whether an operation is a control
/// operation. This function cannot be defined within ControlInterface
/// because its implementation attempts to cast the operation to an
/// SOSTInterface, which may not be declared at the point where the default
/// trait's method is defined. Therefore, the default implementation of
/// ControlInterface's isControl method simply calls this function.
bool isControlOpImpl(Operation *op);
} // end namespace handshake
} // end namespace circt

#include "circt/Dialect/Handshake/HandshakeInterfaces.h.inc"

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEINTERFACES_H
