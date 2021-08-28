//===- HandshakeDialect.cpp - Implement the Handshake dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace circt::handshake;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void HandshakeOpsDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Handshake/Handshake.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#include "circt/Dialect/Handshake/HandshakeAttrs.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeDialect.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeEnums.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
