//===- Handshake.cpp - C interface for the Handshake dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Handshake.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

void registerHandshakePasses() {
  circt::handshake::registerPasses();
  circt::registerCFToHandshakePass();
  circt::registerHandshakeToHWPass();
}
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Handshake, handshake,
                                      circt::handshake::HandshakeDialect)
