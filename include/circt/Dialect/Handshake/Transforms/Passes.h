//===- Passes.h - Handshake Passes ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_TRANSFORMS_PASSES_H
#define CIRCT_DIALECT_HANDSHAKE_TRANSFORMS_PASSES_H

#include "circt/Dialect/ESI/ESIDialect.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace handshake {

std::unique_ptr<mlir::Pass> createHandshakeWrapESIPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Handshake/Transforms/Passes.h.inc"

} // namespace handshake
} // namespace circt

#endif // CIRCT_DIALECT_HANDSHAKE_TRANSFORMS_PASSES_H
