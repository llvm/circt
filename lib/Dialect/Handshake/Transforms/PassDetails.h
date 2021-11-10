//===- PassDetails.h - Handshake pass class details -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_HANDSHAKE_TRANSFORMS_PASSDETAILS_H
#define DIALECT_HANDSHAKE_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Handshake/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace handshake {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Handshake/Transforms/Passes.h.inc"

} // namespace handshake
} // namespace circt

#endif // DIALECT_HANDSHAKE_TRANSFORMS_PASSDETAILS_H
