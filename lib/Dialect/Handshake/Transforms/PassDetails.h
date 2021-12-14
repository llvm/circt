//===- PassDetails.h - Handshake pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the stuff shared between the different Handshake passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the
// header guard on some systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_HANDSHAKE_TRANSFORMS_PASSDETAILS_H
#define DIALECT_HANDSHAKE_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace handshake {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"

} // namespace handshake
} // namespace circt

#endif // DIALECT_HANDSHAKE_TRANSFORMS_PASSDETAILS_H
