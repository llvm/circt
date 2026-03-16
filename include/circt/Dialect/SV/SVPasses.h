//===- SVPasses.h - SV pass entry points ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVPASSES_H
#define CIRCT_DIALECT_SV_SVPASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {
namespace sv {

#define GEN_PASS_DECL
#include "circt/Dialect/SV/SVPasses.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/SV/SVPasses.h.inc"

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_SVPASSES_H
