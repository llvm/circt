//===- GAAPasses.h - GAA pass entry points ----------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_GAA_GAAPASSES_H
#define CIRCT_DIALECT_GAA_GAAPASSES_H

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace gaa {

std::unique_ptr<mlir::Pass> createGenerateConflictMatrix();
std::unique_ptr<mlir::Pass> createReferRules();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/GAA/GAAPasses.h.inc"

} // namespace gaa
} // namespace circt

#endif // CIRCT_DIALECT_GAA_GAAPASSES_H
