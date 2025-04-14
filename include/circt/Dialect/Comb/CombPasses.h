//===- Passes.h - Comb pass entry points ------------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_COMB_COMBPASSES_H
#define CIRCT_DIALECT_COMB_COMBPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace mlir {
class DataFlowSolver;
}

namespace circt {
namespace comb {

/// Generate the code for registering passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Comb/Passes.h.inc"

/// Add patterns for int range based narrowing.
void populateCombNarrowingPatterns(RewritePatternSet &patterns,
                                   mlir::DataFlowSolver &solver);

} // namespace comb
} // namespace circt

#endif // CIRCT_DIALECT_COMB_COMBPASSES_H
