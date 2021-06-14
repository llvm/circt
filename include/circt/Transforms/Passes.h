//===- Passes.h - Generic CIRCT Passes --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.  These
// passes are dialect independent.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSFORMS_PASSES_H
#define CIRCT_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {

std::unique_ptr<mlir::Pass> createSimpleCanonicalizerPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Transforms/Passes.h.inc"

} // namespace circt

#endif // CIRCT_TRANSFORMS_PASSES_H
