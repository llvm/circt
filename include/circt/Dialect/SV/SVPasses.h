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

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
namespace sv {

std::unique_ptr<mlir::Pass> createRTLCleanupPass();
std::unique_ptr<mlir::Pass> createRTLStubExternalModulesPass();
std::unique_ptr<mlir::Pass> createRTLLegalizeNamesPass();
std::unique_ptr<mlir::Pass> createRTLMemSimImplPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/SV/SVPasses.h.inc"

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_SVPASSES_H
