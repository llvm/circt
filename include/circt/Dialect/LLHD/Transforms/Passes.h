//===- Passes.h - LLHD pass entry points ------------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H
#define CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
namespace llhd {

class ProcOp;

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createProcessLoweringPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createFunctionEliminationPass();

std::unique_ptr<mlir::OperationPass<ProcOp>> createMemoryToBlockArgumentPass();

std::unique_ptr<mlir::OperationPass<ProcOp>> createEarlyCodeMotionPass();

/// Register the LLHD Transformation passes.
void initLLHDTransformationPasses();

} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H
