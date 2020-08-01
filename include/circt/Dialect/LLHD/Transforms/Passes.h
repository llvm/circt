//===- Passes.h - LLHD pass entry points ------------------------*- C++ -*-===//
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

namespace mlir {
namespace llhd {

class ProcOp;

std::unique_ptr<OperationPass<ModuleOp>> createProcessLoweringPass();

std::unique_ptr<OperationPass<ModuleOp>> createFunctionEliminationPass();

std::unique_ptr<OperationPass<ProcOp>> createEarlyCodeMotionPass();

/// Register the LLHD Transformation passes.
void initLLHDTransformationPasses();

} // namespace llhd
} // namespace mlir

#endif // CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H
