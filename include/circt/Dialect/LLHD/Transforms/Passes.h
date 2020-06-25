#ifndef CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H
#define CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H

#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace llhd {

#define GEN_PASS_CLASSES
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createProcessLoweringPass();

std::unique_ptr<OperationPass<ModuleOp>> createFunctionEliminationPass();

/// Register the LLHD Transformation passes.
inline void initLLHDTransformationPasses() {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
}

} // namespace llhd
} // namespace mlir

#endif // CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H
