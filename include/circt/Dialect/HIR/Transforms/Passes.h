//===- Passes.h - HIR pass entry points ------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Support/LLVM.h"
#include <memory>
namespace circt {
namespace hir {

std::unique_ptr<OperationPass<hir::FuncOp>> createSimplifyLoopPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createIndexLoweringPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createRegisterAllocationPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createSeqSchedulerPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createMemrefLoweringPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createScheduleVerificationPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createLoopUnrollPass();

void registerHIRLoweringPassPipeline();
void initHIRTransformationPasses();
} // namespace hir
} // namespace circt
#endif // CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H
