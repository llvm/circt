//===- Passes.h - HIR pass entry points ------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H

#include "circt/Dialect/HIR/HIR.h"
#include "circt/Support/LLVM.h"
#include <memory>
namespace mlir {
namespace hir {

std::unique_ptr<OperationPass<hir::FuncOp>> createCanonicalizationPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createScheduleVerificationPass();
void initHIRTransformationPasses();
} // namespace hir
} // namespace mlir
#endif // CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H
