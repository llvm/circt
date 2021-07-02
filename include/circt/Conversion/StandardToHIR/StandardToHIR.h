//===--- StandardToHIR.h - Standard to HW conversion pass ---------*-C++-*-===//
//
// This file declares pass to convert Standard + scf + memref to HIR dialect.
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_StandardToHIR_H
#define CIRCT_CONVERSION_StandardToHIR_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

/// Creates the lower-to-hir pass.
std::unique_ptr<mlir::Pass> createConvertStandardToHIRPass();

} // namespace circt

#endif // CIRCT_CONVERSION_StandardToHIR_H
