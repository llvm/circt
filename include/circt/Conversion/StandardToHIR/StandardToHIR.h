//===--- AffineToHIR.h - Affine to HW conversion pass ------------*- C++-*-===//
//
// This file declares pass to convert Affine + std + memref dialect to HIR.
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_AffineToHIR_H
#define CIRCT_CONVERSION_AffineToHIR_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

/// Creates the lower-to-hir pass.
std::unique_ptr<mlir::Pass> createConvertStandardToHIRPass();

} // namespace circt

#endif // CIRCT_CONVERSION_AffineToHIR_H
