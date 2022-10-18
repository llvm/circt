//===- InteropLoweringPatterns.h - Interop lowering patterns ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose container and interop
// mechanism lowering patterns.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_INTEROP_INTEROPLOWERINGPATTERNS_H
#define CIRCT_DIALECT_INTEROP_INTEROPLOWERINGPATTERNS_H

// Forward declarations.
namespace mlir {
class RewritePatternSet;
class MLIRContext;
} // namespace mlir

namespace circt {
namespace interop {

/// Populate the four rewrite patterns to lower the unrealized interop
/// operations using the lowering provided by the containter operation
/// interface.
void populateContainerInteropPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::MLIRContext *ctx);

} // namespace interop
} // namespace circt

#endif // CIRCT_DIALECT_INTEROP_INTEROPLOWERINGPATTERNS_H
