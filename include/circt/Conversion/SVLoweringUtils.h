//===- SVLoweringUtils.h - Shared helpers for SV lowering -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helpers shared across conversions that lower into the SV
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_SVLOWERINGUTILS_H
#define CIRCT_CONVERSION_SVLOWERINGUTILS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace circt::sv {

/// Return a reference to the shared file descriptor runtime fragment.
FlatSymbolRefAttr getFileDescriptorFragmentRef(MLIRContext *context);

/// Emit the shared file descriptor runtime declarations into a file-level
/// symbol table operation. The builder insertion point must be directly within
/// `fileScopeOp`. If these declarations already exist in `fileScopeOp`, this
/// function is a no-op.
void emitFileDescriptorRuntime(Operation *fileScopeOp,
                               ImplicitLocOpBuilder &builder);

/// Create a call to the shared file descriptor getter from a procedural region.
Value createProceduralFileDescriptorGetterCall(OpBuilder &builder, Location loc,
                                               Value fileName);

} // namespace circt::sv

#endif // CIRCT_CONVERSION_SVLOWERINGUTILS_H
