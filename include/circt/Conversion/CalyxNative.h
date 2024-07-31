//===- CalyxNative.h - Calyx Native pass ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Calyx dialect to the
// HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_CALYXNATIVE_H
#define CIRCT_CONVERSION_CALYXNATIVE_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_CALYXNATIVE
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCalyxNativePass();

} // namespace circt

#endif // CIRCT_CONVERSION_CALYXNATIVE_H
