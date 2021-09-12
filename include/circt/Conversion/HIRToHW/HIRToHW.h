//===- HIRToHW.h - HIR to HW conversion pass --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the HIR dialect to
// HW, Comb and SV dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HIRTOHW_H
#define CIRCT_CONVERSION_HIRTOHW_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

/// Creates the hir-to-hw pass.
std::unique_ptr<mlir::Pass> createHIRToHWPass();

} // namespace circt

#endif // CIRCT_CONVERSION_HIRTOHW_H
