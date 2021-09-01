//===- PassDetail.h - Analysis Pass class details ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ANALYSIS_PASSDETAIL_H
#define ANALYSIS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace circt {

// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#include "circt/Analysis/Passes.h.inc"

} // namespace circt

#endif // ANALYSIS_PASSDETAIL_H
