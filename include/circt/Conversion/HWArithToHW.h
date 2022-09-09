//===- HWArithToHW.h - HWArith to HW conversions ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the HWArithToHW pass
// constructor.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWARITHTOHW_HWARITHTOHW_H
#define CIRCT_CONVERSION_HWARITHTOHW_HWARITHTOHW_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
/// Get the HWArith to HW conversion patterns.
void populateHWArithToHWConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createHWArithToHWPass();
} // namespace circt

#endif // CIRCT_CONVERSION_HWARITHTOHW_HWARITHTOHW_H
