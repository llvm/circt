//===- CombToSMT.h - Comb to SMT dialect conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_COMBTOSMT_H
#define CIRCT_CONVERSION_COMBTOSMT_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

/// Get the HW to SMT conversion patterns.
void populateCombToSMTConversionPatterns(TypeConverter &converter,
                                         RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTCOMBTOSMT
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CIRCT_CONVERSION_COMBTOSMT_H
