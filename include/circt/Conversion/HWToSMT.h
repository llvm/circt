//===- HWToSMT.h - HW to SMT dialect conversion -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWTOSMT_H
#define CIRCT_CONVERSION_HWTOSMT_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

#define GEN_PASS_DECL_CONVERTHWTOSMT
#include "circt/Conversion/Passes.h.inc"

/// Get the HW to SMT conversion patterns.
void populateHWToSMTConversionPatterns(TypeConverter &converter,
                                       RewritePatternSet &patterns);

/// Get the HW to SMT type conversions.
void populateHWToSMTTypeConverter(TypeConverter &converter);

} // namespace circt

#endif // CIRCT_CONVERSION_HWTOSMT_H
