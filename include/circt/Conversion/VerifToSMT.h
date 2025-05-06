//===- VerifToSMT.h - Verif to SMT dialect conversion -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_VERIFTOSMT_H
#define CIRCT_CONVERSION_VERIFTOSMT_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {
class Namespace;

#define GEN_PASS_DECL_CONVERTVERIFTOSMT
#include "circt/Conversion/Passes.h.inc"

/// Get the Verif to SMT conversion patterns.
void populateVerifToSMTConversionPatterns(TypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          Namespace &names,
                                          bool risingClocksOnly);

} // namespace circt

#endif // CIRCT_CONVERSION_VERIFTOSMT_H
