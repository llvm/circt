//===- HWToLLVM.h - Registration of HW to LLVM patterns ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the HW to LLVM operation
// conversion pattern registration.
//
//===----------------------------------------------------------------------===//

#ifndef HWTOLLVM_H
#define HWTOLLVM_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace circt {

/// Get the HW to LLVM type conversions.
void populateHWToLLVMTypeConversions(mlir::LLVMTypeConverter &converter);

/// Get the HW to LLVM conversion patterns.
void populateHWToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

} // namespace circt

#endif // HWTOLLVM_H
