//===- LLHDToLLVM.h - Registration of LLHD to LLVM patterns -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the LLHD to LLVM operation
// conversion pattern registration.
//
//===----------------------------------------------------------------------===//

#ifndef LLHDTOLLVM_H
#define LLHDTOLLVM_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class LLVMConversionTarget;
} // namespace mlir

namespace circt {

/// Register conversion patterns and illegal ops for a partial pre-pass.
void setupPartialLLHDPrePass(mlir::LLVMTypeConverter &converter,
                             RewritePatternSet &patterns,
                             mlir::LLVMConversionTarget &target);

/// Get the LLHD to LLVM type conversions.
void populateLLHDToLLVMTypeConversions(mlir::LLVMTypeConverter &converter);

/// Get the LLHD to LLVM conversion patterns.
void populateLLHDToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          size_t &sigCounter,
                                          size_t &regCounter);

} // namespace circt

#endif // LLHDTOLLVM_H
