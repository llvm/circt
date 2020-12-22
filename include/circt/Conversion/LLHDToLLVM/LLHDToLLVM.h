//===- LLHDToLLVM.h - LLHD to LLVM pass entry point -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the LLHDToLLVM pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H
#define CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H

#include <memory>

namespace mlir {
class ModuleOp;
class LLVMTypeConverter;
class OwningRewritePatternList;
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
namespace llhd {
using namespace mlir;

/// Get the LLHD to LLVM conversion patterns.
void populateLLHDToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns,
                                          size_t &sigCounter,
                                          size_t &regCounter);

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertLLHDToLLVMPass();

void initLLHDToLLVMPass();
} // namespace llhd
} // namespace circt

#endif // CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H
