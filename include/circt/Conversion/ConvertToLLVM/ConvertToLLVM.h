//===- ConvertToLLVM.h - Convert to LLVM pass entry point -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the ConvertToLLVM pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_CONVERTTOLLVM_CONVERTTOLLVM_H
#define CIRCT_CONVERSION_CONVERTTOLLVM_CONVERTTOLLVM_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

/// Create an LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass();

} // namespace circt

#endif // CIRCT_CONVERSION_CONVERTTOLLVM_CONVERTTOLLVM_H
