//===- FIRRTLToLLHD.h - LLHD to LLVM pass entry point -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the FIRRTLToLLHD pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FIRRTLTOLLHD_FIRRTLTOLLHD_H_
#define CIRCT_CONVERSION_FIRRTLTOLLHD_FIRRTLTOLLHD_H_

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
namespace llhd {
using namespace mlir;
std::unique_ptr<OperationPass<ModuleOp>> createConvertFIRRTLToLLHDPass();
void registerFIRRTLToLLHDPasses();
} // namespace llhd
} // namespace circt

#endif // CIRCT_CONVERSION_FIRRTLTOLLHD_FIRRTLTOLLHD_H_
