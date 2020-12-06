//===- FIRRTLToRTL.h - FIRRTL to RTL conversion pass ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the FIRRTL dialect to
// RTL and SV dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FIRRTLTORTL_FIRRTLTORTL_H
#define CIRCT_CONVERSION_FIRRTLTORTL_FIRRTLTORTL_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace firrtl {

std::unique_ptr<mlir::Pass> createLowerFIRRTLToRTLModulePass();
std::unique_ptr<mlir::Pass> createLowerFIRRTLToRTLPass();

} // namespace firrtl
} // namespace circt

#endif // CIRCT_CONVERSION_FIRRTLTORTL_FIRRTLTORTL_H
