//===- Passes.h - FIRRTL pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_PASSES_H
#define CIRCT_DIALECT_FIRRTL_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace firrtl {

std::unique_ptr<mlir::Pass> createLowerFIRRTLToRTLModulePass();
std::unique_ptr<mlir::Pass> createLowerFIRRTLToRTLPass();
std::unique_ptr<mlir::Pass> createLowerFIRRTLTypesPass();

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_PASSES_H
