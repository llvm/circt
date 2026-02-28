//===- LTLToCore.h - LTL to Core conversion pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will convert the LTL and Verif
// operations to Core operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_LTLTOCORE_H
#define CIRCT_CONVERSION_LTLTOCORE_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {

#define GEN_PASS_DECL_LOWERLTLTOCORE
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLowerLTLToCorePass();

} // namespace circt

#endif // CIRCT_CONVERSION_LTLTOCORE_H
