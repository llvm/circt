//===- OMPasses.h - OM dialect passes -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMPASSES_H
#define CIRCT_DIALECT_OM_OMPASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace om {

std::unique_ptr<mlir::Pass> createOMLinkModulesPass();
std::unique_ptr<mlir::Pass> createFreezePathsPass(
    std::function<StringAttr(Operation *)> getOpNameFallback = {});
std::unique_ptr<mlir::Pass> createVerifyObjectFieldsPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/OM/OMPasses.h.inc"

} // namespace om
} // namespace circt

#endif // CIRCT_DIALECT_OM_OMPASSES_H
