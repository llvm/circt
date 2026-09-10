//===- Passes.h - DC dialect passes --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_DCPASSES_H
#define CIRCT_DIALECT_DC_DCPASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace dc {

#define GEN_PASS_DECL
#include "circt/Dialect/DC/DCPasses.h.inc"
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createDCDotPrintPass();

std::unique_ptr<mlir::Pass> createDCMaterializeForksSinksPass();
std::unique_ptr<mlir::Pass> createDCDematerializeForksSinksPass();

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/DC/DCPasses.h.inc"

} // namespace dc
} // namespace circt

#endif // CIRCT_DIALECT_DC_DCPASSES_H
