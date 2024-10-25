//===- Evaluator.h - RTG Evaluator ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_RTGTOOL_EVALUATOR_H
#define CIRCT_RTGTOOL_EVALUATOR_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace circt {
namespace rtg {

void evaluate(mlir::ModuleOp moduleOp, const std::string &filename);
mlir::OwningOpRef<mlir::ModuleOp> evaluate(const std::string &packageName,
                                           const std::string &filename);

} // namespace rtg
} // namespace circt

#endif // CIRCT_RTGTOOL_EVALUATOR_H
