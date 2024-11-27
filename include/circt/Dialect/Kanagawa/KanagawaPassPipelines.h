//===- KanagawaPassPipelines.h - Kanagawa pass pipelines --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_KANAGAWAPASSPIPELINES_H
#define CIRCT_DIALECT_KANAGAWA_KANAGAWAPASSPIPELINES_H

#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <optional>

namespace circt {
namespace kanagawa {

// Loads a pass pipeline to transform low-level Kanagawa constructs.
void loadKanagawaLowLevelPassPipeline(mlir::PassManager &pm);

// Loads a pass pipeline to transform high-level Kanagawa constructs.
void loadKanagawaHighLevelPassPipeline(mlir::PassManager &pm);

} // namespace kanagawa
} // namespace circt

#endif // CIRCT_DIALECT_KANAGAWA_KANAGAWAPASSPIPELINES_H
