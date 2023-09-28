//===- IbisPassPipelines.h - Ibis pass pipelines -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISPASSPIPELINES_H
#define CIRCT_DIALECT_IBIS_IBISPASSPIPELINES_H

#include "circt/Dialect/Ibis/IbisPasses.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <optional>

namespace circt {
namespace ibis {

// Loads a pass pipeline to transform low-level Ibis constructs.
void loadIbisLowLevelPassPipeline(mlir::PassManager &pm);

// Loads a pass pipeline to transform high-level Ibis constructs.
void loadIbisHighLevelPassPipeline(mlir::PassManager &pm);

} // namespace ibis
} // namespace circt

#endif // CIRCT_DIALECT_IBIS_IBISPASSPIPELINES_H
