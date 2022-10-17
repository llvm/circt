//===- PipelineToCalyx.h - Pipeline to Calyx pass entry point -----------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the PipelineToCalyx pass
// constructor.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PIPELINETOCALYX_H
#define CIRCT_CONVERSION_PIPELINETOCALYX_H

#include "circt/Support/LLVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <memory>

namespace circt {

/// Create a Pipeline to Calyx conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createPipelineToCalyxPass();

} // namespace circt

#endif // CIRCT_CONVERSION_PIPELINETOCALYX_H
