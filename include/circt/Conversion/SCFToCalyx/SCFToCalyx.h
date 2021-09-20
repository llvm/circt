//===- SCFToCalyx.h - SCF to Calyx pass entry point -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the SCFToCalyx pass
// constructor.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_SCFTOCALYX_SCFTOCALYX_H
#define CIRCT_CONVERSION_SCFTOCALYX_SCFTOCALYX_H

#include "circt/Support/LLVM.h"
#include "mlir/Dialect/SCF/SCF.h"
#include <memory>

namespace circt {

/// Create an SCF to Calyx conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createSCFToCalyxPass();

} // namespace circt

#endif // CIRCT_CONVERSION_SCFTOCALYX_SCFTOCALYX_H
