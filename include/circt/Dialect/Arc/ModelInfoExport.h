//===- ModelInfoExport.h - Exports model info to JSON format --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Register the MLIR translation to export model info to JSON format.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_MODELINFOEXPORT_H
#define CIRCT_DIALECT_ARC_MODELINFOEXPORT_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace arc {

/// Collects and exports Arc model info to JSON.
mlir::LogicalResult collectAndExportModelInfo(mlir::ModuleOp module,
                                              llvm::raw_ostream &os);

/// Registers CIRCT translation from Arc to JSON model info.
void registerArcModelInfoTranslation();

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_MODELINFOEXPORT_H
