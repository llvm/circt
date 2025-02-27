//===- BLIFEmitter.h - BLIF dialect to .blif emitter ------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the .blif file emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_BLIF_BLIFEMITTER_H
#define CIRCT_DIALECT_BLIF_BLIFEMITTER_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace blif {

mlir::LogicalResult exportBLIFFile(mlir::ModuleOp module,
                                   llvm::raw_ostream &os);

void registerToBLIFFileTranslation();

} // namespace blif
} // namespace circt

#endif // CIRCT_DIALECT_BLIF_BLIFEMITTER_H
