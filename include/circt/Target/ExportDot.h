//===- ExportDot.h - Dot Exporter ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Dot emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_EXPORTDOT_H
#define CIRCT_TARGET_EXPORTDOT_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace ExportDot {


/// Run the ExportDot pass.
LogicalResult
exportDot(Operation *module, llvm::raw_ostream &os);

/// Register the ExportDot pass.
void registerExportDotTranslation();

} // namespace ExportDot
} // namespace circt

#endif // CIRCT_TARGET_EXPORTDot_H
