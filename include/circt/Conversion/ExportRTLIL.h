//===- ExportRTLIL.h - Export core dialects to Yosys RTLIL --===-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which lower core dialects to RTLIL in-memory IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_EXPORTRTLIL_H
#define CIRCT_CONVERSION_EXPORTRTLIL_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <string>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_EXPORTYOSYS
#define GEN_PASS_DECL_EXPORTYOSYSPARALLEL
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createYosysOptimizer();
std::unique_ptr<mlir::Pass> createYosysOptimizerParallel();

/// Register the `(import|export)-rtlil` MLIR translation.
void registerRTLILImport();
void registerRTLILExport();
void registerRTLILTranslation();

} // namespace circt

#endif // CIRCT_CONVERSION_EXPORTRTLIL_H
