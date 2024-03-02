//===- SeqToSV.h - SV conversion for seq ops ----------------===-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which lower `seq` to `sv` and `hw`.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_EXPORTYOSYS_H
#define CIRCT_CONVERSION_EXPORTYOSYS_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_EXPORTYOSYS
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createExportYosys();

} // namespace circt

#endif // CIRCT_CONVERSION_SEQTOSV_H