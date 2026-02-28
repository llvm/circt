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

#ifndef CIRCT_CONVERSION_SEQTOSV_H
#define CIRCT_CONVERSION_SEQTOSV_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

#define GEN_PASS_DECL_LOWERFIRMEM
#define GEN_PASS_DECL_LOWERSEQTOSV
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass>
createLowerSeqToSVPass(const LowerSeqToSVOptions &options = {});
std::unique_ptr<mlir::Pass> createLowerFirMemPass();
std::unique_ptr<mlir::Pass> createLowerSeqFIRRTLInitToSV();

} // namespace circt

#endif // CIRCT_CONVERSION_SEQTOSV_H
