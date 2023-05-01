//===- SeqToSV.h - Seq to SV lowering -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_SEQTOSV_H_
#define CIRCT_CONVERSION_SEQTOSV_H_

#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
class ModuleOp;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_LOWERSEQTOSV
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerSeqToSVPass(const LowerSeqToSVOptions &options = {});

} // namespace circt

#endif // CIRCT_CONVERSION_SEQTOSV_H_
