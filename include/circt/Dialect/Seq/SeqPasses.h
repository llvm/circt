//===- SeqPasses.h - Seq pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQPASSES_H
#define CIRCT_DIALECT_SEQ_SEQPASSES_H

#include "circt/Dialect/Seq/SeqEnums.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace seq {

#define GEN_PASS_DECL_EXTERNALIZECLOCKGATE
#define GEN_PASS_DECL_HWMEMSIMIMPL
#include "circt/Dialect/Seq/SeqPasses.h.inc"

std::unique_ptr<mlir::Pass> createLowerSeqHLMemPass();
std::unique_ptr<mlir::Pass>
createExternalizeClockGatePass(const ExternalizeClockGateOptions &options = {});
std::unique_ptr<mlir::Pass> createLowerSeqFIFOPass();
std::unique_ptr<mlir::Pass>
createHWMemSimImplPass(const HWMemSimImplOptions &options = {});
std::unique_ptr<mlir::Pass> createLowerSeqShiftRegPass();
std::unique_ptr<mlir::Pass> createRegOfVecToMem();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQPASSES_H
