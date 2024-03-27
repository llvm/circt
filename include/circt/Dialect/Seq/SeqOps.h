//===- SeqOps.h - Declare Seq dialect operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Seq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQOPS_H
#define CIRCT_DIALECT_SEQ_SEQOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOpInterfaces.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BuilderUtils.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Seq/Seq.h.inc"

namespace circt {
namespace seq {

// Returns true if the given sequence of addresses match the shape of the given
// HLMemType'd handle.
bool isValidIndexValues(Value hlmemHandle, ValueRange addresses);

/// Helper structure carrying information about FIR memory generated ops.
struct FirMemory {
  size_t numReadPorts;
  size_t numWritePorts;
  size_t numReadWritePorts;
  size_t dataWidth;
  size_t depth;
  size_t maskGran;
  size_t readLatency;
  size_t writeLatency;
  seq::RUW readUnderWrite;
  seq::WUW writeUnderWrite;
  SmallVector<int32_t> writeClockIDs;
  StringRef initFilename;
  bool initIsBinary;
  bool initIsInline;

  FirMemory(hw::HWModuleGeneratedOp op);
};

} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQOPS_H
