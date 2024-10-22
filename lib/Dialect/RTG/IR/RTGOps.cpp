//===- RTGOps.cpp - Implement the RTG operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTG ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include <numeric>

using namespace mlir;
using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// LabelDeclOp
//===----------------------------------------------------------------------===//

APInt LabelDeclOp::getBinary() {
  assert(false && "must not be used for label resources");
  return APInt();
}

void LabelDeclOp::printAssembly(llvm::raw_ostream &stream) {
  // TODO: perform substitutions
  stream << getFormatString();
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

LogicalResult SequenceOp::verifyRegions() {
  if (getBody()->getArgumentTypes() != getResult().getType().getArgTypes())
    return emitOpError("argument types must match sequence type");

  return success();
}

//===----------------------------------------------------------------------===//
// SelectRandomOp
//===----------------------------------------------------------------------===//

LogicalResult SelectRandomOp::verify() {
  if (getSequences().size() != getSequenceArgs().size())
    return emitOpError("number of sequences and sequence arg lists must match");

  if (getSequences().size() != getRatios().size())
    return emitOpError("number of sequences and ratios must match");

  for (auto [seq, args] : llvm::zip(getSequences(), getSequenceArgs()))
    if (TypeRange(cast<SequenceType>(seq.getType()).getArgTypes()) !=
        args.getTypes())
      return emitOpError(
          "sequence argument types do not match sequence requirements");

  return success();
}


//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#include "circt/Dialect/RTG/IR/RTGInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/RTG/IR/RTG.cpp.inc"
