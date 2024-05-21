//===- FIRParserAsserts.cpp - Printf-encoded assert handling --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements handling of printf-encoded verification operations
// embedded in when blocks.
//
// While no longer supported, this file retains enough to classify printf
// operations so that we may error on their unsupported use.
//
//===----------------------------------------------------------------------===//

#include "FIRAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

/// Chisel has a tendency to emit complex assert/assume/cover statements encoded
/// as print operations with special formatting and metadata embedded in the
/// message literal.  One variant looks like:
///
///     when invCond:
///       printf(clock, UInt<1>(1), "...[verif-library-assert]...")
///       stop(clock, UInt<1>(1), 1)
///
/// Depending on the nature the verification operation, the `stop` may be
/// optional. The Scala implementation simply removes all `stop`s that have the
/// same condition as the printf.
///
/// This is no longer supported.
/// This method retains enough support to accurately detect when this is used
/// and returns whether this is a recognized (legacy) printf+when-encoded verif.
bool circt::firrtl::isRecognizedPrintfEncodedVerif(PrintFOp printOp) {
  auto whenStmt = dyn_cast<WhenOp>(printOp->getParentOp());

  // If the parent of printOp is not when, printOp doesn't encode a
  // verification.
  if (!whenStmt)
    return false;

  // The when blocks we're interested in don't have an else region.
  if (whenStmt.hasElseRegion())
    return false;

  // The when blocks we're interested in contain a `PrintFOp` and an optional
  // `StopOp` with the same clock and condition as the print.
  Block &thenBlock = whenStmt.getThenBlock();
  auto opIt = std::next(printOp->getIterator());
  auto opEnd = thenBlock.end();

  // Detect if we're dealing with a verification statement.
  auto fmt = printOp.getFormatString();
  if (!(fmt.contains("[verif-library-assert]") ||
        fmt.contains("[verif-library-assume]") ||
        fmt.contains("[verif-library-cover]") || fmt.starts_with("assert:") ||
        fmt.starts_with("assume:") || fmt.starts_with("cover:") ||
        fmt.starts_with("assertNotX:") || fmt.starts_with("Assertion failed")))
    return false;

  // optional `stop(clock, enable, ...)`
  //
  // NOTE: This code is retained as-is from when these were fully supported,
  // to maintain the classification portion.  Previously this code would then
  // delete the "stop" operation, which it no longer does.
  if (opIt != opEnd) {
    auto stopOp = dyn_cast<StopOp>(*opIt++);
    if (!stopOp || opIt != opEnd || stopOp.getClock() != printOp.getClock() ||
        stopOp.getCond() != printOp.getCond())
      return false;
    // (This code used to erase the stop operation here)
  }
  return true;
}
