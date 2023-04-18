//===- DCDialect.cpp - DC dialect implementation --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCTypes.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace circt;
using namespace dc;

namespace {
/// This class defines the interface for handling inlining with DC operations.
struct DCInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  // All DC ops are inlineable.
  bool isLegalToInline(Operation *, Operation *, bool) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only "dc.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
} // end anonymous namespace

void DCDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/DC/DC.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<DCInlinerInterface>();
}

#include "circt/Dialect/DC/DCDialect.cpp.inc"
