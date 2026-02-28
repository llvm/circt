//===- StripContracts.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace verif {
#define GEN_PASS_DEF_STRIPCONTRACTSPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace verif;

namespace {
struct StripContractsPass
    : public verif::impl::StripContractsPassBase<StripContractsPass> {
  void runOnOperation() override {
    getOperation()->walk([](ContractOp op) {
      op->replaceUsesWithIf(op.getInputs(), [&](OpOperand &operand) {
        return operand.getOwner() != op;
      });
      // Prevent removal of self-referential contracts that have other users.
      // Removing them would result in invalid IR.
      for (auto operand : op->getOperands())
        if (operand.getDefiningOp() == op)
          for (auto *user : operand.getUsers())
            if (user != op)
              return;
      op->dropAllReferences();
      op->erase();
    });
  }
};
} // namespace
