//===- GenericReductions.cpp - Generic Reduction patterns -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Reduce/GenericReductions.h"
#include "circt/Reduce/ReductionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Reduction Patterns
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that removes operations which either produce no
/// results or their results have no users.
struct OperationPruner : public Reduction {
  uint64_t match(Operation *op) override {
    if (!isa<ModuleOp>(op) && (op->getNumResults() == 0 || op->use_empty()) &&
        !op->hasAttr(SymbolTable::getSymbolAttrName()))
      return true;

    auto *symbolTableOp = SymbolTable::getNearestSymbolTable(op);
    SymbolUserMap userMap(table, symbolTableOp);

    return !isa<ModuleOp>(op) &&
           (op->getNumResults() == 0 || op->use_empty()) &&
           userMap.useEmpty(op);
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override { return "operation-pruner"; }

  SymbolTableCollection table;
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return createCanonicalizerPass(config);
}

void circt::populateGenericReducePatterns(MLIRContext *context,
                                          ReducePatternSet &patterns) {
  patterns.add<PassReduction, 3>(context, createCSEPass());
  patterns.add<PassReduction, 2>(context, createSimpleCanonicalizerPass());
  patterns.add<OperationPruner, 1>();
}
