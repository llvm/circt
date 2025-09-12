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

namespace {

/// A sample reduction pattern that removes operations which either produce no
/// results or their results have no users.
struct OperationPruner : public Reduction {
  void beforeReduction(mlir::ModuleOp module) override {
    userMap = std::make_unique<SymbolUserMap>(table, module);
  }
  uint64_t match(Operation *op) override {
    if (op->hasTrait<OpTrait::IsTerminator>())
      return false;

    return !isa<ModuleOp>(op) &&
           (op->getNumResults() == 0 || op->use_empty()) &&
           userMap->useEmpty(op);
  }
  LogicalResult rewrite(Operation *op) override {
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override { return "operation-pruner"; }

  SymbolTableCollection table;
  std::unique_ptr<SymbolUserMap> userMap;
};

/// Remove unused symbol ops.
struct UnusedSymbolPruner : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    symbolTables = std::make_unique<SymbolTableCollection>();
    symbolUsers = std::make_unique<SymbolUserMap>(*symbolTables, op);
  }

  uint64_t match(Operation *op) override {
    return isa<SymbolOpInterface>(op) && symbolUsers->useEmpty(op);
  }

  LogicalResult rewrite(Operation *op) override {
    op->erase();
    return success();
  }

  std::string getName() const override { return "unused-symbol-pruner"; }

  std::unique_ptr<SymbolTableCollection> symbolTables;
  std::unique_ptr<SymbolUserMap> symbolUsers;
};

} // namespace

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  return createCanonicalizerPass(config);
}

void circt::populateGenericReducePatterns(MLIRContext *context,
                                          ReducePatternSet &patterns) {
  patterns.add<PassReduction, 103>(context, createSymbolDCEPass());
  patterns.add<PassReduction, 102>(context, createCSEPass());
  patterns.add<PassReduction, 101>(context, createSimpleCanonicalizerPass());
  patterns.add<UnusedSymbolPruner, 100>();
  patterns.add<OperationPruner, 1>();
}
