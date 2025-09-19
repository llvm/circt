//===- GenericReductions.cpp - Generic Reduction patterns -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Reduce/GenericReductions.h"
#include "circt/Reduce/ReductionUtils.h"
#include "mlir/IR/SymbolTable.h"
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
  void beforeReduction(mlir::ModuleOp op) override {
    symbolUses = reduce::InnerSymbolUses(op);
  }
  uint64_t match(Operation *op) override {
    if (op->hasTrait<OpTrait::IsTerminator>())
      return 0;
    if (isa<ModuleOp>(op))
      return 0;
    if (op->getNumResults() > 0 && !op->use_empty())
      return 0;
    if (symbolUses.hasRef(op))
      return 0;
    return 1;
  }
  LogicalResult rewrite(Operation *op) override {
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override { return "operation-pruner"; }

  reduce::InnerSymbolUses symbolUses;
};

/// Remove unused symbol ops.
struct UnusedSymbolPruner : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    symbolUses = reduce::InnerSymbolUses(op);
  }

  uint64_t match(Operation *op) override {
    if (op->hasAttr(SymbolTable::getSymbolAttrName()))
      if (!symbolUses.hasRef(op))
        return 1;
    return 0;
  }

  LogicalResult rewrite(Operation *op) override {
    op->erase();
    return success();
  }

  std::string getName() const override { return "unused-symbol-pruner"; }

  reduce::InnerSymbolUses symbolUses;
};

/// A reduction pattern that changes symbol visibility from "public" to
/// "private".
struct MakeSymbolsPrivate : public Reduction {
  uint64_t match(Operation *op) override {
    if (!op->getParentOp() || !isa<SymbolOpInterface>(op))
      return 0;
    return SymbolTable::getSymbolVisibility(op) !=
           SymbolTable::Visibility::Private;
  }

  LogicalResult rewrite(Operation *op) override {
    // Change the visibility from public to private
    SymbolTable::setSymbolVisibility(op, SymbolTable::Visibility::Private);
    return success();
  }

  std::string getName() const override { return "make-symbols-private"; }
  bool acceptSizeIncrease() const override { return true; }
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
  patterns.add<PassReduction, 102>(context, createSimpleCanonicalizerPass());
  patterns.add<PassReduction, 101>(context, createCSEPass());
  patterns.add<MakeSymbolsPrivate, 100>();
  patterns.add<UnusedSymbolPruner, 99>();
  patterns.add<OperationPruner, 1>();
}
