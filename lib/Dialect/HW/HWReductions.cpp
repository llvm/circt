//===- HWReductions.cpp - Reduction patterns for the HW dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWReductions.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Reduce/ReductionUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hw-reductions"

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Utility to track the transitive size of modules.
struct ModuleSizeCache {
  void clear() { moduleSizes.clear(); }

  uint64_t getModuleSize(HWModuleLike module,
                         hw::InstanceGraph &instanceGraph) {
    if (auto it = moduleSizes.find(module); it != moduleSizes.end())
      return it->second;
    uint64_t size = 1;
    module->walk([&](Operation *op) {
      size += 1;
      if (auto instOp = dyn_cast<HWInstanceLike>(op)) {
        for (auto moduleName : instOp.getReferencedModuleNamesAttr()) {
          auto *node = instanceGraph.lookup(cast<StringAttr>(moduleName));
          if (auto instModule =
                  dyn_cast_or_null<hw::HWModuleLike>(*node->getModule()))
            size += getModuleSize(instModule, instanceGraph);
        }
      }
    });
    moduleSizes.insert({module, size});
    return size;
  }

private:
  llvm::DenseMap<Operation *, uint64_t> moduleSizes;
};

//===----------------------------------------------------------------------===//
// Reduction patterns
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that maps `hw.module` to `hw.module.extern`.
struct ModuleExternalizer : public OpReduction<HWModuleOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    instanceGraph = std::make_unique<InstanceGraph>(op);
    moduleSizes.clear();
  }

  uint64_t match(HWModuleOp op) override {
    return moduleSizes.getModuleSize(op, *instanceGraph);
  }

  LogicalResult rewrite(HWModuleOp op) override {
    OpBuilder builder(op);
    HWModuleExternOp::create(builder, op->getLoc(), op.getModuleNameAttr(),
                             op.getPortList(), StringRef(), op.getParameters());
    op->erase();
    return success();
  }

  std::string getName() const override { return "hw-module-externalizer"; }

  std::unique_ptr<InstanceGraph> instanceGraph;
  ModuleSizeCache moduleSizes;
};

/// A sample reduction pattern that replaces all uses of an operation with one
/// of its operands. This can help pruning large parts of the expression tree
/// rapidly.
template <unsigned OpNum>
struct HWOperandForwarder : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2 ||
        OpNum >= op->getNumOperands())
      return 0;
    auto resultTy = dyn_cast<IntegerType>(op->getResult(0).getType());
    auto opTy = dyn_cast<IntegerType>(op->getOperand(OpNum).getType());
    return resultTy && opTy && resultTy == opTy &&
           op->getResult(0) != op->getOperand(OpNum);
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    ImplicitLocOpBuilder builder(op->getLoc(), op);
    auto result = op->getResult(0);
    auto operand = op->getOperand(OpNum);
    LLVM_DEBUG(llvm::dbgs()
               << "Forwarding " << operand << " in " << *op << "\n");
    result.replaceAllUsesWith(operand);
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override {
    return ("hw-operand" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that replaces integer operations with a constant
/// zero of their type.
struct HWConstantifier : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() == 0 || op->getNumOperands() == 0)
      return 0;
    return llvm::all_of(op->getResults(), [](Value result) {
      return isa<IntegerType>(result.getType());
    });
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    OpBuilder builder(op);
    for (auto result : op->getResults()) {
      auto type = cast<IntegerType>(result.getType());
      auto newOp = hw::ConstantOp::create(builder, op->getLoc(), type, 0);
      result.replaceAllUsesWith(newOp);
    }
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override { return "hw-constantifier"; }
};

/// Remove the first or last output of the top-level module depending on the
/// 'Front' template parameter.
template <bool Front>
struct ModuleOutputPruner : public OpReduction<HWModuleOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    useEmpty.clear();

    SymbolTableCollection table;
    SymbolUserMap users(table, op);
    for (auto module : op.getOps<HWModuleOp>())
      if (users.useEmpty(module))
        useEmpty.insert(module);
  }

  uint64_t match(HWModuleOp op) override {
    return op.getNumOutputPorts() != 0 && useEmpty.contains(op);
  }

  LogicalResult rewrite(HWModuleOp op) override {
    Operation *terminator = op.getBody().front().getTerminator();
    auto operands = terminator->getOperands();
    ValueRange newOutputs = operands.drop_back();
    unsigned portToErase = op.getNumOutputPorts() - 1;
    if (Front) {
      newOutputs = operands.drop_front();
      portToErase = 0;
    }

    terminator->setOperands(newOutputs);
    op.erasePorts({}, {portToErase});

    return success();
  }

  std::string getName() const override {
    return Front ? "hw-module-output-pruner-front"
                 : "hw-module-output-pruner-back";
  }

  DenseSet<HWModuleOp> useEmpty;
};

/// Remove all input ports of the top-level module that have no users
struct ModuleInputPruner : public OpReduction<HWModuleOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    useEmpty.clear();

    SymbolTableCollection table;
    SymbolUserMap users(table, op);
    for (auto module : op.getOps<HWModuleOp>())
      if (users.useEmpty(module))
        useEmpty.insert(module);
  }

  uint64_t match(HWModuleOp op) override { return useEmpty.contains(op); }

  LogicalResult rewrite(HWModuleOp op) override {
    SmallVector<unsigned> inputsToErase;
    BitVector toErase(op.getNumPorts());
    for (auto [i, arg] : llvm::enumerate(op.getBody().getArguments())) {
      if (arg.use_empty()) {
        toErase.set(i);
        inputsToErase.push_back(i);
      }
    }

    op.erasePorts(inputsToErase, {});
    op.getBodyBlock()->eraseArguments(toErase);

    return success();
  }

  std::string getName() const override { return "hw-module-input-pruner"; }

  DenseSet<HWModuleOp> useEmpty;
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

void HWReducePatternDialectInterface::populateReducePatterns(
    circt::ReducePatternSet &patterns) const {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // assigned reasonable benefit indicators (higher benefit patterns are
  // prioritized). For example, things that can knock out entire modules while
  // being cheap should be tried first (and thus have higher benefit), before
  // trying to tweak operands of individual arithmetic ops.
  patterns.add<ModuleExternalizer, 6>();
  patterns.add<HWConstantifier, 5>();
  patterns.add<HWOperandForwarder<0>, 4>();
  patterns.add<HWOperandForwarder<1>, 3>();
  patterns.add<HWOperandForwarder<2>, 2>();
  patterns.add<ModuleOutputPruner<true>, 2>();
  patterns.add<ModuleOutputPruner<false>, 2>();
  patterns.add<ModuleInputPruner, 2>();
}

void hw::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HWDialect *dialect) {
    dialect->addInterfaces<HWReducePatternDialectInterface>();
  });
}
