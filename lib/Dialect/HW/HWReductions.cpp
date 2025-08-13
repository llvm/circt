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
  void matches(Operation *op,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    for (auto result : op->getResults())
      if (!result.use_empty())
        if (isa<IntegerType>(result.getType()))
          addMatch(1, result.getResultNumber());
  }
  LogicalResult rewriteMatches(Operation *op,
                               ArrayRef<uint64_t> indices) override {
    OpBuilder builder(op);
    for (auto idx : indices) {
      auto result = op->getResult(idx);
      auto type = cast<IntegerType>(result.getType());
      auto newOp = hw::ConstantOp::create(builder, result.getLoc(), type, 0);
      result.replaceAllUsesWith(newOp);
    }
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override { return "hw-constantifier"; }
};

/// Remove unused module input ports.
struct ModuleInputPruner : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    symbolTables = std::make_unique<SymbolTableCollection>();
    symbolUsers = std::make_unique<SymbolUserMap>(*symbolTables, op);
  }

  void matches(Operation *op,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    auto mod = dyn_cast<HWModuleLike>(op);
    if (!mod)
      return;
    auto modType = mod.getHWModuleType();
    if (modType.getNumInputs() == 0)
      return;
    auto users = symbolUsers->getUsers(op);
    if (!llvm::all_of(users, [](auto *user) { return isa<InstanceOp>(user); }))
      return;
    auto *block = mod.getBodyBlock();
    for (unsigned idx = 0; idx < modType.getNumInputs(); ++idx)
      if (!block || block->getArgument(idx).use_empty())
        addMatch(1, idx);
  }

  LogicalResult rewriteMatches(Operation *op,
                               ArrayRef<uint64_t> matches) override {
    auto mod = cast<HWMutableModuleLike>(op);

    // Remove the ports from the module.
    SmallVector<unsigned> indexList;
    BitVector indexSet(mod.getNumInputPorts());
    for (auto idx : matches) {
      indexList.push_back(idx);
      indexSet.set(idx);
    }
    llvm::sort(indexList);
    mod.erasePorts(indexList, {});
    if (auto *block = mod.getBodyBlock())
      block->eraseArguments(indexSet);

    // Remove the ports from the instances.
    for (auto *user : symbolUsers->getUsers(op)) {
      auto instOp = cast<InstanceOp>(user);
      SmallVector<Value> newOperands;
      SmallVector<Attribute> newArgNames;
      for (auto [idx, data] : llvm::enumerate(
               llvm::zip(instOp.getInputs(), instOp.getArgNames()))) {
        if (indexSet.test(idx))
          continue;
        auto [operand, argName] = data;
        newOperands.push_back(operand);
        newArgNames.push_back(argName);
      }
      instOp.getInputsMutable().assign(newOperands);
      instOp.setArgNamesAttr(ArrayAttr::get(op->getContext(), newArgNames));
    }

    return success();
  }

  std::string getName() const override { return "hw-module-input-pruner"; }

  std::unique_ptr<SymbolTableCollection> symbolTables;
  std::unique_ptr<SymbolUserMap> symbolUsers;
};

/// Remove unused module output ports.
struct ModuleOutputPruner : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    symbolTables = std::make_unique<SymbolTableCollection>();
    symbolUsers = std::make_unique<SymbolUserMap>(*symbolTables, op);
  }

  void matches(Operation *op,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    auto mod = dyn_cast<HWModuleLike>(op);
    if (!mod)
      return;
    auto modType = mod.getHWModuleType();
    if (modType.getNumOutputs() == 0)
      return;
    auto users = symbolUsers->getUsers(op);
    if (!llvm::all_of(users, [](auto *user) { return isa<InstanceOp>(user); }))
      return;
    for (unsigned idx = 0; idx < modType.getNumOutputs(); ++idx)
      if (llvm::all_of(users, [&](auto *user) {
            return user->getResult(idx).use_empty();
          }))
        addMatch(1, idx);
  }

  LogicalResult rewriteMatches(Operation *op,
                               ArrayRef<uint64_t> matches) override {
    auto mod = cast<HWMutableModuleLike>(op);

    // Remove the ports from the module.
    SmallVector<unsigned> indexList;
    BitVector indexSet(mod.getNumOutputPorts());
    for (auto idx : matches) {
      indexList.push_back(idx);
      indexSet.set(idx);
    }
    llvm::sort(indexList);
    mod.erasePorts({}, indexList);

    // Update the `hw.output` op.
    if (auto *block = mod.getBodyBlock()) {
      auto outputOp = cast<OutputOp>(block->getTerminator());
      SmallVector<Value> newOutputs;
      for (auto [idx, output] : llvm::enumerate(outputOp.getOutputs()))
        if (!indexSet.test(idx))
          newOutputs.push_back(output);
      outputOp.getOutputsMutable().assign(newOutputs);
    }

    // Remove the ports from the instances.
    for (auto *user : symbolUsers->getUsers(op)) {
      OpBuilder builder(user);
      auto instOp = cast<InstanceOp>(user);
      SmallVector<Value> oldResults;
      SmallVector<Type> newResultTypes;
      SmallVector<Attribute> newResultNames;
      for (auto [idx, data] : llvm::enumerate(
               llvm::zip(instOp.getResults(), instOp.getResultNames()))) {
        if (indexSet.test(idx))
          continue;
        auto [result, resultName] = data;
        oldResults.push_back(result);
        newResultTypes.push_back(result.getType());
        newResultNames.push_back(resultName);
      }
      auto newOp = InstanceOp::create(
          builder, instOp.getLoc(), newResultTypes,
          instOp.getInstanceNameAttr(), instOp.getModuleNameAttr(),
          instOp.getInputs(), instOp.getArgNamesAttr(),
          builder.getArrayAttr(newResultNames), instOp.getParametersAttr(),
          instOp.getInnerSymAttr(), instOp.getDoNotPrintAttr());
      for (auto [oldResult, newResult] :
           llvm::zip(oldResults, newOp.getResults()))
        oldResult.replaceAllUsesWith(newResult);
      instOp.erase();
    }

    return success();
  }

  std::string getName() const override { return "hw-module-output-pruner"; }

  std::unique_ptr<SymbolTableCollection> symbolTables;
  std::unique_ptr<SymbolUserMap> symbolUsers;
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
  patterns.add<ModuleOutputPruner, 2>();
  patterns.add<ModuleInputPruner, 2>();
}

void hw::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HWDialect *dialect) {
    dialect->addInterfaces<HWReducePatternDialectInterface>();
  });
}
