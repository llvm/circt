//===- LowerContracts.cpp - Formal Preparations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower contracts into verif.formal tests.
//
//===----------------------------------------------------------------------===//

#include <queue>

#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "mlir/IR/IRMapping.h"

using namespace circt;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_LOWERCONTRACTSPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace verif;
using namespace hw;

namespace {
struct LowerContractsPass
    : verif::impl::LowerContractsPassBase<LowerContractsPass> {
  void runOnOperation() override;
};

Operation *replaceContractOp(OpBuilder &builder, RequireLike op,
                             IRMapping &mapping, bool assumeContract) {
  StringAttr labelAttr = op.getLabelAttr();

  Value enableValue;
  if (auto enable = op.getEnable())
    enable = mapping.lookup(enable);

  auto loc = op.getLoc();
  auto property = mapping.lookup(op.getProperty());

  if ((isa<EnsureOp>(op) && !assumeContract) ||
      (isa<RequireOp>(op) && assumeContract))
    return builder.create<AssertOp>(loc, property, enableValue, labelAttr);
  if ((isa<EnsureOp>(op) && assumeContract) ||
      (isa<RequireOp>(op) && !assumeContract))
    return builder.create<AssumeOp>(loc, property, enableValue, labelAttr);
  return nullptr;
}

LogicalResult cloneOp(OpBuilder &builder, Operation *opToClone,
                      IRMapping &mapping, bool assumeContract) {
  Operation *clonedOp;
  // Replace ensure/require ops, otherwise clone
  if (auto requireLike = dyn_cast<RequireLike>(*opToClone)) {
    clonedOp = replaceContractOp(builder, requireLike, mapping, assumeContract);
    if (!clonedOp) {
      return failure();
    }
  } else {
    clonedOp = builder.clone(*opToClone, mapping);
  }

  // Add mappings for results
  for (auto [x, y] :
       llvm::zip(opToClone->getResults(), clonedOp->getResults())) {
    mapping.map(x, y);
  }
  return llvm::success();
}

void buildOpsToClone(OpBuilder &builder, IRMapping &mapping, Operation *op,
                     SmallVector<Operation *> &opsToClone,
                     std::queue<Operation *> &workList,
                     DenseSet<Operation *> &seen, bool assumeContract,
                     Operation *parent = nullptr) {
  auto contract = dyn_cast<ContractOp>(*op);
  if (contract && assumeContract) {
    // Assume it holds
    for (auto result : contract.getResults()) {
      auto sym =
          builder.create<SymbolicValueOp>(result.getLoc(), result.getType());
      mapping.map(result, sym);
    }
    auto &contractOps = contract.getBody().front().getOperations();
    for (auto it = contractOps.rbegin(); it != contractOps.rend(); ++it) {
      if (!seen.contains(&*it)) {
        workList.push(&*it);
      }
    }
    return;
  }
  for (auto operand : op->getOperands()) {
    if (mapping.contains(operand))
      continue;
    // Don't need to clone if defining op is within parent op
    if (parent && parent->isAncestor(operand.getParentBlock()->getParentOp()))
      continue;
    if (auto *definingOp = operand.getDefiningOp()) {
      if (!seen.contains(definingOp)) {
        workList.push(definingOp);
      }
    } else {
      // Create symbolic values for arguments
      auto sym = builder.create<verif::SymbolicValueOp>(operand.getLoc(),
                                                        operand.getType());
      mapping.map(operand, sym);
    }
  }
}

LogicalResult cloneFanIn(OpBuilder &builder, Operation *opToClone,
                         IRMapping &mapping) {
  DenseSet<Operation *> seen;
  SmallVector<Operation *> opsToClone;
  std::queue<Operation *> workList;
  workList.push(opToClone);
  while (!workList.empty()) {
    auto *currentOp = workList.front();
    workList.pop();
    if (seen.contains(currentOp))
      continue;
    seen.insert(currentOp);
    auto contract = dyn_cast<ContractOp>(*currentOp);
    bool assumeContract = contract && (currentOp != opToClone);

    buildOpsToClone(builder, mapping, currentOp, opsToClone, workList, seen,
                    assumeContract);
    if (contract && assumeContract)
      continue;
    currentOp->walk([&](Operation *nestedOp) {
      buildOpsToClone(builder, mapping, nestedOp, opsToClone, workList, seen,
                      assumeContract, currentOp);
    });
    // Contracts are cloned differently
    if (!contract)
      opsToClone.push_back(currentOp);
  }

  for (auto it = opsToClone.rbegin(); it != opsToClone.rend(); ++it) {
    if (failed(cloneOp(builder, *it, mapping, true)))
      return failure();
  }
  return success();
}

LogicalResult inlineContract(ContractOp &contract, OpBuilder &builder,
                             bool assumeContract, bool shouldCloneFanIn,
                             bool replaceResults) {
  IRMapping mapping;
  DenseSet<Operation *> seen;
  if (shouldCloneFanIn) {
    // Clone fan in cone for contract
    if (failed(cloneFanIn(builder, contract, mapping)))
      return failure();
  }

  if (assumeContract) {
    // Create symbolic values for results
    for (auto result : contract.getResults()) {
      auto sym =
          builder.create<SymbolicValueOp>(result.getLoc(), result.getType());
      mapping.map(result, sym);
    }
  } else {
    // Map results by looking up the input mappings
    for (auto [result, input] :
         llvm::zip(contract.getResults(), contract.getInputs())) {
      mapping.map(result, mapping.lookup(input));
    }
  }

  // Clone body of the contract
  for (auto &op : contract.getBody().front().getOperations()) {
    if (failed(cloneOp(builder, &op, mapping, assumeContract)))
      return failure();
  }

  if (replaceResults) {
    for (auto result : contract.getResults())
      result.replaceAllUsesWith(mapping.lookup(result));
  }
  return success();
}

LogicalResult runOnHWModule(HWModuleOp hwModule, ModuleOp mlirModule) {
  OpBuilder mlirModuleBuilder(mlirModule);
  mlirModuleBuilder.setInsertionPointAfter(hwModule);
  OpBuilder hwModuleBuilder(hwModule);

  // Collect contract ops
  SmallVector<ContractOp> contracts;
  hwModule.walk([&](ContractOp op) { contracts.push_back(op); });

  for (unsigned i = 0; i < contracts.size(); i++) {
    auto contract = contracts[i];

    // Create verif.formal op
    auto name = mlirModuleBuilder.getStringAttr(
        hwModule.getNameAttr().getValue() + "_CheckContract_" + Twine(i));
    auto formalOp = mlirModuleBuilder.create<verif::FormalOp>(
        contract.getLoc(), name, mlirModuleBuilder.getDictionaryAttr({}));

    // Fill in verif.formal body
    OpBuilder formalBuilder(formalOp);
    formalBuilder.createBlock(&formalOp.getBody());

    if (failed(inlineContract(contract, formalBuilder, false, true, false)))
      return failure();
  }

  for (auto contract : contracts) {
    // Inline contract into hwModule
    hwModuleBuilder.setInsertionPointAfter(contract);
    if (failed(inlineContract(contract, hwModuleBuilder, true, false, true)))
      return failure();
    contract.erase();
  }

  return success();
}
} // namespace

void LowerContractsPass::runOnOperation() {
  auto mlirModule = getOperation();
  for (auto module : mlirModule.getOps<HWModuleOp>())
    if (failed(runOnHWModule(module, mlirModule)))
      signalPassFailure();
}
