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

void assumeContractHolds(OpBuilder &builder, IRMapping &mapping,
                         ContractOp contract,
                         SmallVector<Operation *> &opsToClone,
                         std::queue<Operation *> &workList,
                         DenseSet<Operation *> &seen) {
  // Assume it holds by creating symbolic values for the results and inlining
  // the contract ops into the worklist, by default the clone logic replaces
  // contract ops with the assume variant
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
}

void buildOpsToClone(OpBuilder &builder, IRMapping &mapping, Operation *op,
                     SmallVector<Operation *> &opsToClone,
                     std::queue<Operation *> &workList,
                     DenseSet<Operation *> &seen, Operation *parent = nullptr) {
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

LogicalResult cloneFanIn(OpBuilder &builder, Operation *contractToClone,
                         IRMapping &mapping) {
  DenseSet<Operation *> seen;
  SmallVector<Operation *> opsToClone;
  std::queue<Operation *> workList;
  workList.push(contractToClone);

  while (!workList.empty()) {
    auto *currentOp = workList.front();
    workList.pop();

    if (seen.contains(currentOp))
      continue;

    seen.insert(currentOp);

    auto contract = dyn_cast<ContractOp>(*currentOp);
    // If it's not the contract being cloned, assume it holds
    if (contract && (currentOp != contractToClone)) {
      assumeContractHolds(builder, mapping, contract, opsToClone, workList,
                          seen);
      // If a contract is assumed, it's nested ops are inlined into the worklist
      // so no need to walk the body or clone the op
      continue;
    }

    // Clone fan in for operands
    buildOpsToClone(builder, mapping, currentOp, opsToClone, workList, seen);
    // Clone fan in for nested ops
    currentOp->walk([&](Operation *nestedOp) {
      buildOpsToClone(builder, mapping, nestedOp, opsToClone, workList, seen,
                      currentOp);
    });

    // Source contract is cloned externally
    if (currentOp != contractToClone)
      opsToClone.push_back(currentOp);
  }

  // Clone the ops in reverse order so mapping is available for operands
  for (auto it = opsToClone.rbegin(); it != opsToClone.rend(); ++it) {
    if (failed(cloneOp(builder, *it, mapping, true)))
      return failure();
  }
  return success();
}

LogicalResult inlineContract(ContractOp &contract, OpBuilder &builder,
                             bool assumeContract) {
  IRMapping mapping;
  DenseSet<Operation *> seen;
  if (!assumeContract) {
    // Inlining into formal op, need to clone fan in cone for contract
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

  if (assumeContract) {
    // Inlining into hw module, need to update the results using the mapping
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

    if (failed(inlineContract(contract, formalBuilder, false)))
      return failure();
  }

  for (auto contract : contracts) {
    // Inline contract into hwModule
    hwModuleBuilder.setInsertionPointAfter(contract);
    if (failed(inlineContract(contract, hwModuleBuilder, true)))
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
