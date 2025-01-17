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
  StringAttr labelAttr;
  if (auto label = op.getLabel())
    labelAttr = builder.getStringAttr(label.value());

  Value enableValue;
  if (auto enable = op.getEnable())
    enable = mapping.lookup(enable);

  auto loc = op.getLoc();
  auto property = mapping.lookup(op.getProperty());

  if ((isa<EnsureOp>(op) && !assumeContract) || (isa<RequireOp>(op) && assumeContract))
    return builder.create<AssertOp>(loc, property, enableValue, labelAttr);
  if ((isa<EnsureOp>(op) && assumeContract) || (isa<RequireOp>(op) && !assumeContract))
    return builder.create<AssumeOp>(loc, property, enableValue, labelAttr);
  return nullptr;
}

LogicalResult cloneFanIn(OpBuilder &builder, Operation *opToClone,
                         IRMapping &mapping, DenseSet<Operation *> &seen,
                         bool assumeContract);

LogicalResult cloneContractBody(ContractOp &contract, OpBuilder &builder,
                                IRMapping &mapping, DenseSet<Operation *> &seen,
                                bool assumeContract) {
  for (auto &op : contract.getBody().front().getOperations()) {
    if (failed(cloneFanIn(builder, &op, mapping, seen, assumeContract)))
      return failure();
  }
  return llvm::success();
}

LogicalResult inlineContract(ContractOp &contract, OpBuilder &builder,
                             IRMapping &mapping, DenseSet<Operation *> &seen) {
  for (auto result : contract.getResults()) {
    auto sym =
        builder.create<SymbolicValueOp>(result.getLoc(), result.getType());
    mapping.map(result, sym);
  }
  return cloneContractBody(contract, builder, mapping, seen, true);
}

LogicalResult cloneFanIn(OpBuilder &builder, Operation *opToClone,
                         IRMapping &mapping, DenseSet<Operation *> &seen,
                         bool assumeContract) {
  if (seen.contains(opToClone))
    return llvm::success();
  seen.insert(opToClone);

  // Ensure all operands have been mapped
  for (auto operand : opToClone->getOperands()) {
    if (mapping.contains(operand))
      continue;

    if (auto *definingOp = operand.getDefiningOp()) {
      // Recurse and clone defining op
      if (failed(
              cloneFanIn(builder, definingOp, mapping, seen, assumeContract)))
        return failure();
    } else {
      // Create symbolic values for arguments
      auto sym = builder.create<verif::SymbolicValueOp>(operand.getLoc(),
                                                        operand.getType());
      mapping.map(operand, sym);
    }
  }

  if (auto contract = dyn_cast<ContractOp>(opToClone)) {
    // Assume it holds
    return inlineContract(contract, builder, mapping, seen);
  }

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
  return success();
}

LogicalResult runOnHWModule(HWModuleOp hwModule, ModuleOp mlirModule) {
  OpBuilder mlirModuleBuilder(mlirModule);
  mlirModuleBuilder.setInsertionPointAfter(hwModule);

  // Collect contract ops
  SmallVector<ContractOp> contracts;
  hwModule.walk([&](ContractOp op) { contracts.push_back(op); });

  for (unsigned i = 0; i < contracts.size(); i++) {
    auto contract = contracts[i];

    // Create verif.formal op
    auto name =
        mlirModuleBuilder.getStringAttr(hwModule.getNameAttr().getValue() +
                                        "_CheckContract_" + std::to_string(i));
    auto formalOp = mlirModuleBuilder.create<verif::FormalOp>(
        contract.getLoc(), name, mlirModuleBuilder.getDictionaryAttr({}));

    // Fill in verif.formal body
    OpBuilder formalBuilder(formalOp);
    formalBuilder.createBlock(&formalOp.getBody());

    IRMapping mapping;
    DenseSet<Operation *> seen;

    // Clone fan in cone for contract operands
    for (auto operand : contract.getOperands()) {
      auto *definingOp = operand.getDefiningOp();
      if (failed(cloneFanIn(formalBuilder, definingOp, mapping, seen, false)))
        return failure();
    }

    // Map results by looking up the input mappings
    for (auto [result, input] :
         llvm::zip(contract.getResults(), contract.getInputs())) {
      mapping.map(result, mapping.lookup(input));
    }

    // Clone body of the contract
    if (failed(cloneContractBody(contract, formalBuilder, mapping, seen, false))) return failure();
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
