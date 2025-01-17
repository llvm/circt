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

template <typename TO>
Operation *replaceContractOp(OpBuilder &builder, RequireLike op,
                             IRMapping &mapping) {
  StringAttr labelAttr;
  if (auto label = op.getLabel())
    labelAttr = builder.getStringAttr(label.value());

  Value enableValue;
  if (auto enable = op.getEnable())
    enable = mapping.lookup(enable);

  return builder.create<TO>(op.getLoc(), mapping.lookup(op.getProperty()),
                            enableValue, labelAttr);
}

void cloneFanIn(OpBuilder &builder, Operation *opToClone, IRMapping &mapping,
                DenseSet<Operation *> &seen) {
  if (seen.contains(opToClone))
    return;
  seen.insert(opToClone);

  // Ensure all operands have been mapped
  for (auto operand : opToClone->getOperands()) {
    if (mapping.contains(operand))
      continue;

    auto *definingOp = operand.getDefiningOp();
    if (definingOp) {
      // Recurse and clone defining op
      cloneFanIn(builder, definingOp, mapping, seen);
    } else {
      // Create symbolic values for arguments
      auto sym = builder.create<verif::SymbolicValueOp>(operand.getLoc(),
                                                        operand.getType());
      mapping.map(operand, sym);
    }
  }

  Operation *clonedOp;
  // Replace ensure/require ops, otherwise clone
  if (isa<EnsureOp>(opToClone)) {
    clonedOp = replaceContractOp<AssertOp>(
        builder, dyn_cast<RequireLike>(*opToClone), mapping);
  } else if (isa<RequireOp>(opToClone)) {
    clonedOp = replaceContractOp<AssumeOp>(
        builder, dyn_cast<RequireLike>(*opToClone), mapping);
  } else {
    clonedOp = builder.clone(*opToClone, mapping);
  }

  // Add mappings for results
  for (auto [x, y] :
       llvm::zip(opToClone->getResults(), clonedOp->getResults())) {
    mapping.map(x, y);
  }
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
      cloneFanIn(formalBuilder, definingOp, mapping, seen);
    }

    // Map results by looking up the input mappings
    for (auto [result, input] :
         llvm::zip(contract.getResults(), contract.getInputs())) {
      mapping.map(result, mapping.lookup(input));
    }

    // Clone body of the contract
    for (auto &op : contract.getBody().front().getOperations()) {
      cloneFanIn(formalBuilder, &op, mapping, seen);
    }
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
