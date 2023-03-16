//===- SinkInputs.cpp------ -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-sink-inputs"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct SinkInputsPass : public SinkInputsBase<SinkInputsPass> {
  void runOnOperation() override;
  void runOnModule();
};
} // namespace

void SinkInputsPass::runOnOperation() {
  DenseMap<StringAttr, SmallVector<Operation *>> arcConstArgs;
  DenseMap<StringAttr, SmallVector<StateOp>> arcUses;

  // Find all arc uses that use constant operands.
  auto module = getOperation();
  module.walk([&](StateOp stateOp) {
    auto arcName = stateOp.getArcAttr().getAttr();
    arcUses[arcName].push_back(stateOp);

    SmallVector<Operation *> stateConsts(stateOp.getInputs().size());
    for (auto [constArg, input] : llvm::zip(stateConsts, stateOp.getInputs()))
      if (auto *op = input.getDefiningOp())
        if (op->hasTrait<OpTrait::ConstantLike>())
          constArg = op;

    auto &arcConsts = arcConstArgs[arcName];
    bool isFirst = arcConsts.empty();
    if (isFirst)
      arcConsts.resize(stateOp.getInputs().size());

    for (auto [arcConstArg, stateConstArg] :
         llvm::zip(arcConsts, stateConsts)) {
      if (isFirst) {
        arcConstArg = stateConstArg;
        continue;
      }
      if (arcConstArg && stateConstArg &&
          arcConstArg->getName() == stateConstArg->getName() &&
          arcConstArg->getAttrDictionary() ==
              stateConstArg->getAttrDictionary())
        continue;
      arcConstArg = nullptr;
    }
  });

  // Now we go through all the defines and move the constant ops into the
  // bodies and rewrite the function types.
  for (auto defOp : module.getOps<DefineOp>()) {
    // Move the constants into the arc and erase the block arguments.
    auto builder = OpBuilder::atBlockBegin(&defOp.getBodyBlock());
    llvm::BitVector toDelete(defOp.getBodyBlock().getNumArguments());
    for (auto [constArg, arg] :
         llvm::zip(arcConstArgs[defOp.getNameAttr()], defOp.getArguments())) {
      if (!constArg)
        continue;
      auto *inlinedConst = builder.clone(*constArg);
      arg.replaceAllUsesWith(inlinedConst->getResult(0));
      toDelete.set(arg.getArgNumber());
    }
    defOp.getBodyBlock().eraseArguments(toDelete);
    defOp.setType(builder.getFunctionType(
        defOp.getBodyBlock().getArgumentTypes(), defOp.getResultTypes()));

    // Rewrite all arc uses to not pass in the constant anymore.
    for (auto stateOp : arcUses[defOp.getNameAttr()]) {
      SmallPtrSet<Value, 4> maybeUnusedValues;
      SmallVector<Value> newInputs;
      for (auto [index, value] : llvm::enumerate(stateOp.getInputs())) {
        if (toDelete[index])
          maybeUnusedValues.insert(value);
        else
          newInputs.push_back(value);
      }
      stateOp.getInputsMutable().assign(newInputs);
      for (auto value : maybeUnusedValues)
        if (value.use_empty())
          value.getDefiningOp()->erase();
    }
  }
}

std::unique_ptr<Pass> arc::createSinkInputsPass() {
  return std::make_unique<SinkInputsPass>();
}
