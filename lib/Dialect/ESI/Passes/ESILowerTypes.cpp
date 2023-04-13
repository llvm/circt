//===- ESIPasses.cpp - ESI to HW/SV conversion passes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower high-level ESI types to HW conversions and pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;

namespace {
/// Lower all "high-level" ESI types on modules to some lower construct.
struct ESILowerTypesPass : public LowerESITypesBase<ESILowerTypesPass> {
  void runOnOperation() override;

private:
  void lowerMod(hw::HWMutableModuleLike mod);
  void lowerInstance(hw::HWInstanceLike inst);

  /// Lower an individual port, modifing 'port'. Returns 'true' if the port was
  /// changed.
  bool lowerPort(hw::HWMutableModuleLike mod, hw::PortInfo &port);

  /// Cache the lowered types to avoid running the same type lowering again and
  /// again.
  DenseMap<WindowType, hw::UnionType> typeCache;
  /// Track the modules we've modified both to make lookup easier and to avoid
  /// attempting to lower instances which do not have to be lowered.
  DenseMap<StringAttr, hw::HWMutableModuleLike> modsMutated;
};
} // anonymous namespace

bool ESILowerTypesPass::lowerPort(hw::HWMutableModuleLike mod,
                                  hw::PortInfo &port) {
  auto window = hw::type_dyn_cast<WindowType>(port.type);
  if (!window)
    return false;
  hw::UnionType &lowered = typeCache[window];
  if (!lowered)
    lowered = window.getLoweredType();
  port.type = lowered;
  return true;
}

void ESILowerTypesPass::lowerMod(hw::HWMutableModuleLike mod) {
  hw::ModulePortInfo ports = mod.getPorts();
  Block *body = nullptr;
  Operation *terminator;
  OpBuilder b(&getContext());

  // If 'mod' has a body, set up the necessary variables to modify it.
  if (!mod->getRegions().empty() && !mod->getRegion(0).empty()) {
    body = &mod->getRegion(0).getBlocks().front();
    b.setInsertionPointToStart(body);
    terminator = body->getTerminator();
  }

  // Lower the input ports.
  SmallVector<std::pair<unsigned, hw::PortInfo>> loweredInputs;
  SmallVector<unsigned> loweredInputIdxs;
  for (auto port : ports.inputs) {
    if (!lowerPort(mod, port))
      continue;
    loweredInputs.emplace_back(port.argNum, port);
    loweredInputIdxs.push_back(port.argNum);
    if (!body)
      continue;

    // Build a wrapper op to wrap the lowered value. This will hopefully be
    // canonicalized away.
    BlockArgument oldArg = body->getArgument(port.argNum);
    auto arg = body->insertArgument(port.argNum, port.type, oldArg.getLoc());
    auto wrap = b.create<WrapWindow>(arg.getLoc(), oldArg.getType(), arg);
    oldArg.replaceAllUsesWith(wrap);
    body->eraseArgument(port.argNum + 1);
  }

  // Lower the output ports.
  SmallVector<std::pair<unsigned, hw::PortInfo>> loweredOutputs;
  SmallVector<unsigned> loweredOutputIdxs;
  if (body)
    b.setInsertionPoint(terminator);
  for (auto port : ports.outputs) {
    if (!lowerPort(mod, port))
      continue;
    loweredOutputs.emplace_back(port.argNum, port);
    loweredOutputIdxs.push_back(port.argNum);
    if (!body)
      continue;

    // Build an unwrap to unwrap the window into the lowered output. This will
    // hopefully be canonicalized away.
    assert(terminator->getNumOperands() > port.argNum);
    auto unwrap = b.create<UnwrapWindow>(port.loc, port.type,
                                         terminator->getOperand(port.argNum));
    terminator->setOperand(port.argNum, unwrap.getFrame());
  }

  // Run the modifications if need to.
  if (loweredInputs.empty() && loweredOutputs.empty())
    return;
  modsMutated[SymbolTable::getSymbolName(mod)] = mod;
  mod.modifyPorts(loweredInputs, loweredOutputs, loweredInputIdxs,
                  loweredOutputIdxs);
}

void ESILowerTypesPass::lowerInstance(hw::HWInstanceLike inst) {
  if (!modsMutated.contains(inst.getReferencedModuleNameAttr()))
    return;

  // Build a list of new operands and drive the windows with new unwrap ops.
  SmallVector<Value> newOperands;
  OpBuilder b(inst);
  for (Value operand : inst->getOperands()) {
    WindowType window = hw::type_dyn_cast<WindowType>(operand.getType());
    if (!window) {
      newOperands.push_back(operand);
      continue;
    }
    assert(typeCache.contains(window));
    auto unwrapped =
        b.create<UnwrapWindow>(inst.getLoc(), typeCache[window], operand);
    newOperands.push_back(unwrapped.getFrame());
  }

  // Build a list of new result types.
  SmallVector<Type> newResultTypes;
  b.setInsertionPointAfter(inst);
  for (OpResult result : inst->getResults()) {
    WindowType window = hw::type_dyn_cast<WindowType>(result.getType());
    if (!window) {
      newResultTypes.push_back(result.getType());
      continue;
    }
    assert(typeCache.contains(window));
    newResultTypes.push_back(typeCache[window]);
  }

  // Since we cannot change the return type, we have to create a new operation
  // with the new operands and result types.
  Operation *newInst =
      Operation::create(inst.getLoc(), inst->getName(), newResultTypes,
                        newOperands, inst->getAttrs());
  b.insert(newInst);

  // Replace the old result users with the new results. For lowered outputs,
  // insert a wrap to drive the original.
  for (auto [oldResult, newResult] :
       llvm::zip(inst->getResults(), newInst->getResults())) {
    WindowType window = hw::type_dyn_cast<WindowType>(oldResult.getType());
    if (!window) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    auto wrap =
        b.create<WrapWindow>(inst.getLoc(), oldResult.getType(), newResult);
    oldResult.replaceAllUsesWith(wrap.getWindow());
  }

  // Delete the old inst.
  inst->erase();
}

void ESILowerTypesPass::runOnOperation() {
  auto design = cast<ModuleOp>(getOperation());
  for (auto mod : design.getOps<hw::HWMutableModuleLike>())
    lowerMod(mod);
  design.walk([&](hw::HWInstanceLike inst) { lowerInstance(inst); });
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESITypeLoweringPass() {
  return std::make_unique<ESILowerTypesPass>();
}
