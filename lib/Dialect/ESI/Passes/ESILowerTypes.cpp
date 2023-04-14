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
};
} // anonymous namespace

namespace {
struct ModuleConversionPattern
    : public mlir::OpInterfaceConversionPattern<hw::HWMutableModuleLike> {
public:
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;
  LogicalResult
  matchAndRewrite(hw::HWMutableModuleLike mod, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;

  /// Lower an individual port, modifing 'port'. Returns 'true' if the port was
  /// changed.
  bool lowerPort(hw::HWMutableModuleLike mod, hw::PortInfo &port) const {
    Type newType = getTypeConverter()->convertType(port.type);
    if (newType == port.type)
      return false;
    port.type = newType;
    return true;
  }
};
} // namespace

LogicalResult ModuleConversionPattern::matchAndRewrite(
    hw::HWMutableModuleLike mod, ArrayRef<Value>,
    ConversionPatternRewriter &rewriter) const {
  hw::ModulePortInfo ports = mod.getPorts();
  Block *body = nullptr;
  Operation *terminator;

  // If 'mod' has a body, set up the necessary variables to modify it.
  if (!mod->getRegions().empty() && !mod->getRegion(0).empty()) {
    body = &mod->getRegion(0).getBlocks().front();
    rewriter.setInsertionPointToStart(body);
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
    auto wrap =
        rewriter.create<WrapWindow>(arg.getLoc(), oldArg.getType(), arg);
    oldArg.replaceAllUsesWith(wrap);
    body->eraseArgument(port.argNum + 1);
  }

  // Lower the output ports.
  SmallVector<std::pair<unsigned, hw::PortInfo>> loweredOutputs;
  SmallVector<unsigned> loweredOutputIdxs;
  if (body)
    rewriter.setInsertionPoint(terminator);
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
    auto unwrap = rewriter.create<UnwrapWindow>(
        port.loc, port.type, terminator->getOperand(port.argNum));
    terminator->setOperand(port.argNum, unwrap.getFrame());
  }

  rewriter.updateRootInPlace(mod, [&]() {
    mod.modifyPorts(loweredInputs, loweredOutputs, loweredInputIdxs,
                    loweredOutputIdxs);
  });
  return success();
}

namespace {
struct InstanceConversionPattern
    : public mlir::OpInterfaceConversionPattern<hw::HWInstanceLike> {
public:
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;
  LogicalResult
  matchAndRewrite(hw::HWInstanceLike inst, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult InstanceConversionPattern::matchAndRewrite(
    hw::HWInstanceLike inst, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  // Build a list of new result types.
  SmallVector<Type> newResultTypes;
  if (failed(getTypeConverter()->convertTypes(inst->getResultTypes(),
                                              newResultTypes)))
    return rewriter.notifyMatchFailure(inst.getLoc(),
                                       "could not convert all types");

  // Since we cannot change the return types, we have to create a new operation
  // with the new operands and result types.
  Operation *newInst =
      rewriter.create(OperationState(inst.getLoc(), inst->getName(), operands,
                                     newResultTypes, inst->getAttrs()));
  // Delete the old inst.
  rewriter.replaceOp(inst, newInst->getResults());
  return success();
}

void ESILowerTypesPass::runOnOperation() {
  TypeConverter types;
  types.addConversion([](Type t) { return t; });
  types.addConversion(
      [](WindowType window) { return window.getLoweredType(); });
  types.addSourceMaterialization(
      [&](OpBuilder &b, WindowType resultType, ValueRange inputs,
          Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        auto wrap = b.create<WrapWindow>(loc, resultType, inputs[0]);
        return wrap.getWindow();
      });

  types.addTargetMaterialization(
      [&](OpBuilder &b, hw::UnionType resultType, ValueRange inputs,
          Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1 || !isa<WindowType>(inputs[0].getType()))
          return std::nullopt;
        auto unwrap = b.create<UnwrapWindow>(loc, resultType, inputs[0]);
        return unwrap.getFrame();
      });

  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    return TypeSwitch<Operation *, bool>(op)
        .Case([](hw::HWInstanceLike inst) {
          return !(
              llvm::any_of(inst->getOperandTypes(), hw::type_isa<WindowType>) ||
              llvm::any_of(inst->getResultTypes(), hw::type_isa<WindowType>));
        })
        .Case([](hw::HWMutableModuleLike mod) {
          auto isWindowPort = [](hw::PortInfo p) {
            return hw::type_isa<WindowType>(p.type);
          };
          return !(llvm::any_of(mod.getPorts().inputs, isWindowPort) ||
                   llvm::any_of(mod.getPorts().outputs, isWindowPort));
        })
        .Default([](Operation *) { return true; });
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<ModuleConversionPattern>(types, &getContext());
  patterns.add<InstanceConversionPattern>(types, &getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESITypeLoweringPass() {
  return std::make_unique<ESILowerTypesPass>();
}
