//===- DropConst.cpp - Check and remove const types -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DropConst pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/Support/SaveAndRestore.h"

#define DEBUG_TYPE "drop-const"

using namespace circt;
using namespace firrtl;

// NOLINTBEGIN(misc-no-recursion)
static bool typeHasConstLeaf(FIRRTLBaseType type, bool outerTypeIsConst = false,
                             bool isFlip = false) {
  auto typeIsConst = outerTypeIsConst || type.isConst();

  if (typeIsConst && type.isPassive())
    return !isFlip;

  if (auto bundleType = type.dyn_cast<BundleType>())
    return llvm::any_of(bundleType.getElements(), [&](auto &element) {
      return typeHasConstLeaf(element.type, typeIsConst,
                              isFlip ^ element.isFlip);
    });

  if (auto vectorType = type.dyn_cast<FVectorType>())
    return typeHasConstLeaf(vectorType.getElementType(), typeIsConst, isFlip);

  return typeIsConst && !isFlip;
}
// NOLINTEND(misc-no-recursion)

namespace {
class OpVisitor : public FIRRTLVisitor<OpVisitor, LogicalResult> {
public:
  using FIRRTLVisitor<OpVisitor, LogicalResult>::visitExpr;
  using FIRRTLVisitor<OpVisitor, LogicalResult>::visitDecl;
  using FIRRTLVisitor<OpVisitor, LogicalResult>::visitStmt;

  LogicalResult handleModule(FModuleLike module) {
    auto fmodule = dyn_cast<FModuleOp>(*module);

    // Check the module body
    if (fmodule) {
      if (failed(visitDecl(fmodule)))
        return failure();
    }

    // Find 'const' ports
    auto portTypes = SmallVector<Attribute>(module.getPortTypes());
    for (size_t portIndex = 0, numPorts = module.getPortTypes().size();
         portIndex < numPorts; ++portIndex) {
      if (auto convertedType = convertType(module.getPortType(portIndex))) {
        // If this is an FModuleOp, register the block argument to drop 'const'
        if (fmodule)
          fmodule.getArgument(portIndex).setType(convertedType);
        portTypes[portIndex] = TypeAttr::get(convertedType);
      }
    }

    // Update the module signature with non-'const' ports
    module->setAttr(FModuleLike::getPortTypesAttrName(),
                    ArrayAttr::get(module.getContext(), portTypes));

    return success();
  }

  LogicalResult visitDecl(FModuleOp module) {
    for (auto &op :
         llvm::make_early_inc_range(llvm::reverse(*module.getBodyBlock()))) {
      if (failed(dispatchVisitor(&op)))
        return failure();
    }
    return success();
  }

  LogicalResult visitExpr(ConstCastOp constCast) {
    // Remove any `ConstCastOp`, replacing results with inputs
    constCast.getResult().replaceAllUsesWith(constCast.getInput());
    constCast->erase();
    return success();
  }

  LogicalResult visitStmt(WhenOp when) {
    // Check that 'const' destinations are only connected to within
    // 'const'-conditioned when blocks, unless the destination is local to the
    // when block's scope

    llvm::SaveAndRestore<Block *> blockSave(nonConstConditionedBlock);
    bool isWithinNonconstCondition = !when.getCondition().getType().isConst();

    if (isWithinNonconstCondition)
      nonConstConditionedBlock = &when.getThenBlock();
    for (auto &op :
         llvm::make_early_inc_range(llvm::reverse(when.getThenBlock())))
      if (failed(dispatchVisitor(&op)))
        return failure();

    if (when.hasElseRegion()) {
      if (isWithinNonconstCondition)
        nonConstConditionedBlock = &when.getElseBlock();
      for (auto &op :
           llvm::make_early_inc_range(llvm::reverse(when.getElseBlock())))
        if (failed(dispatchVisitor(&op)))
          return failure();
    }

    return success();
  }

  LogicalResult visitStmt(ConnectOp connect) { return handleConnect(connect); }

  LogicalResult visitStmt(StrictConnectOp connect) {
    return handleConnect(connect);
  }

  LogicalResult visitUnhandledOp(Operation *op) {
    // Register any 'const' results to drop 'const'
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      if (auto convertedType = convertType(result.getType()))
        result.setType(convertedType);
    }

    return success();
  }

  LogicalResult visitInvalidOp(Operation *op) { return success(); }

private:
  /// Returns null type if no conversion is needed.
  Type convertType(Type type) {
    if (auto base = type.dyn_cast<FIRRTLBaseType>()) {
      return convertType(base);
    }

    if (auto refType = type.dyn_cast<RefType>()) {
      if (auto converted = convertType(refType.getType()))
        return RefType::get(converted.cast<FIRRTLBaseType>(),
                            refType.getForceable());
    }

    return {};
  }

  /// Returns null type if no conversion is needed.
  FIRRTLBaseType convertType(FIRRTLBaseType type) {
    auto nonConstType = type.getAllConstDroppedType();
    return nonConstType != type ? nonConstType : FIRRTLBaseType{};
  }

  LogicalResult handleConnect(FConnectLike connect) {
    auto dest = connect.getDest();
    auto destType = dest.getType().cast<FIRRTLBaseType>();
    if (nonConstConditionedBlock && destType && destType.containsConst()) {
      // 'const' connects are allowed if `dest` is local to the non-'const' when
      // block.
      auto *destBlock = dest.getParentBlock();
      auto *block = connect->getBlock();
      while (block) {
        // The connect is local to the `dest` declaration, both local to the
        // non-'const' block, so the connect is valid.
        if (block == destBlock)
          return success();
        // The connect is within the non-'const' condition, non-local to the
        // `dest` declaration, so the connect is invalid
        if (block == nonConstConditionedBlock)
          break;

        if (auto *parentOp = block->getParentOp())
          block = parentOp->getBlock();
        else
          break;
      }

      // A const dest is allowed if leaf elements are effectively non-const due
      // to flips.
      if (typeHasConstLeaf(destType)) {
        if (destType.isConst())
          return connect.emitOpError()
                 << "assignment to 'const' type " << destType
                 << " is dependent on a non-'const' condition";
        return connect->emitOpError()
               << "assignment to nested 'const' member of type " << destType
               << " is dependent on a non-'const' condition";
      }
    }

    return success();
  }

  Block *nonConstConditionedBlock = nullptr;
};

class DropConstPass : public DropConstBase<DropConstPass> {
  void runOnOperation() override {
    if (failed(OpVisitor().handleModule(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createDropConstPass() {
  return std::make_unique<DropConstPass>();
}
