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

#define DEBUG_TYPE "drop-const"

using namespace circt;
using namespace firrtl;

namespace {

class OpVisitor : public FIRRTLVisitor<OpVisitor> {
public:
  using FIRRTLVisitor<OpVisitor>::visitExpr;
  using FIRRTLVisitor<OpVisitor>::visitDecl;
  using FIRRTLVisitor<OpVisitor>::visitStmt;

  LogicalResult checkModule(FModuleLike module) {
    auto fmodule = dyn_cast<FModuleOp>(*module);
    // Find 'const' ports
    auto portTypes = SmallVector<Attribute>(module.getPortTypes());
    for (size_t portIndex = 0, numPorts = module.getPortTypes().size();
         portIndex < numPorts; ++portIndex) {
      if (auto convertedType = convertType(module.getPortType(portIndex))) {
        // If this is an FModuleOp, register the block argument to drop 'const'
        if (fmodule)
          constValuesToConvert.push_back(
              {fmodule.getArgument(portIndex), convertedType});
        portTypes[portIndex] = TypeAttr::get(convertedType);
      }
    }

    // Update the module signature with non-'const' ports
    module->setAttr(FModuleLike::getPortTypesAttrName(),
                    ArrayAttr::get(module.getContext(), portTypes));

    if (!fmodule)
      return result;

    // Check the module body
    visitDecl(fmodule);

    // Drop 'const' from all registered values
    for (auto [value, type] : constValuesToConvert)
      value.setType(type);

    return result;
  }

  void visitDecl(FModuleOp module) {
    for (auto &op : llvm::make_early_inc_range(*module.getBodyBlock()))
      dispatchVisitor(&op);
  }

  void visitExpr(ConstCastOp constCast) {
    // Remove any `ConstCastOp`, replacing results with inputs
    constCast.getResult().replaceAllUsesWith(constCast.getInput());
    constCast->erase();
  }

  void visitStmt(WhenOp when) {
    // Check that 'const' destinations are only connected to within
    // 'const'-conditioned when blocks, unless the destination is local to the
    // when block's scope

    auto *previousNonConstConditionedBlock = nonConstConditionedBlock;
    bool isWithinNonconstCondition = !when.getCondition().getType().isConst();

    if (isWithinNonconstCondition)
      nonConstConditionedBlock = &when.getThenBlock();
    for (auto &op : llvm::make_early_inc_range(when.getThenBlock()))
      dispatchVisitor(&op);

    if (when.hasElseRegion()) {
      if (isWithinNonconstCondition)
        nonConstConditionedBlock = &when.getElseBlock();
      for (auto &op : llvm::make_early_inc_range(when.getElseBlock()))
        dispatchVisitor(&op);
    }

    nonConstConditionedBlock = previousNonConstConditionedBlock;
  }

  void visitStmt(ConnectOp connect) { handleConnect(connect); }

  void visitStmt(StrictConnectOp connect) { handleConnect(connect); }

  void visitUnhandledOp(Operation *op) {
    // Register any 'const' results to drop 'const'
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      if (auto convertedType = convertType(result.getType()))
        constValuesToConvert.push_back({result, convertedType});
    }
  }

private:
  /// Returns null type if no conversion is needed
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

  /// Returns null type if no conversion is needed
  FIRRTLBaseType convertType(FIRRTLBaseType type) {
    auto nonConstType = type.getAllConstDroppedType();
    return nonConstType != type ? nonConstType : FIRRTLBaseType{};
  }

  void handleConnect(FConnectLike connect) {
    auto dest = connect.getDest();
    auto destType = dest.getType();
    if (nonConstConditionedBlock && containsConst(destType)) {
      // 'const' connects are allowed if `dest` is local to the non-'const' when
      // block
      auto *destBlock = dest.getParentBlock();
      auto *block = connect->getBlock();
      while (block) {
        // The connect is local to the `dest` declaration, both local to the
        // non-'const' block, so the connect is valid
        if (block == destBlock)
          return;
        // The connect is within the non-'const' condition, non-local to the
        // `dest` declaration, so the connect is invalid
        if (block == nonConstConditionedBlock)
          break;

        if (auto *parentOp = block->getParentOp())
          block = parentOp->getBlock();
        else
          break;
      }

      result = failure();
      if (isConst(destType))
        connect.emitOpError() << "assignment to 'const' type " << destType
                              << " is dependent on a non-'const' condition";
      else
        connect->emitOpError()
            << "assignment to nested 'const' member of type " << destType
            << " is dependent on a non-'const' condition";
    }
  }

  SmallVector<std::pair<Value, Type>> constValuesToConvert;
  Block *nonConstConditionedBlock = nullptr;
  LogicalResult result = success();
};

class DropConstPass : public DropConstBase<DropConstPass> {
  void runOnOperation() override {
    OpVisitor visitor;
    if (failed(visitor.checkModule(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createDropConstPass() {
  return std::make_unique<DropConstPass>();
}
