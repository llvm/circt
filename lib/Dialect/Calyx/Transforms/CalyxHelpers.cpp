//===- CalyxHelpers.cpp - Calyx helper methods -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various helper methods for building Calyx programs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"

namespace circt {
namespace calyx {

calyx::RegisterOp createRegister(Location loc, OpBuilder &builder,
                                 ComponentOp component, size_t width,
                                 Twine prefix) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(component.getBodyBlock());
  return RegisterOp::create(builder, loc, (prefix + "_reg").str(), width);
}

hw::ConstantOp createConstant(Location loc, OpBuilder &builder,
                              ComponentOp component, size_t width,
                              size_t value) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(component.getBodyBlock());
  return hw::ConstantOp::create(builder, loc,
                                APInt(width, value, /*isSigned=*/false));
}

calyx::InstanceOp createInstance(Location loc, OpBuilder &builder,
                                 ComponentOp component,
                                 SmallVectorImpl<Type> &resultTypes,
                                 StringRef instanceName,
                                 StringRef componentName) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(component.getBodyBlock());
  return InstanceOp::create(builder, loc, resultTypes, instanceName,
                            componentName);
}

std::string getInstanceName(mlir::func::CallOp callOp) {
  SmallVector<StringRef, 2> strVet = {callOp.getCallee(), "instance"};
  return llvm::join(strVet, /*separator=*/"_");
}

bool isControlLeafNode(Operation *op) { return isa<calyx::EnableOp>(op); }

DictionaryAttr getMandatoryPortAttr(MLIRContext *ctx, StringRef name) {
  return DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, name), UnitAttr::get(ctx))});
}

void addMandatoryComponentPorts(PatternRewriter &rewriter,
                                SmallVectorImpl<calyx::PortInfo> &ports) {
  MLIRContext *ctx = rewriter.getContext();
  ports.push_back({
      rewriter.getStringAttr(clkPort),
      rewriter.getI1Type(),
      calyx::Direction::Input,
      getMandatoryPortAttr(ctx, clkPort),
  });
  ports.push_back({
      rewriter.getStringAttr(resetPort),
      rewriter.getI1Type(),
      calyx::Direction::Input,
      getMandatoryPortAttr(ctx, resetPort),
  });
  ports.push_back({
      rewriter.getStringAttr(goPort),
      rewriter.getI1Type(),
      calyx::Direction::Input,
      getMandatoryPortAttr(ctx, goPort),
  });
  ports.push_back({
      rewriter.getStringAttr(donePort),
      rewriter.getI1Type(),
      calyx::Direction::Output,
      getMandatoryPortAttr(ctx, donePort),
  });
}

unsigned handleZeroWidth(int64_t dim) {
  return std::max(llvm::Log2_64_Ceil(dim), 1U);
}

} // namespace calyx
} // namespace circt
