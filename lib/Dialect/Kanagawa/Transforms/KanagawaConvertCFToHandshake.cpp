//===- KanagawaConvertCFToHandshakePass.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"

#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Conversion/CFToHandshake.h"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWACONVERTCFTOHANDSHAKE
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace kanagawa;

namespace {

struct ConvertCFToHandshakePass
    : public circt::kanagawa::impl::KanagawaConvertCFToHandshakeBase<
          ConvertCFToHandshakePass> {
  void runOnOperation() override;

  LogicalResult convertMethod(MethodOp method);
};
} // anonymous namespace

LogicalResult ConvertCFToHandshakePass::convertMethod(MethodOp method) {
  // Add a control input/output to the method.
  OpBuilder b(method);
  llvm::SmallVector<Type> newArgTypes, newResTypes;
  auto methodLikeOp = cast<MethodLikeOpInterface>(method.getOperation());
  llvm::copy(methodLikeOp.getArgumentTypes(), std::back_inserter(newArgTypes));
  llvm::copy(methodLikeOp.getResultTypes(), std::back_inserter(newResTypes));
  newArgTypes.push_back(b.getNoneType());
  newResTypes.push_back(b.getNoneType());
  auto newFuncType = b.getFunctionType(newArgTypes, newResTypes);
  auto dataflowMethodOp = DataflowMethodOp::create(
      b, method.getLoc(), method.getInnerSymAttr(), TypeAttr::get(newFuncType),
      method.getArgNamesAttr(), method.getArgAttrsAttr(),
      method.getResAttrsAttr());
  dataflowMethodOp.getFunctionBody().takeBody(method.getBody());
  dataflowMethodOp.getBodyBlock()->addArgument(b.getNoneType(),
                                               method.getLoc());
  Value entryCtrl = dataflowMethodOp.getBodyBlock()->getArguments().back();
  method.erase();

  handshake::HandshakeLowering fol(dataflowMethodOp.getBody());
  if (failed(handshake::lowerRegion<kanagawa::ReturnOp, kanagawa::ReturnOp>(
          fol,
          /*sourceConstants*/ false, /*disableTaskPipelining*/ false,
          entryCtrl)))
    return failure();

  return success();
}

void ConvertCFToHandshakePass::runOnOperation() {
  ClassOp classOp = getOperation();
  for (auto method : llvm::make_early_inc_range(classOp.getOps<MethodOp>())) {
    if (failed(convertMethod(method)))
      return signalPassFailure();
  }
}

std::unique_ptr<Pass> circt::kanagawa::createConvertCFToHandshakePass() {
  return std::make_unique<ConvertCFToHandshakePass>();
}
