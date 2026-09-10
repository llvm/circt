//===- KanagawaAddOperatorLibraryPass.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "circt/Transforms/Passes.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <iterator>

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWAADDOPERATORLIBRARY
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace kanagawa;

namespace {

struct AddOperatorLibraryPass
    : public circt::kanagawa::impl::KanagawaAddOperatorLibraryBase<
          AddOperatorLibraryPass> {
  void runOnOperation() override;
};

// TODO @mortbopet: once we have c++20, change  this to a templated lambda to
// avoid passing the builder around.
template <typename TOp>
void addOperator(ImplicitLocOpBuilder &b, int latency) {
  ssp::OperatorTypeOp::create(
      b, b.getStringAttr(TOp::getOperationName()),
      b.getArrayAttr({b.getAttr<ssp::LatencyAttr>(latency)}));
}

} // anonymous namespace

void AddOperatorLibraryPass::runOnOperation() {
  auto b = ImplicitLocOpBuilder::atBlockBegin(getOperation().getLoc(),
                                              getOperation().getBody());
  auto opLib = ssp::OperatorLibraryOp::create(b);
  opLib.setSymNameAttr(b.getStringAttr(kKanagawaOperatorLibName));
  b.setInsertionPointToStart(opLib.getBodyBlock());

  // Provide definitions for some comb ops - just latency properties for now.

  // Arithmetic operators
  addOperator<comb::AddOp>(b, 1);
  addOperator<comb::SubOp>(b, 1);
  addOperator<comb::MulOp>(b, 2);
  addOperator<comb::ModSOp>(b, 2);
  addOperator<comb::ModUOp>(b, 2);

  // Boolean operators
  addOperator<comb::AndOp>(b, 0);
  addOperator<comb::OrOp>(b, 0);
  addOperator<comb::XorOp>(b, 0);

  // Comparison
  addOperator<comb::ICmpOp>(b, 1);

  // Shift
  addOperator<comb::ShlOp>(b, 0);
  addOperator<comb::ShrUOp>(b, 0);
  addOperator<comb::ShrSOp>(b, 1);
}

std::unique_ptr<Pass> circt::kanagawa::createAddOperatorLibraryPass() {
  return std::make_unique<AddOperatorLibraryPass>();
}
