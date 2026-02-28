//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_LOWERTESTSPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace verif;

namespace {
struct LowerTestsPass : verif::impl::LowerTestsPassBase<LowerTestsPass> {
  void runOnOperation() override;
  LogicalResult lowerTest(FormalOp op);
  LogicalResult lowerTest(SimulationOp op);
  void addComment(hw::HWModuleOp op, StringRef kind, StringAttr name,
                  DictionaryAttr params);
};
} // namespace

/// Convert a `verif.formal` to an `hw.module`.
LogicalResult LowerTestsPass::lowerTest(FormalOp op) {
  IRRewriter rewriter(op);

  // Collect all symbolic values in the test and create ports for each.
  SmallVector<SymbolicValueOp> symOps;
  SmallVector<hw::PortInfo> ports;
  for (auto symOp : op.getOps<SymbolicValueOp>()) {
    symOps.push_back(symOp);
    hw::PortInfo port;
    port.name = rewriter.getStringAttr("symbolic_value_" + Twine(ports.size()));
    port.type = symOp.getType();
    port.dir = hw::ModulePort::Input;
    ports.push_back(port);
  }

  // Create a module with these ports and inline the body of the formal op.
  auto moduleOp =
      hw::HWModuleOp::create(rewriter, op.getLoc(), op.getNameAttr(), ports);
  moduleOp.getBodyBlock()->getOperations().splice(
      moduleOp.getBodyBlock()->begin(), op.getBody().front().getOperations());
  addComment(moduleOp, "FORMAL", op.getNameAttr(), op.getParametersAttr());

  // Replace symbolic values with module arguments.
  for (auto [symOp, arg] :
       llvm::zip(symOps, moduleOp.getBody().getArguments())) {
    rewriter.replaceAllUsesWith(symOp.getResult(), arg);
    rewriter.eraseOp(symOp);
  }

  // Remove the original test op.
  rewriter.eraseOp(op);
  return success();
}

/// Convert a `verif.simulation` to an `hw.module`.
LogicalResult LowerTestsPass::lowerTest(SimulationOp op) {
  IRRewriter rewriter(op);

  // Gather the list of ports for the module.
  std::array<hw::PortInfo, 4> ports;

  ports[0].name = rewriter.getStringAttr("clock");
  ports[0].type = seq::ClockType::get(op.getContext());
  ports[0].dir = hw::ModulePort::Input;

  ports[1].name = rewriter.getStringAttr("init");
  ports[1].type = rewriter.getI1Type();
  ports[1].dir = hw::ModulePort::Input;

  ports[2].name = rewriter.getStringAttr("done");
  ports[2].type = rewriter.getI1Type();
  ports[2].dir = hw::ModulePort::Output;

  ports[3].name = rewriter.getStringAttr("success");
  ports[3].type = rewriter.getI1Type();
  ports[3].dir = hw::ModulePort::Output;

  // Create a module with these ports and inline the body of the simulation op.
  auto moduleOp =
      hw::HWModuleOp::create(rewriter, op.getLoc(), op.getNameAttr(), ports);
  moduleOp.getBody().takeBody(op.getBodyRegion());
  addComment(moduleOp, "SIMULATION", op.getNameAttr(), op.getParametersAttr());

  // Replace the `verif.yield` with an `hw.output` op.
  auto *yieldOp = moduleOp.getBodyBlock()->getTerminator();
  rewriter.setInsertionPoint(yieldOp);
  hw::OutputOp::create(rewriter, yieldOp->getLoc(), yieldOp->getOperands());
  rewriter.eraseOp(yieldOp);

  // Remove the original test op.
  rewriter.eraseOp(op);
  return success();
}

/// Add a comment to an `hw.module` describing test kind, name, and parameters.
void LowerTestsPass::addComment(hw::HWModuleOp op, StringRef kind,
                                StringAttr name, DictionaryAttr params) {
  SmallString<128> comment;
  llvm::raw_svector_ostream(comment)
      << kind << " TEST: " << name.getValue() << " " << params;
  op.setComment(comment);
}

void LowerTestsPass::runOnOperation() {
  bool changed = false;
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    auto result = TypeSwitch<Operation *, LogicalResult>(&op)
                      .Case<FormalOp, SimulationOp>([&](auto op) {
                        changed = true;
                        return lowerTest(op);
                      })
                      .Default([](auto) { return success(); });
    if (failed(result))
      return signalPassFailure();
  }
  if (!changed)
    return markAllAnalysesPreserved();
}
