//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This pass moves inner symbols from ports to wires, then bypasses wire
//  operations with inner symbols by replacing uses with their inputs while
//  keeping the wire to preserve the symbol. This enables optimizations to
//  cross symbol boundaries while maintaining symbol references.
//
//  Note: This transformation assumes that values associated with inner
//  symbols are not mutated through inner symbols (e.g. force). This assumption
//  may not hold in design verification contexts, but is safe in synthesis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWBYPASSINNERSYMBOLS
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace circt::hw;

namespace {
struct HWBypassInnerSymbolsPass
    : public impl::HWBypassInnerSymbolsBase<HWBypassInnerSymbolsPass> {
  void runOnOperation() override;
};

/// Pattern to bypass wire operations with inner symbols.
struct BypassWireWithInnerSym : public OpRewritePattern<WireOp> {
  using OpRewritePattern<WireOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WireOp wire,
                                PatternRewriter &rewriter) const override {
    // Only bypass wires that have an inner symbol or use
    if (!wire.getInnerSymAttr() || wire.use_empty())
      return failure();

    // Replace all uses of the wire with its input
    rewriter.modifyOpInPlace(wire, [&] {
      wire->replaceAllUsesWith(ArrayRef<Value>{wire.getInput()});
    });
    return success();
  }
};
} // namespace

void HWBypassInnerSymbolsPass::runOnOperation() {
  auto module = getOperation();

  // First move the inner symbol from the port to allow constprop and other
  // optimizations to cross the boundary.
  hw::ModulePortInfo portList(module.getPortList());
  size_t outputIndex = 0;
  auto *outputOp = module.getBodyBlock()->getTerminator();
  OpBuilder builder(&getContext());
  for (auto [index, port] : llvm::enumerate(portList)) {
    if (auto sym = port.getSym()) {
      // Move the symbol on port to a wire.
      module.setPortSymbolAttr(index, {});
      if (port.isOutput()) {
        auto value = outputOp->getOperand(outputIndex);
        builder.setInsertionPointAfterValue(value);
        auto wire = WireOp::create(builder, value.getLoc(), value);
        wire.setInnerSymAttr(sym);
      } else {
        auto arg = module.getBodyBlock()->getArgument(index - outputIndex);
        builder.setInsertionPointToStart(module.getBodyBlock());
        auto wire = WireOp::create(builder, arg.getLoc(), arg);
        wire.setInnerSymAttr(sym);
      }
    }

    if (port.isOutput())
      outputIndex++;
  }

  RewritePatternSet patterns(&getContext());
  patterns.add<BypassWireWithInnerSym>(&getContext());
  mlir::walkAndApplyPatterns(module, std::move(patterns));
}
