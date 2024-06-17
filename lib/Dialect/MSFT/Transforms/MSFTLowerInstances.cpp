//===- MSFTLowerInstances.cpp - Instace lowering pass -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace msft {
#define GEN_PASS_DEF_LOWERINSTANCES
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace msft
} // namespace circt

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Lower dynamic instances to global refs.
//===----------------------------------------------------------------------===//

namespace {
struct LowerInstancesPass
    : public circt::msft::impl::LowerInstancesBase<LowerInstancesPass> {
  void runOnOperation() override;

  LogicalResult lower(DynamicInstanceOp inst, InstanceHierarchyOp hier,
                      OpBuilder &b);

  // Cache the top-level symbols. Insert the new ones we're creating for new
  // HierPathOps.
  SymbolCache topSyms;
};
} // anonymous namespace

LogicalResult LowerInstancesPass::lower(DynamicInstanceOp inst,
                                        InstanceHierarchyOp hier,
                                        OpBuilder &b) {

  hw::HierPathOp ref = nullptr;

  // If 'inst' doesn't contain any ops which use a hierpath op, don't create
  // one.
  if (llvm::any_of(inst.getOps(), [](Operation &op) {
        return isa<DynInstDataOpInterface>(op);
      })) {

    // Come up with a unique symbol name.
    auto refSym = StringAttr::get(&getContext(), "instref");
    auto origRefSym = refSym;
    unsigned ctr = 0;
    while (topSyms.getDefinition(refSym))
      refSym = StringAttr::get(&getContext(),
                               origRefSym.getValue() + "_" + Twine(++ctr));

    // Create a hierpath to replace us.
    ArrayAttr hierPath = inst.getPath();
    ref = b.create<hw::HierPathOp>(inst.getLoc(), refSym, hierPath);

    // Add the new symbol to the symbol cache.
    topSyms.addDefinition(refSym, ref);
  }

  // Relocate all my children.
  OpBuilder hierBlock(&hier.getBody().front().front());
  for (Operation &op : llvm::make_early_inc_range(inst.getOps())) {
    // Child instances should have been lowered already.
    assert(!isa<DynamicInstanceOp>(op));
    op.remove();
    hierBlock.insert(&op);

    // Assign a ref for ops which need it.
    if (auto specOp = dyn_cast<UnaryDynInstDataOpInterface>(op)) {
      assert(ref);
      specOp.setPathOp(ref);
    }
  }

  inst.erase();
  return success();
}
void LowerInstancesPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  // Populate the top level symbol cache.
  topSyms.addDefinitions(top);

  size_t numFailed = 0;
  OpBuilder builder(ctxt);

  // Find all of the InstanceHierarchyOps.
  for (Operation &op : llvm::make_early_inc_range(top.getOps())) {
    auto instHierOp = dyn_cast<InstanceHierarchyOp>(op);
    if (!instHierOp)
      continue;
    builder.setInsertionPoint(&op);
    // Walk the child dynamic instances in _post-order_ so we lower and delete
    // the children first.
    instHierOp->walk<mlir::WalkOrder::PostOrder>([&](DynamicInstanceOp inst) {
      if (failed(lower(inst, instHierOp, builder)))
        ++numFailed;
    });
  }
  if (numFailed)
    signalPassFailure();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createLowerInstancesPass() {
  return std::make_unique<LowerInstancesPass>();
}
} // namespace msft
} // namespace circt
