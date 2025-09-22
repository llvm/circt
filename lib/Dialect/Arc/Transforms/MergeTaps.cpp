//===- MergeTaps.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_MERGETAPS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct MergeTapsPass : public arc::impl::MergeTapsBase<MergeTapsPass> {
  using MergeTapsBase::MergeTapsBase;

  void mergeTaps();

  void runOnOperation() override { mergeTaps(); }
};
} // namespace

void MergeTapsPass::mergeTaps() {
  // Collect Tap ops with the same SSA operand and merge them
  // into a single TapOp. Erases TapOps while traversing the
  // module body, but will never erase the op currently visited.
  for (auto tapOp : getOperation().getBodyBlock()->getOps<arc::TapOp>()) {
    SmallVector<TapOp> aliasTaps;
    for (auto user : tapOp.getOperand().getUsers()) {
      if (user == tapOp)
        continue;
      if (auto otherTap = dyn_cast<arc::TapOp>(user)) {
        // Don't combine taps across blocks
        if (tapOp->getBlock() == otherTap->getBlock())
          aliasTaps.push_back(otherTap);
      }
    }
    if (aliasTaps.empty())
      continue;
    // Collect the names and erase all other taps
    SmallVector<Attribute> names;
    aliasTaps.push_back(tapOp);
    for (auto mergeTap : aliasTaps) {
      for (auto nameAttr : mergeTap.getNames())
        names.push_back(nameAttr);
      if (mergeTap != tapOp)
        mergeTap.erase();
    }
    // Sort the names and remove duplicates
    llvm::sort(names, [](Attribute a, Attribute b) {
      return cast<StringAttr>(a).getValue() < cast<StringAttr>(b).getValue();
    });
    names.erase(llvm::unique(names), names.end());
    tapOp.setNamesAttr(ArrayAttr::get(tapOp.getContext(), names));
  }
}
