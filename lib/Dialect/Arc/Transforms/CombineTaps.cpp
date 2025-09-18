//===- CombineTaps.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/Pass/Pass.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_COMBINETAPS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace hw;

using llvm::SmallSetVector;

namespace {
struct CombineTapsPass : public arc::impl::CombineTapsBase<CombineTapsPass> {
  using CombineTapsBase::CombineTapsBase;

  void combineTaps(hw::HWModuleOp hwMod);

  void runOnOperation() override {
    for (auto hwMod : getOperation().getBody()->getOps<hw::HWModuleOp>())
      combineTaps(hwMod);
    return;
  }
};
} // namespace

void CombineTapsPass::combineTaps(hw::HWModuleOp hwMod) {
  // Collect Tap ops with the same SSA operand and merge them
  // into a single TapOp.
  for (auto tapOp : hwMod.getBodyBlock()->getOps<arc::TapOp>()) {
    SmallVector<TapOp> aliasTaps;
    for (auto user : tapOp.getOperand().getUsers()) {
      if (user == tapOp)
        continue;
      if (auto otherTap = dyn_cast<arc::TapOp>(user))
        aliasTaps.push_back(otherTap);
    }
    if (aliasTaps.empty())
      continue;
    SmallSetVector<Attribute, 8> names;
    aliasTaps.push_back(tapOp);
    for (auto mergeTap : aliasTaps) {
      for (auto nameAttr : mergeTap.getNames())
        names.insert(nameAttr);
      if (mergeTap != tapOp)
        mergeTap.erase();
    }
    tapOp.setNamesAttr(ArrayAttr::get(tapOp.getContext(), names.takeVector()));
  }
}
