//===- InnerSymbolDCE.cpp - Delete Unused Inner Symbols----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass removes inner symbols which have no uses.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

namespace {
struct InnerRefInfo {
  unsigned refcount = 0;
  Operation *op;
};
} // namespace

struct InnerSymbolDCEPass : public InnerSymbolDCEBase<InnerSymbolDCEPass> {
  void runOnOperation() override;
};

void InnerSymbolDCEPass::runOnOperation() {
  DenseMap<hw::InnerRefAttr, InnerRefInfo> innerRefMap;

  CircuitOp circuit = getOperation();

  auto examineAnnotation = [&](Annotation anno) -> void {
    anno.getDict().walkSubAttrs([&](Attribute attr) {
      auto innerRef = attr.dyn_cast<hw::InnerRefAttr>();
      if (!innerRef)
        return;
      innerRefMap[innerRef].refcount++;
    });
  };

  auto examineAnnotations = [&](Operation *op) -> void {
    for (auto anno : AnnotationSet(op))
      examineAnnotation(anno);
  };

  examineAnnotations(circuit);
  for (Operation &op : circuit.getBody()->getOperations()) {
    examineAnnotations(&op);

    // Extract any hw::InnerRefAttrs from NLAs.
    if (auto nla = dyn_cast<NonLocalAnchor>(op)) {
      for (auto path : nla.namepath()) {
        auto innerref = path.dyn_cast<hw::InnerRefAttr>();
        if (!innerref || !innerRefMap.count(innerref))
          continue;
        innerRefMap[innerref].refcount++;
      }
      continue;
    }

    // Only walk inside modules.
    auto mod = dyn_cast<FModuleOp>(op);
    if (!mod)
      continue;

    auto modNameAttr = mod.moduleNameAttr();
    mod.walk([&](Operation *op) {
      examineAnnotations(op);
      // If there is no symbol, then just continue.
      auto symbol = op->getAttrOfType<StringAttr>("inner_sym");
      if (!symbol)
        return WalkResult::advance();

      auto &info = innerRefMap[hw::InnerRefAttr::get(modNameAttr, symbol)];
      assert(!info.op && "inner ref map should not have an op at this point");
      info.op = op;

      // If the visibility is public, blindly increment the reference count.
      auto visibility = op->getAttrOfType<StringAttr>("inner_sym_visibility");
      if (!visibility)
        info.refcount++;
      return WalkResult::advance();
    });
  }

  for (auto keyValue : innerRefMap) {
    auto info = keyValue.second;
    // If the reference count is non-zero, then do nothing.  Otherwise, remove
    // the inner symbol from the operation.
    if (info.refcount)
      continue;
    info.op->removeAttr("inner_sym");
    info.op->removeAttr("inner_sym_visibility");
    ++numSymbolsRemoved;
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInnerSymbolDCEPass() {
  return std::make_unique<InnerSymbolDCEPass>();
}
