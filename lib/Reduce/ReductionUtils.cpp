//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Reduce/ReductionUtils.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Reduce/Reduction.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallSet.h"

using namespace circt;
using namespace circt::reduce;

void reduce::pruneUnusedOps(Operation *initialOp, Reduction &reduction) {
  SmallVector<Operation *> worklist;
  SmallSet<Operation *, 4> handled;
  worklist.push_back(initialOp);
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    if (!op->use_empty() || op->hasAttr("inner_sym"))
      continue;
    for (auto arg : op->getOperands())
      if (auto *argOp = arg.getDefiningOp())
        if (handled.insert(argOp).second)
          worklist.push_back(argOp);
    reduction.notifyOpErased(op);
    op->erase();
  }
}

//===----------------------------------------------------------------------===//
// InnerSymbolUses
//===----------------------------------------------------------------------===//

static StringAttr getSymbolName(Operation *op) {
  return op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

InnerSymbolUses::InnerSymbolUses(Operation *root) {
  root->walk([&](Operation *op) {
    auto collect = [&](Attribute attr) {
      if (auto symbolRef = dyn_cast<FlatSymbolRefAttr>(attr))
        symbolRefs.insert(symbolRef.getAttr());

      if (auto innerRef = dyn_cast<hw::InnerRefAttr>(attr)) {
        innerRefs.insert({innerRef.getModule(), innerRef.getName()});
        innerRefModules.insert(innerRef.getModule());
      }
    };
    for (auto namedAttr : op->getAttrs())
      namedAttr.getValue().walk(collect);
    for (auto result : op->getResults())
      result.getType().walk(collect);
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto arg : block.getArguments())
          arg.getType().walk(collect);
  });
}

bool InnerSymbolUses::hasInnerRef(Operation *op) const {
  if (auto symbol = getSymbolName(op))
    return hasInnerRef(symbol);

  if (auto innerSym = hw::InnerSymbolTable::getInnerSymbol(op)) {
    StringAttr symbol;
    auto *parent = op->getParentOp();
    while (parent && !(symbol = getSymbolName(parent)))
      parent = parent->getParentOp();
    if (symbol)
      return hasInnerRef(symbol, innerSym);
  }

  return false;
}

bool InnerSymbolUses::hasInnerRef(hw::InnerRefAttr innerRef) const {
  return innerRefs.contains({innerRef.getModule(), innerRef.getName()});
}

bool InnerSymbolUses::hasInnerRef(StringAttr symbol) const {
  return innerRefModules.contains(symbol);
}

bool InnerSymbolUses::hasInnerRef(StringAttr symbol,
                                  StringAttr innerSym) const {
  return innerRefs.contains({symbol, innerSym});
}

bool InnerSymbolUses::hasSymbolRef(Operation *op) const {
  return symbolRefs.contains(getSymbolName(op));
}

bool InnerSymbolUses::hasSymbolRef(StringAttr symbol) const {
  return symbolRefs.contains(symbol);
}

bool InnerSymbolUses::hasRef(Operation *op) const {
  return hasInnerRef(op) || hasSymbolRef(op);
}

bool InnerSymbolUses::hasRef(StringAttr symbol) const {
  return hasInnerRef(symbol) || hasSymbolRef(symbol);
}
