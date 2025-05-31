//===- Naming.cpp - Utilities for handling names ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Naming.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;

bool circt::isUselessName(StringRef name) {
  if (name.empty())
    return true;
  // Ignore _.*
  return name.starts_with("_T") || name.starts_with("_WIRE");
}

// Heuristic to pick the best name.
// Good names are not useless, don't start with an underscore, minimize
// underscores in them, and are short. This function deterministically favors
// the second name on ties.
static bool isNameBetter(StringRef a, StringRef b) {
  if (a.empty())
    return false;
  if (b.empty())
    return true;
  if (isUselessName(a))
    return false;
  if (isUselessName(b))
    return true;
  if (a.starts_with("_"))
    return false;
  if (b.starts_with("_"))
    return true;
  if (b.count('_') < a.count('_'))
    return false;
  if (b.count('_') > a.count('_'))
    return true;
  return a.size() <= b.size();
}

StringRef circt::chooseName(StringRef a, StringRef b) {
  return isNameBetter(a, b) ? a : b;
}

StringAttr circt::chooseName(StringAttr a, StringAttr b) {
  if (!a)
    return b;
  if (!b)
    return a;
  return isNameBetter(a.getValue(), b.getValue()) ? a : b;
}

static StringAttr getNameOrHint(Operation *a) {
  StringAttr name = a->getAttrOfType<StringAttr>("name");
  if (!name || name.getValue().empty())
    return a->getAttrOfType<StringAttr>("sv.namehint");
  return name;
}

StringAttr circt::chooseName(Operation *a, Operation *b) {
  return chooseName(getNameOrHint(a), getNameOrHint(b));
}

/// A wrapper of `PatternRewriter::replaceOp` to propagate "sv.namehint"
/// attribute. If a replaced op has a "sv.namehint" attribute, this function
/// propagates the name to the new value.
void circt::replaceOpAndCopyNamehint(PatternRewriter &rewriter, Operation *op,
                                     Value newValue) {
  if (auto *newOp = newValue.getDefiningOp()) {
    auto name = op->getAttrOfType<StringAttr>("sv.namehint");
    if (name && !newOp->hasAttr("sv.namehint"))
      rewriter.modifyOpInPlace(newOp,
                               [&] { newOp->setAttr("sv.namehint", name); });
  }
  rewriter.replaceOp(op, newValue);
}
