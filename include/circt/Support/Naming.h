//===- Naming.h - Utilities for handling names ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_NAMING_H
#define CIRCT_SUPPORT_NAMING_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"

namespace circt {

/// Return true if this is a possibly useless temporary name.
/// This method is FIRRTL-centric, dropping useless temporaries.
bool isUselessName(StringRef name);

/// Choose a good name for an item from two options.
StringRef chooseName(StringRef a, StringRef b);

/// Choose a good name for an item from two options.
StringAttr chooseName(StringAttr a, StringAttr b);

/// Choose the better name between two ops. Picks the "name" attribute as first
/// preference, using "sv.namehint" as an alternative.
StringAttr chooseName(Operation *a, Operation *b);

/// A wrapper of `PatternRewriter::replaceOp` to propagate "sv.namehint"
/// attribute. If a replaced op has a "sv.namehint" attribute, this function
/// propagates the name to the new value.
void replaceOpAndCopyNamehint(PatternRewriter &rewriter, Operation *op,
                              Value newValue);

/// A wrapper of `PatternRewriter::replaceOpWithNewOp` to propagate
/// "sv.namehint" attribute. If a replaced op has a "sv.namehint" attribute,
/// this function propagates the name to the new value.
template <typename OpTy, typename... Args>
static OpTy replaceOpWithNewOpAndCopyNamehint(PatternRewriter &rewriter,
                                              Operation *op, Args &&...args) {
  auto name = op->getAttrOfType<StringAttr>("sv.namehint");
  auto newOp =
      rewriter.replaceOpWithNewOp<OpTy>(op, std::forward<Args>(args)...);
  if (name && !newOp->hasAttr("sv.namehint"))
    rewriter.modifyOpInPlace(newOp,
                             [&] { newOp->setAttr("sv.namehint", name); });

  return newOp;
}

} // namespace circt

#endif // CIRCT_SUPPORT_NAMING_H
