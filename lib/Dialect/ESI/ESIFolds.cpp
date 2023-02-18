//===- ESIFolds.cpp - ESI op folders ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace circt::esi;

LogicalResult WrapValidReadyOp::fold(FoldAdaptor,
                                     SmallVectorImpl<OpFoldResult> &results) {
  if (!getChanOutput().getUsers().empty())
    return failure();
  results.push_back(mlir::UnitAttr::get(getContext()));
  results.push_back(IntegerAttr::get(IntegerType::get(getContext(), 1), 1));
  return success();
}

LogicalResult UnwrapFIFOOp::mergeAndErase(UnwrapFIFOOp unwrap, WrapFIFOOp wrap,
                                          PatternRewriter &rewriter) {
  if (unwrap && wrap) {
    rewriter.replaceOp(unwrap, {wrap.getData(), wrap.getEmpty()});
    rewriter.replaceOp(wrap, {{}, unwrap.getRden()});
    return success();
  }
  return failure();
}
LogicalResult UnwrapFIFOOp::canonicalize(UnwrapFIFOOp unwrap,
                                         PatternRewriter &rewriter) {
  auto wrap =
      dyn_cast_or_null<WrapFIFOOp>(unwrap.getChanInput().getDefiningOp());
  if (succeeded(UnwrapFIFOOp::mergeAndErase(unwrap, wrap, rewriter)))
    return success();
  return failure();
}

LogicalResult WrapFIFOOp::canonicalize(WrapFIFOOp wrap,
                                       PatternRewriter &rewriter) {
  if (wrap.getChanOutput().getUsers().empty()) {
    auto c0_i1 = rewriter.create<hw::ConstantOp>(
        wrap.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
    rewriter.replaceOp(wrap, {{}, c0_i1});
    return success();
  }
  auto unwrap =
      dyn_cast_or_null<UnwrapFIFOOp>(*wrap.getChanOutput().getUsers().begin());
  if (succeeded(UnwrapFIFOOp::mergeAndErase(unwrap, wrap, rewriter)))
    return success();
  return failure();
}
