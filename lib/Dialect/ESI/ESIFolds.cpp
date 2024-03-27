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
  results.push_back(NullChannelAttr::get(
      getContext(), TypeAttr::get(getChanOutput().getType())));
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
LogicalResult UnwrapFIFOOp::canonicalize(UnwrapFIFOOp op,
                                         PatternRewriter &rewriter) {
  auto wrap = dyn_cast_or_null<WrapFIFOOp>(op.getChanInput().getDefiningOp());
  if (succeeded(UnwrapFIFOOp::mergeAndErase(op, wrap, rewriter)))
    return success();
  return failure();
}

LogicalResult WrapFIFOOp::fold(FoldAdaptor,
                               SmallVectorImpl<OpFoldResult> &results) {
  if (getChanOutput().getUsers().empty()) {
    results.push_back({});
    results.push_back(IntegerAttr::get(
        IntegerType::get(getContext(), 1, IntegerType::Signless), 0));
    return success();
  }
  return failure();
}

LogicalResult WrapFIFOOp::canonicalize(WrapFIFOOp op,
                                       PatternRewriter &rewriter) {
  auto unwrap =
      dyn_cast_or_null<UnwrapFIFOOp>(*op.getChanOutput().getUsers().begin());
  if (succeeded(UnwrapFIFOOp::mergeAndErase(unwrap, op, rewriter)))
    return success();
  return failure();
}

OpFoldResult WrapWindow::fold(FoldAdaptor) {
  if (auto unwrap = dyn_cast_or_null<UnwrapWindow>(getFrame().getDefiningOp()))
    return unwrap.getWindow();
  return {};
}
OpFoldResult UnwrapWindow::fold(FoldAdaptor) {
  if (auto wrap = dyn_cast_or_null<WrapWindow>(getWindow().getDefiningOp()))
    return wrap.getFrame();
  return {};
}

LogicalResult PackBundleOp::canonicalize(PackBundleOp pack,
                                         PatternRewriter &rewriter) {
  Value bundle = pack.getBundle();
  // This condition should be caught by the verifier, but we don't want to
  // crash if we assume it since canonicalize can get run on IR in a broken
  // state.
  if (!bundle.hasOneUse())
    return rewriter.notifyMatchFailure(pack,
                                       "bundle has zero or more than one user");

  // unpack(pack(x)) -> x
  auto unpack = dyn_cast<UnpackBundleOp>(*bundle.getUsers().begin());
  if (unpack) {
    for (auto [a, b] :
         llvm::zip_equal(pack.getToChannels(), unpack.getToChannels()))
      rewriter.replaceAllUsesWith(b, a);
    for (auto [a, b] :
         llvm::zip_equal(unpack.getFromChannels(), pack.getFromChannels()))
      rewriter.replaceAllUsesWith(b, a);
    rewriter.eraseOp(unpack);
    rewriter.eraseOp(pack);
    return success();
  }
  return rewriter.notifyMatchFailure(pack,
                                     "could not find corresponding unpack");
}

LogicalResult UnpackBundleOp::canonicalize(UnpackBundleOp unpack,
                                           PatternRewriter &rewriter) {
  Value bundle = unpack.getBundle();
  // This condition should be caught by the verifier, but we don't want to
  // crash if we assume it since canonicalize can get run on IR in a broken
  // state.
  if (!bundle.hasOneUse())
    return rewriter.notifyMatchFailure(unpack,
                                       "bundle has zero or more than one user");

  // Reuse pack's canonicalizer.
  auto pack = dyn_cast_or_null<PackBundleOp>(bundle.getDefiningOp());
  if (pack)
    return PackBundleOp::canonicalize(pack, rewriter);
  return rewriter.notifyMatchFailure(unpack,
                                     "could not find corresponding pack");
}
