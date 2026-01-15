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

LogicalResult WrapValidReadyOp::canonicalize(WrapValidReadyOp op,
                                             PatternRewriter &rewriter) {
  // If the channel has no consumers but the ready signal does, replace ready
  // with a constant true (always ready to accept data).
  if (op.getChanOutput().use_empty() && !op.getReady().use_empty()) {
    auto trueConst =
        hw::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI1Type(), 1);
    rewriter.replaceAllUsesWith(op.getReady(), trueConst);
    rewriter.eraseOp(op);
    return success();
  }

  // If the wrap has no users at all, just erase it.
  if (op->use_empty()) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
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

LogicalResult WrapFIFOOp::canonicalize(WrapFIFOOp op,
                                       PatternRewriter &rewriter) {
  // If the channel has no consumers but the rden signal does, replace rden
  // with a constant false (not reading since there's no consumer).
  if (op.getChanOutput().use_empty() && !op.getRden().use_empty()) {
    auto falseConst =
        hw::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI1Type(), 0);
    rewriter.replaceAllUsesWith(op.getRden(), falseConst);
    rewriter.eraseOp(op);
    return success();
  }

  // If the wrap has no users at all, just erase it.
  if (op->use_empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  // Existing wrap-unwrap canonicalization logic
  if (!op.getChanOutput().hasOneUse())
    return rewriter.notifyMatchFailure(
        op, "channel output doesn't have exactly one use");
  auto unwrap = dyn_cast_or_null<UnwrapFIFOOp>(
      op.getChanOutput().getUses().begin()->getOwner());
  if (succeeded(UnwrapFIFOOp::mergeAndErase(unwrap, op, rewriter)))
    return success();
  return rewriter.notifyMatchFailure(
      op, "could not find corresponding unwrap for wrap");
}

OpFoldResult WrapWindow::fold(FoldAdaptor) {
  if (auto unwrap = dyn_cast_or_null<UnwrapWindow>(getFrame().getDefiningOp()))
    return unwrap.getWindow();
  return {};
}
LogicalResult WrapWindow::canonicalize(WrapWindow op,
                                       PatternRewriter &rewriter) {
  // Loop over all users and replace the frame of all UnwrapWindow users. Delete
  // op if no users remain.
  bool edited = false;
  bool allUsersAreUnwraps = true;
  for (auto &use : llvm::make_early_inc_range(op.getWindow().getUses())) {
    if (auto unwrap = dyn_cast<UnwrapWindow>(use.getOwner())) {
      rewriter.replaceOp(unwrap, op.getFrame());
      edited = true;
    } else {
      allUsersAreUnwraps = false;
    }
  }
  if (allUsersAreUnwraps || op.getWindow().getUses().empty()) {
    rewriter.eraseOp(op);
    edited = true;
  }
  return success(edited);
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
