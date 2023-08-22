//===- VerifyInnerRefNamespace.cpp - InnerRefNamespace verification Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple pass to drive verification of operations
// that want to be InnerRefNamespace's but don't have the trait to verify
// themselves.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWPasses.h"

/// VerifyInnerRefNamespace pass until have container operation.

namespace {

class VerifyInnerRefNamespacePass
    : public circt::hw::VerifyInnerRefNamespaceBase<
          VerifyInnerRefNamespacePass> {
public:
  void runOnOperation() override {
    auto *irnLike = getOperation();
    if (!irnLike->hasTrait<mlir::OpTrait::InnerRefNamespace>())
      if (failed(circt::hw::detail::verifyInnerRefNamespace(irnLike)))
        return signalPassFailure();

    return markAllAnalysesPreserved();
  };
  bool canScheduleOn(mlir::RegisteredOperationName opInfo) const override {
    return llvm::isa<circt::hw::InnerRefNamespaceLike>(opInfo);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::hw::createVerifyInnerRefNamespacePass() {
  return std::make_unique<VerifyInnerRefNamespacePass>();
}
