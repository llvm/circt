//===- AffineToStaticlogic.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToStaticLogic/AffineToStaticLogic.h"
#include "../PassDetail.h"

using namespace circt;

namespace {

struct AffineToStaticLogic
    : public AffineToStaticLogicBase<AffineToStaticLogic> {
  void runOnFunction() override;
};

} // namespace

void AffineToStaticLogic::runOnFunction() {}

std::unique_ptr<mlir::Pass> circt::createAffineToStaticLogic() {
  return std::make_unique<AffineToStaticLogic>();
}
