//===- ReferRules.cpp - ReferRules Pass ----C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Use InstanceGraph and CallGraph to illustrate the method call information,
//
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/GAA/CallInfo.h"
#include "circt/Dialect/GAA/GAAOps.h"
#include "circt/Dialect/GAA/GAAPasses.h"
#include "circt/Dialect/GAA/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace gaa;
using namespace mlir;

class ReferRules : public ReferRulesBase<ReferRules> {
  void runOnOperation() override;

private:
};

void ReferRules::runOnOperation() { return; }

std::unique_ptr<mlir::Pass> circt::gaa::createReferRules() {
  return std::make_unique<ReferRules>();
}
