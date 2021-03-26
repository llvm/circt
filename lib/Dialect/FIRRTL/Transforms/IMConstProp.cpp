//===- IMConstProp.cpp - Intermodule ConstProp and DCE ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct IMConstPropPass : public IMConstPropBase<IMConstPropPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void IMConstPropPass::runOnOperation() {}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMConstPropPass() {
  return std::make_unique<IMConstPropPass>();
}
