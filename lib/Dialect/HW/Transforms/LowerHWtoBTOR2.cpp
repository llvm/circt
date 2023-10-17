//===- LowerHWtoBTOR2.cpp - Print out a btor version of the module --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Converts a hw module to a btor file and prints it out
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace circt;
using namespace hw;

namespace {
// The goal here is to traverse the operations in order and convert them one by one into btor2
struct LowerHWtoBTOR2Pass : public LowerHWtoBTOR2Base<LowerHWtoBTOR2Pass> {

  // Executes the pass
  void runOnOperation() override;

};
} // end anonymous namespace

void LowerHWtoBTOR2Pass::runOnOperation() {
  getOperation().walk([&](hw::HWModuleOp module) {
  });
}

// Basic constructor for the pass
std::unique_ptr<mlir::Pass> circt::hw::createLowerHWtoBTOR2Pass() {
  return std::make_unique<LowerHWtoBTOR2Pass>();
}
