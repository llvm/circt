//===- GrandCentralDataTaps.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GrandCentralDataTaps pass.
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

class GrandCentralDataTapsPass
    : public GrandCentralDataTapsBase<GrandCentralDataTapsPass> {
  void runOnOperation() override;
};

void GrandCentralDataTapsPass::runOnOperation() {
  // TODO: Do magic here
  LLVM_DEBUG(llvm::dbgs() << "Running the GCT Data Taps pass\n");
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralDataTapsPass() {
  return std::make_unique<GrandCentralDataTapsPass>();
}
