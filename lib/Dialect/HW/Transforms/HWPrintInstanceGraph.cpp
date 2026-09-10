//===- HWPrintInstanceGraph.cpp - Print the instance graph ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print the module hierarchy.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_PRINTINSTANCEGRAPH
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct PrintInstanceGraphPass
    : public circt::hw::impl::PrintInstanceGraphBase<PrintInstanceGraphPass> {
  PrintInstanceGraphPass() : os(llvm::errs()) {}
  void runOnOperation() override {
    InstanceGraph &instanceGraph = getAnalysis<InstanceGraph>();
    llvm::WriteGraph(os, &instanceGraph, /*ShortNames=*/false);
    markAllAnalysesPreserved();
  }
  raw_ostream &os;
};
} // end anonymous namespace
