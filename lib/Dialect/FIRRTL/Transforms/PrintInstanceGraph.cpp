//===- PrintInstanceGraph.cpp - Print the instance graph --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print the module hierarchy.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_PRINTINSTANCEGRAPH
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct PrintInstanceGraphPass
    : public circt::firrtl::impl::PrintInstanceGraphBase<
          PrintInstanceGraphPass> {
  PrintInstanceGraphPass() : os(llvm::errs()) {}
  void runOnOperation() override {
    auto circuitOp = getOperation();
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    llvm::WriteGraph(os, &instanceGraph, /*ShortNames=*/false,
                     circuitOp.getName());
    markAllAnalysesPreserved();
  }
  raw_ostream &os;
};
} // end anonymous namespace
