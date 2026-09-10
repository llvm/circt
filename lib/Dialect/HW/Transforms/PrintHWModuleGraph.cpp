//===- PrintHWModuleGraph.cpp - Print the instance graph --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Prints an HW module as a .dot graph.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_PRINTHWMODULEGRAPH
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct PrintHWModuleGraphPass
    : public circt::hw::impl::PrintHWModuleGraphBase<PrintHWModuleGraphPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation().walk([&](hw::HWModuleOp module) {
      // We don't really have any other way of forwarding draw arguments to the
      // DOTGraphTraits for HWModule except through the module itself - as an
      // attribute.
      module->setAttr("dot_verboseEdges",
                      BoolAttr::get(module.getContext(), verboseEdges));

      llvm::WriteGraph(llvm::errs(), module, /*ShortNames=*/false);
    });
  }
};
} // end anonymous namespace
