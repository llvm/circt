//===- PrintFIRRTLFieldSource.cpp - Print the Field Source ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print the FieldSource analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLFieldSource.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_PRINTFIRRTLFIELDSOURCEPASS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct PrintFIRRTLFieldSourcePass
    : public circt::firrtl::impl::PrintFIRRTLFieldSourcePassBase<
          PrintFIRRTLFieldSourcePass> {
  PrintFIRRTLFieldSourcePass(raw_ostream &os) : os(os) {}

  void visitValue(const FieldSource &fieldRefs, Value v) {
    auto *p = fieldRefs.nodeForValue(v);
    if (p) {
      if (p->isRoot())
        os << "ROOT: ";
      else
        os << "SUB:  " << v;
      os << " : " << p->src << " : {";
      llvm::interleaveComma(p->path, os);
      os << "} ";
      os << ((p->flow == Flow::Source) ? "Source"
             : (p->flow == Flow::Sink) ? "Sink"
                                       : "Duplex");
      if (p->isSrcWritable())
        os << " writable";
      if (p->isSrcTransparent())
        os << " transparent";
      os << "\n";
    }
  }

  void visitOp(const FieldSource &fieldRefs, Operation *op) {
    for (auto r : op->getResults())
      visitValue(fieldRefs, r);
    // recurse in to regions
    for (auto &r : op->getRegions())
      for (auto &b : r.getBlocks())
        for (auto &op : b)
          visitOp(fieldRefs, &op);
  }

  template <typename Op>
  void runOnOp(Op modOp) {
    Block *body = modOp.getBodyBlock();
    os << "** " << modOp.getName() << "\n";
    auto &fieldRefs = getAnalysis<FieldSource>();
    for (auto port : body->getArguments())
      visitValue(fieldRefs, port);
    for (auto &op : *body)
      visitOp(fieldRefs, &op);

    markAllAnalysesPreserved();
  }

  void runOnOperation() override {
    TypeSwitch<Operation *>(&(*getOperation()))
        .Case<FModuleOp, ClassOp, FormalOp>([&](auto op) { runOnOp(op); })
        // All other ops are ignored -- particularly ops that don't implement
        // the `getBodyBlock()` method. We don't want an error here because the
        // pass wasn't designed to run on those ops.
        .Default([&](auto) {});
  }
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createFIRRTLFieldSourcePass() {
  return std::make_unique<PrintFIRRTLFieldSourcePass>(llvm::errs());
}
