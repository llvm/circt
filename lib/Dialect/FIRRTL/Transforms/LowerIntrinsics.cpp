//===- LowerIntrinsics.cpp - Lower Intrinsics -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntrinsics pass.  This pass processes FIRRTL
// extmodules with intrinsic annotations and rewrites the instances as
// appropriate.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace firrtl;

// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass : public LowerIntrinsicsBase<LowerIntrinsicsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static bool lowerCirctSizeof(InstancePathCache& instancePathCache, FExtModuleOp mod) {
      auto ports = mod.getPorts();
      if (ports.size() != 2) {
        mod.emitError("circt.sizeof does not have 2 ports");
        return false;
      }
      if (!ports[0].getName().equals("i")) {
        mod.emitError("circt.sizeof first port named '")
            << ports[0].getName() << "' instead of 'i'";
        return false;
      }
      if (!ports[1].getName().equals("size")) {
        mod.emitError("circt.sizeof second port named '")
            << ports[0].getName() << "' instead of 'size'";
return false;
      }
      if (!ports[1].type.isa<UIntType>()) {
        mod.emitError("circt.sizeof second port not a UInt<32>");
return false;
      }
      if (ports[1].type.cast<UIntType>().getWidth() != 32) {
        mod.emitError("circt.sizeof second port not a UInt<32>");
return false;
      }
      if (mod.getParameters() && mod.getParameters().size()) {
        mod.emitError("circt.sizeof has parameters");
        return false;
      }

      for (auto *use : instancePathCache.instanceGraph[mod]->uses()) {
        auto inst = cast<InstanceOp>(use->getInstance().getOperation());
        ImplicitLocOpBuilder builder(inst.getLoc(), inst);
        auto inputWire = builder.create<WireOp>(ports[0].type);
        inst.getResult(0).replaceAllUsesWith(inputWire);
        auto size = builder.create<SizeOfIntrinsicOp>(inputWire);
        inst.getResult(1).replaceAllUsesWith(size);
        inst.erase();
      }
  return true;

}

static bool lowerCirctIsX(InstancePathCache& instancePathCache, FExtModuleOp mod) {
      auto ports = mod.getPorts();
      if (ports.size() != 2) {
        mod.emitError("circt.sizeof does not have 2 ports");
        return false;
      }
      if (!ports[0].getName().equals("i")) {
        mod.emitError("circt.isX first port named '")
            << ports[0].getName() << "' instead of 'i'";
        return false;
      }
      if (!ports[1].getName().equals("found")) {
        mod.emitError("circt.isX second port named '")
            << ports[0].getName() << "' instead of 'found'";
return false;
      }
      if (!ports[1].type.isa<UIntType>()) {
        mod.emitError("circt.isX second port not a UInt<1>");
return false;
      }
      if (mod.getParameters() && mod.getParameters().size()) {
        mod.emitError("circt.isX has parameters");
        return false;
      }

      for (auto *use : instancePathCache.instanceGraph[mod]->uses()) {
        auto inst = cast<InstanceOp>(use->getInstance().getOperation());
        ImplicitLocOpBuilder builder(inst.getLoc(), inst);
        auto inputWire = builder.create<WireOp>(ports[0].type);
        inst.getResult(0).replaceAllUsesWith(inputWire);
        auto size = builder.create<IsXIntrinsicOp>(inputWire);
        inst.getResult(1).replaceAllUsesWith(size);
        inst.erase();
      }
  return true;

}
//      if (!mod.getParameters() || mod.getParameters().size() != 1) {
//        mod.emitError("circt.isX doesn't have a single parameter named FORMAT");
//        return false;
//      }


// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  size_t numFailures = 0;
  size_t numConverted = 0;
  InstancePathCache instancePathCache(getAnalysis<InstanceGraph>());
  for (auto op :
       llvm::make_early_inc_range(getOperation().getOps<FExtModuleOp>())) {
    auto anno = AnnotationSet(op).getAnnotation("circt.intrinsic");
    if (!anno)
      continue;
    auto intname = anno.getMember<StringAttr>("intrinsic");
    if (!intname) {
      op.emitError("Intrinsic annotation with no intrinsic name");
      ++numFailures;
      continue;
    }
    if (intname.getValue().equals("circt.sizeof")) {
      if (lowerCirctSizeof(instancePathCache, op)) {
        ++numConverted;      
        op.erase();
      } else {
      ++numFailures;
      }
      continue;
    }
        if (intname.getValue().equals("circt.isX")) {
      if (lowerCirctIsX(instancePathCache, op)) {
        ++numConverted;      
        op.erase();
      } else {
      ++numFailures;
      }
      continue;
    }

  }

  if (numFailures)
    signalPassFailure();
  if (!numConverted)
    markAllAnalysesPreserved();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerIntrinsicsPass() {
  return std::make_unique<LowerIntrinsicsPass>();
}
