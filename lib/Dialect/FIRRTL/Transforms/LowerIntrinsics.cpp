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

static ParseResult hasNPorts(StringRef name, FExtModuleOp mod, unsigned n) {
  if (mod.getPorts().size() != n) {
    mod.emitError(name) << " has " << mod.getPorts().size()
                        << " ports instead of " << n;
    return failure();
  }
  return success();
}

static ParseResult namedPort(StringRef name, FExtModuleOp mod, unsigned n,
                             StringRef portName) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (!ports[n].getName().equals(portName)) {
    mod.emitError(name) << " port " << n << " named '" << ports[n].getName()
                        << "' instead of '" << portName << "'";
    return failure();
  }
  return success();
}

template <typename T>
static ParseResult sizedPort(StringRef name, FExtModuleOp mod, unsigned n,
                             int32_t size) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (!ports[n].type.isa<T>()) {
    mod.emitError(name) << " port " << n << " not a correct type";
    return failure();
  }
  if (ports[n].type.cast<T>().getWidth() != size) {
    mod.emitError(name) << " port " << n << " not size " << size;
    return failure();
  }
  return success();
}

static ParseResult hasNParam(StringRef name, FExtModuleOp mod, unsigned n) {
  unsigned num = 0;
  if (mod.getParameters())
    num = mod.getParameters().size();
  if (n != num) {
    mod.emitError(name) << " has " << num << " parameters instead of " << n;
    return failure();
  }
  return success();
}
static ParseResult namedParam(StringRef name, FExtModuleOp mod,
                              StringRef paramName) {
  for (auto a : mod.getParameters()) {
    auto param = a.cast<ParamDeclAttr>();
    if (param.getName().getValue().equals(paramName)) {
      if (param.getValue().isa<StringAttr>())
        return success();

      mod.emitError(name) << " test has parameter '" << param.getName()
                          << "' which should be a string but is not";
      return failure();
    }
  }
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

static bool lowerCirctSizeof(InstancePathCache &instancePathCache,
                             FExtModuleOp mod) {
  auto ports = mod.getPorts();
  if (hasNPorts("circt.sizeof", mod, 2) ||
      namedPort("circt.sizeof", mod, 0, "i") ||
      namedPort("circt.sizeof", mod, 1, "size") ||
      sizedPort<UIntType>("circt.sizeof", mod, 1, 32) ||
      hasNParam("circt.sizeof", mod, 0))
    return false;

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

static bool lowerCirctIsX(InstancePathCache &instancePathCache,
                          FExtModuleOp mod) {
  auto ports = mod.getPorts();
  if (hasNPorts("circt.isaX", mod, 2) || namedPort("circt.isX", mod, 0, "i") ||
      namedPort("circt.isX", mod, 1, "found") ||
      sizedPort<UIntType>("circt.isX", mod, 1, 1) ||
      hasNParam("circt.isX", mod, 0))
    return false;

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

static bool lowerCirctPlusArgTest(InstancePathCache &instancePathCache,
                                  FExtModuleOp mod) {
  if (hasNPorts("circt.plusargs.test", mod, 1) ||
      namedPort("circt.plusargs.test", mod, 0, "found") ||
      sizedPort<UIntType>("circt.plusargs.test", mod, 0, 1) ||
      hasNParam("circt.plusargs.test", mod, 1) ||
      namedParam("circt.plusargs.test", mod, "FORMAT"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();
  for (auto *use : instancePathCache.instanceGraph[mod]->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto newop = builder.create<PlusArgsTestIntrinsicOp>(
        param.getValue().cast<StringAttr>());
    inst.getResult(0).replaceAllUsesWith(newop);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPlusArgValue(InstancePathCache &instancePathCache,
                                   FExtModuleOp mod) {
  if (hasNPorts("circt.plusargs.value", mod, 2) ||
      namedPort("circt.plusargs.value", mod, 0, "found") ||
      namedPort("circt.plusargs.value", mod, 1, "result") ||
      sizedPort<UIntType>("circt.plusargs.value", mod, 0, 1) ||
      hasNParam("circt.plusargs.value", mod, 1) ||
      namedParam("circt.plusargs.value", mod, "FORMAT"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();

  for (auto *use : instancePathCache.instanceGraph[mod]->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto newop = builder.create<PlusArgsValueIntrinsicOp>(
        inst.getResultTypes(), param.getValue().cast<StringAttr>());
    inst.getResult(0).replaceAllUsesWith(newop.getFound());
    inst.getResult(1).replaceAllUsesWith(newop.getResult());
    inst.erase();
  }
  return true;
}

std::pair<const char *, std::function<bool(InstancePathCache &, FExtModuleOp)>>
    intrinsics[] = {
        {"circt.sizeof", lowerCirctSizeof},
        {"circt.isX", lowerCirctIsX},
        {"circt.plusargs.test", lowerCirctPlusArgTest},
        {"circt.plusargs.value", lowerCirctPlusArgValue},
};

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  size_t numFailures = 0;
  size_t numConverted = 0;
  InstancePathCache instancePathCache(getAnalysis<InstanceGraph>());
  for (auto op :
       llvm::make_early_inc_range(getOperation().getOps<FExtModuleOp>())) {
    auto anno = AnnotationSet(op).getAnnotation("circt.Intrinsic");
    if (!anno)
      continue;
    auto intname = anno.getMember<StringAttr>("intrinsic");
    if (!intname) {
      op.emitError("Intrinsic annotation with no intrinsic name");
      ++numFailures;
      continue;
    }
    bool found = false;
    for (const auto &intrinsic : intrinsics) {
      if (intname.getValue().equals(intrinsic.first)) {
        found = true;
        if (intrinsic.second(instancePathCache, op)) {
          ++numConverted;
          op.erase();
        } else {
          ++numFailures;
        }
        break;
      }
    }
    if (!found) {
      op.emitError("Unknown intrinsic '") << intname << "'";
      ++numFailures;
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
