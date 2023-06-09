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

static ParseResult hasNPorts(StringRef name, FModuleLike mod, unsigned n) {
  if (mod.getPorts().size() != n) {
    mod.emitError(name) << " has " << mod.getPorts().size()
                        << " ports instead of " << n;
    return failure();
  }
  return success();
}

static ParseResult namedPort(StringRef name, FModuleLike mod, unsigned n,
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
static ParseResult typedPort(StringRef name, FModuleLike mod, unsigned n) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (!ports[n].type.isa<T>()) {
    mod.emitError(name) << " port " << n << " not of correct type";
    return failure();
  }
  return success();
}

template <typename T>
static ParseResult sizedPort(StringRef name, FModuleLike mod, unsigned n,
                             int32_t size) {
  auto ports = mod.getPorts();
  if (failed(typedPort<T>(name, mod, n)))
    return failure();
  if (ports[n].type.cast<T>().getWidth() != size) {
    mod.emitError(name) << " port " << n << " not size " << size;
    return failure();
  }
  return success();
}

static ParseResult hasNParam(StringRef name, FModuleLike mod, unsigned n,
                             unsigned c = 0) {
  unsigned num = 0;
  if (mod.getParameters())
    num = mod.getParameters().size();
  if (num < n || num > n + c) {
    auto d = mod.emitError(name) << " has " << num << " parameters instead of ";
    if (c == 0)
      d << n;
    else
      d << " between " << n << " and " << (n + c);
    return failure();
  }
  return success();
}

static ParseResult namedParam(StringRef name, FModuleLike mod,
                              StringRef paramName, bool optional = false) {
  for (auto a : mod.getParameters()) {
    auto param = a.cast<ParamDeclAttr>();
    if (param.getName().getValue().equals(paramName)) {
      if (param.getValue().isa<StringAttr>())
        return success();

      mod.emitError(name) << " has parameter '" << param.getName()
                          << "' which should be a string but is not";
      return failure();
    }
  }
  if (optional)
    return success();
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

static ParseResult namedIntParam(StringRef name, FModuleLike mod,
                                 StringRef paramName, bool optional = false) {
  for (auto a : mod.getParameters()) {
    auto param = a.cast<ParamDeclAttr>();
    if (param.getName().getValue().equals(paramName)) {
      if (param.getValue().isa<IntegerAttr>())
        return success();

      mod.emitError(name) << " has parameter '" << param.getName()
                          << "' which should be an integer but is not";
      return failure();
    }
  }
  if (optional)
    return success();
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

static InstanceGraphNode *lookupInstNode(InstancePathCache &instancePathCache,
                                         FModuleLike mod) {
  // Seems like you should be able to use a dyn_cast here, but alas
  if (isa<FIntModuleOp>(mod))
    return instancePathCache.instanceGraph[cast<FIntModuleOp>(mod)];
  return instancePathCache.instanceGraph[cast<FExtModuleOp>(mod)];
}

static bool lowerCirctSizeof(InstancePathCache &instancePathCache,
                             FModuleLike mod) {
  auto ports = mod.getPorts();
  if (hasNPorts("circt.sizeof", mod, 2) ||
      namedPort("circt.sizeof", mod, 0, "i") ||
      namedPort("circt.sizeof", mod, 1, "size") ||
      sizedPort<UIntType>("circt.sizeof", mod, 1, 32) ||
      hasNParam("circt.sizeof", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto inputWire = builder.create<WireOp>(ports[0].type).getResult();
    inst.getResult(0).replaceAllUsesWith(inputWire);
    auto size = builder.create<SizeOfIntrinsicOp>(inputWire);
    inst.getResult(1).replaceAllUsesWith(size);
    inst.erase();
  }
  return true;
}

static bool lowerCirctIsX(InstancePathCache &instancePathCache,
                          FModuleLike mod) {
  auto ports = mod.getPorts();
  if (hasNPorts("circt.isX", mod, 2) || namedPort("circt.isX", mod, 0, "i") ||
      namedPort("circt.isX", mod, 1, "found") ||
      sizedPort<UIntType>("circt.isX", mod, 1, 1) ||
      hasNParam("circt.isX", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto inputWire = builder.create<WireOp>(ports[0].type).getResult();
    inst.getResult(0).replaceAllUsesWith(inputWire);
    auto size = builder.create<IsXIntrinsicOp>(inputWire);
    inst.getResult(1).replaceAllUsesWith(size);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPlusArgTest(InstancePathCache &instancePathCache,
                                  FModuleLike mod) {
  if (hasNPorts("circt.plusargs.test", mod, 1) ||
      namedPort("circt.plusargs.test", mod, 0, "found") ||
      sizedPort<UIntType>("circt.plusargs.test", mod, 0, 1) ||
      hasNParam("circt.plusargs.test", mod, 1) ||
      namedParam("circt.plusargs.test", mod, "FORMAT"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();
  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
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
                                   FModuleLike mod) {
  if (hasNPorts("circt.plusargs.value", mod, 2) ||
      namedPort("circt.plusargs.value", mod, 0, "found") ||
      namedPort("circt.plusargs.value", mod, 1, "result") ||
      sizedPort<UIntType>("circt.plusargs.value", mod, 0, 1) ||
      hasNParam("circt.plusargs.value", mod, 1) ||
      namedParam("circt.plusargs.value", mod, "FORMAT"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
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

static bool lowerCirctClockGate(InstancePathCache &instancePathCache,
                                FModuleLike mod) {
  if (hasNPorts("circt.clock_gate", mod, 3) ||
      namedPort("circt.clock_gate", mod, 0, "in") ||
      namedPort("circt.clock_gate", mod, 1, "en") ||
      namedPort("circt.clock_gate", mod, 2, "out") ||
      typedPort<ClockType>("circt.clock_gate", mod, 0) ||
      sizedPort<UIntType>("circt.clock_gate", mod, 1, 1) ||
      typedPort<ClockType>("circt.clock_gate", mod, 2) ||
      hasNParam("circt.clock_gate", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto en = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(en);
    auto out = builder.create<ClockGateIntrinsicOp>(in, en, Value{});
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLAnd(InstancePathCache &instancePathCache,
                             FModuleLike mod) {
  if (hasNPorts("circt.ltl.and", mod, 3) ||
      namedPort("circt.ltl.and", mod, 0, "lhs") ||
      namedPort("circt.ltl.and", mod, 1, "rhs") ||
      namedPort("circt.ltl.and", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.and", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.and", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.and", mod, 2, 1) ||
      hasNParam("circt.ltl.and", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLAndIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLOr(InstancePathCache &instancePathCache,
                            FModuleLike mod) {
  if (hasNPorts("circt.ltl.or", mod, 3) ||
      namedPort("circt.ltl.or", mod, 0, "lhs") ||
      namedPort("circt.ltl.or", mod, 1, "rhs") ||
      namedPort("circt.ltl.or", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.or", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.or", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.or", mod, 2, 1) ||
      hasNParam("circt.ltl.or", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLOrIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLDelay(InstancePathCache &instancePathCache,
                               FModuleLike mod) {
  if (hasNPorts("circt.ltl.delay", mod, 2) ||
      namedPort("circt.ltl.delay", mod, 0, "in") ||
      namedPort("circt.ltl.delay", mod, 1, "out") ||
      sizedPort<UIntType>("circt.ltl.delay", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.delay", mod, 1, 1) ||
      hasNParam("circt.ltl.delay", mod, 1, 2) ||
      namedIntParam("circt.ltl.delay", mod, "delay") ||
      namedIntParam("circt.ltl.delay", mod, "length", true))
    return false;

  auto getI64Attr = [&](int64_t value) {
    return IntegerAttr::get(IntegerType::get(mod.getContext(), 64), value);
  };
  auto params = mod.getParameters();
  auto delay = getI64Attr(params[0]
                              .cast<ParamDeclAttr>()
                              .getValue()
                              .cast<IntegerAttr>()
                              .getValue()
                              .getZExtValue());
  IntegerAttr length;
  if (params.size() >= 2)
    if (auto lengthDecl = params[1].cast<ParamDeclAttr>())
      length = getI64Attr(
          lengthDecl.getValue().cast<IntegerAttr>().getValue().getZExtValue());

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    auto out =
        builder.create<LTLDelayIntrinsicOp>(in.getType(), in, delay, length);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLConcat(InstancePathCache &instancePathCache,
                                FModuleLike mod) {
  if (hasNPorts("circt.ltl.concat", mod, 3) ||
      namedPort("circt.ltl.concat", mod, 0, "lhs") ||
      namedPort("circt.ltl.concat", mod, 1, "rhs") ||
      namedPort("circt.ltl.concat", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.concat", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.concat", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.concat", mod, 2, 1) ||
      hasNParam("circt.ltl.concat", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLConcatIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLNot(InstancePathCache &instancePathCache,
                             FModuleLike mod) {
  if (hasNPorts("circt.ltl.not", mod, 2) ||
      namedPort("circt.ltl.not", mod, 0, "in") ||
      namedPort("circt.ltl.not", mod, 1, "out") ||
      sizedPort<UIntType>("circt.ltl.not", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.not", mod, 1, 1) ||
      hasNParam("circt.ltl.not", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto input =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(input);
    auto out = builder.create<LTLNotIntrinsicOp>(input.getType(), input);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLImplication(InstancePathCache &instancePathCache,
                                     FModuleLike mod) {
  if (hasNPorts("circt.ltl.implication", mod, 3) ||
      namedPort("circt.ltl.implication", mod, 0, "lhs") ||
      namedPort("circt.ltl.implication", mod, 1, "rhs") ||
      namedPort("circt.ltl.implication", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.implication", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.implication", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.implication", mod, 2, 1) ||
      hasNParam("circt.ltl.implication", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out =
        builder.create<LTLImplicationIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLEventually(InstancePathCache &instancePathCache,
                                    FModuleLike mod) {
  if (hasNPorts("circt.ltl.eventually", mod, 2) ||
      namedPort("circt.ltl.eventually", mod, 0, "in") ||
      namedPort("circt.ltl.eventually", mod, 1, "out") ||
      sizedPort<UIntType>("circt.ltl.eventually", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.eventually", mod, 1, 1) ||
      hasNParam("circt.ltl.eventually", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto input =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(input);
    auto out = builder.create<LTLEventuallyIntrinsicOp>(input.getType(), input);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLClock(InstancePathCache &instancePathCache,
                               FModuleLike mod) {
  if (hasNPorts("circt.ltl.clock", mod, 3) ||
      namedPort("circt.ltl.clock", mod, 0, "in") ||
      namedPort("circt.ltl.clock", mod, 1, "clock") ||
      namedPort("circt.ltl.clock", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.clock", mod, 0, 1) ||
      typedPort<ClockType>("circt.ltl.clock", mod, 1) ||
      sizedPort<UIntType>("circt.ltl.clock", mod, 2, 1) ||
      hasNParam("circt.ltl.clock", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto clock =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(clock);
    auto out = builder.create<LTLClockIntrinsicOp>(in.getType(), in, clock);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLDisable(InstancePathCache &instancePathCache,
                                 FModuleLike mod) {
  if (hasNPorts("circt.ltl.disable", mod, 3) ||
      namedPort("circt.ltl.disable", mod, 0, "in") ||
      namedPort("circt.ltl.disable", mod, 1, "condition") ||
      namedPort("circt.ltl.disable", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.disable", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.disable", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.disable", mod, 2, 1) ||
      hasNParam("circt.ltl.disable", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto condition =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(condition);
    auto out =
        builder.create<LTLDisableIntrinsicOp>(in.getType(), in, condition);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

template <class Op>
static bool lowerCirctVerif(InstancePathCache &instancePathCache,
                            FModuleLike mod) {
  if (hasNPorts("circt.verif.assert", mod, 1) ||
      namedPort("circt.verif.assert", mod, 0, "property") ||
      sizedPort<UIntType>("circt.verif.assert", mod, 0, 1) ||
      hasNParam("circt.verif.assert", mod, 0, 1) ||
      namedParam("circt.verif.assert", mod, "label", true))
    return false;

  auto params = mod.getParameters();
  StringAttr label;
  if (!params.empty())
    if (auto labelDecl = params[0].cast<ParamDeclAttr>())
      label = labelDecl.getValue().cast<StringAttr>();

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto property =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(property);
    builder.create<Op>(property, label);
    inst.erase();
  }
  return true;
}

std::pair<const char *, std::function<bool(InstancePathCache &, FModuleLike)>>
    intrinsics[] = {
        {"circt.sizeof", lowerCirctSizeof},
        {"circt_sizeof", lowerCirctSizeof},
        {"circt.isX", lowerCirctIsX},
        {"circt_isX", lowerCirctIsX},
        {"circt.plusargs.test", lowerCirctPlusArgTest},
        {"circt_plusargs_test", lowerCirctPlusArgTest},
        {"circt.plusargs.value", lowerCirctPlusArgValue},
        {"circt_plusargs_value", lowerCirctPlusArgValue},
        {"circt.clock_gate", lowerCirctClockGate},
        {"circt_clock_gate", lowerCirctClockGate},
        {"circt.ltl.and", lowerCirctLTLAnd},
        {"circt_ltl_and", lowerCirctLTLAnd},
        {"circt.ltl.or", lowerCirctLTLOr},
        {"circt_ltl_or", lowerCirctLTLOr},
        {"circt.ltl.delay", lowerCirctLTLDelay},
        {"circt_ltl_delay", lowerCirctLTLDelay},
        {"circt.ltl.concat", lowerCirctLTLConcat},
        {"circt_ltl_concat", lowerCirctLTLConcat},
        {"circt.ltl.not", lowerCirctLTLNot},
        {"circt_ltl_not", lowerCirctLTLNot},
        {"circt.ltl.implication", lowerCirctLTLImplication},
        {"circt_ltl_implication", lowerCirctLTLImplication},
        {"circt.ltl.eventually", lowerCirctLTLEventually},
        {"circt_ltl_eventually", lowerCirctLTLEventually},
        {"circt.ltl.clock", lowerCirctLTLClock},
        {"circt_ltl_clock", lowerCirctLTLClock},
        {"circt.ltl.disable", lowerCirctLTLDisable},
        {"circt_ltl_disable", lowerCirctLTLDisable},
        {"circt.verif.assert", lowerCirctVerif<VerifAssertIntrinsicOp>},
        {"circt_verif_assert", lowerCirctVerif<VerifAssertIntrinsicOp>},
        {"circt.verif.assume", lowerCirctVerif<VerifAssumeIntrinsicOp>},
        {"circt_verif_assume", lowerCirctVerif<VerifAssumeIntrinsicOp>},
        {"circt.verif.cover", lowerCirctVerif<VerifCoverIntrinsicOp>},
        {"circt_verif_cover", lowerCirctVerif<VerifCoverIntrinsicOp>},
};

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  size_t numFailures = 0;
  size_t numConverted = 0;
  InstancePathCache instancePathCache(getAnalysis<InstanceGraph>());
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    if (!isa<FExtModuleOp, FIntModuleOp>(op))
      continue;
    StringAttr intname;
    if (isa<FExtModuleOp>(op)) {
      auto anno = AnnotationSet(&op).getAnnotation("circt.Intrinsic");
      if (!anno)
        continue;
      intname = anno.getMember<StringAttr>("intrinsic");
      if (!intname) {
        op.emitError("intrinsic annotation with no intrinsic name");
        ++numFailures;
        continue;
      }
    } else {
      intname = cast<FIntModuleOp>(op).getIntrinsicAttr();
      if (!intname) {
        op.emitError("intrinsic module with no intrinsic name");
        ++numFailures;
        continue;
      }
    }

    bool found = false;
    for (const auto &intrinsic : intrinsics) {
      if (intname.getValue().equals(intrinsic.first)) {
        found = true;
        if (intrinsic.second(instancePathCache, cast<FModuleLike>(op))) {
          ++numConverted;
          op.erase();
        } else {
          ++numFailures;
        }
        break;
      }
    }
    if (!found) {
      op.emitError("unknown intrinsic: '") << intname.getValue() << "'";
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
