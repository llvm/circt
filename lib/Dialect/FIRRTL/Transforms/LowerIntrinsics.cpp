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
#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace firrtl;

namespace {

class CirctSizeofConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "i") || namedPort(1, "size") ||
           sizedPort<UIntType>(1, 32) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto inputTy = inst.getResult(0).getType();
    auto inputWire = builder.create<WireOp>(inputTy).getResult();
    inst.getResult(0).replaceAllUsesWith(inputWire);
    auto size = builder.create<SizeOfIntrinsicOp>(inputWire);
    inst.getResult(1).replaceAllUsesWith(size);
    inst.erase();
    return success();
  }
};

class CirctIsXConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "i") || namedPort(1, "found") ||
           sizedPort<UIntType>(1, 1) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto inputTy = inst.getResult(0).getType();
    auto inputWire = builder.create<WireOp>(inputTy).getResult();
    inst.getResult(0).replaceAllUsesWith(inputWire);
    auto size = builder.create<IsXIntrinsicOp>(inputWire);
    inst.getResult(1).replaceAllUsesWith(size);
    inst.erase();
    return success();
  }
};

class CirctPlusArgTestConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(1) || namedPort(0, "found") || sizedPort<UIntType>(0, 1) ||
           hasNParam(1) || namedParam("FORMAT");
  }

  LogicalResult convert(InstanceOp inst) override {
    auto param = cast<ParamDeclAttr>(mod.getParameters()[0]);
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto newop = builder.create<PlusArgsTestIntrinsicOp>(
        cast<StringAttr>(param.getValue()));
    inst.getResult(0).replaceAllUsesWith(newop);
    inst.erase();
    return success();
  }
};

class CirctPlusArgValueConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "found") || namedPort(1, "result") ||
           sizedPort<UIntType>(0, 1) || hasNParam(1) || namedParam("FORMAT");
  }

  LogicalResult convert(InstanceOp inst) override {
    auto param = cast<ParamDeclAttr>(mod.getParameters()[0]);
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto newop = builder.create<PlusArgsValueIntrinsicOp>(
        inst.getResultTypes(), cast<StringAttr>(param.getValue()));
    inst.getResult(0).replaceAllUsesWith(newop.getFound());
    inst.getResult(1).replaceAllUsesWith(newop.getResult());
    inst.erase();
    return success();
  }
};

class CirctClockGateConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "in") || namedPort(1, "en") ||
           namedPort(2, "out") || typedPort<ClockType>(0) ||
           sizedPort<UIntType>(1, 1) || typedPort<ClockType>(2) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto en = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(en);
    auto out = builder.create<ClockGateIntrinsicOp>(in, en, Value{});
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctClockInverterConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "in") || namedPort(1, "out") ||
           typedPort<ClockType>(0) || typedPort<ClockType>(1) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    auto out = builder.create<ClockInverterIntrinsicOp>(in);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class EICGWrapperToClockGateConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    if (!AnnotationSet(mod).empty()) {
      mod.emitError(name)
          << " cannot have annotations since it is an intrinsic";
      return true;
    }
    if (mod.getPorts().size() == 4) {
      return namedPort(0, "in") || namedPort(1, "test_en") ||
             namedPort(2, "en") || namedPort(3, "out") ||
             typedPort<ClockType>(0) || sizedPort<UIntType>(1, 1) ||
             sizedPort<UIntType>(2, 1) || typedPort<ClockType>(3) ||
             hasNParam(0);
    }
    if (mod.getPorts().size() == 3) {
      return namedPort(0, "in") || namedPort(1, "en") || namedPort(2, "out") ||
             typedPort<ClockType>(0) || sizedPort<UIntType>(1, 1) ||
             typedPort<ClockType>(2) || hasNParam(0);
    }
    mod.emitError(name) << " has " << mod.getPorts().size()
                        << " ports instead of 3 or 4";
    return true;
  }

  LogicalResult convert(InstanceOp inst) override {
    if (!AnnotationSet(inst).empty())
      return inst.emitError(name)
             << " instance cannot have annotations since it is an intrinsic";
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    if (mod.getPorts().size() == 4) {
      // Ports: in, test_en, en, out
      auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
      auto testEn =
          builder.create<WireOp>(inst.getResult(1).getType()).getResult();
      auto en = builder.create<WireOp>(inst.getResult(2).getType()).getResult();
      auto out = builder.create<ClockGateIntrinsicOp>(in, en, testEn);
      inst.getResult(0).replaceAllUsesWith(in);
      inst.getResult(1).replaceAllUsesWith(testEn);
      inst.getResult(2).replaceAllUsesWith(en);
      inst.getResult(3).replaceAllUsesWith(out);
    } else {
      // Ports: in, en, out
      auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
      auto en = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
      auto out = builder.create<ClockGateIntrinsicOp>(in, en, Value{});
      inst.getResult(0).replaceAllUsesWith(in);
      inst.getResult(1).replaceAllUsesWith(en);
      inst.getResult(2).replaceAllUsesWith(out);
    }
    inst.erase();
    return success();
  }
};

template <bool isMux2>
class CirctMuxCellConverter : public IntrinsicConverter {
private:
  static constexpr unsigned portNum = isMux2 ? 4 : 6;

public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    if (hasNPorts(portNum) || namedPort(0, "sel") || typedPort<UIntType>(0)) {
      return true;
    }
    if (isMux2) {
      if (namedPort(1, "high") || namedPort(2, "low") || namedPort(3, "out"))
        return true;
    } else {
      if (namedPort(1, "v3") || namedPort(2, "v2") || namedPort(3, "v1") ||
          namedPort(4, "v0") || namedPort(5, "out"))
        return true;
    }
    return false;
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    SmallVector<Value> operands;
    operands.reserve(portNum - 1);
    for (unsigned i = 0; i < portNum - 1; i++) {
      auto v = builder.create<WireOp>(inst.getResult(i).getType()).getResult();
      operands.push_back(v);
      inst.getResult(i).replaceAllUsesWith(v);
    }
    Value out;
    if (isMux2)
      out = builder.create<Mux2CellIntrinsicOp>(operands);
    else
      out = builder.create<Mux4CellIntrinsicOp>(operands);
    inst.getResult(portNum - 1).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLAndConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "lhs") || namedPort(1, "rhs") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           sizedPort<UIntType>(1, 1) || sizedPort<UIntType>(2, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLAndIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLOrConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "lhs") || namedPort(1, "rhs") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           sizedPort<UIntType>(1, 1) || sizedPort<UIntType>(2, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLOrIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLDelayConverter : public IntrinsicConverter {
public:
  CirctLTLDelayConverter(StringRef name, FModuleLike mod)
      : IntrinsicConverter(name, mod) {
    auto getI64Attr = [&](int64_t value) {
      return IntegerAttr::get(IntegerType::get(mod.getContext(), 64), value);
    };

    auto params = mod.getParameters();
    delay = getI64Attr(params[0]
                           .cast<ParamDeclAttr>()
                           .getValue()
                           .cast<IntegerAttr>()
                           .getValue()
                           .getZExtValue());

    if (params.size() >= 2)
      if (auto lengthDecl = cast<ParamDeclAttr>(params[1]))
        length = getI64Attr(
            cast<IntegerAttr>(lengthDecl.getValue()).getValue().getZExtValue());
  }

  bool check() override {
    return hasNPorts(2) || namedPort(0, "in") || namedPort(1, "out") ||
           sizedPort<UIntType>(0, 1) || sizedPort<UIntType>(1, 1) ||
           hasNParam(1, 2) || namedIntParam("delay") ||
           namedIntParam("length", true);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    auto out =
        builder.create<LTLDelayIntrinsicOp>(in.getType(), in, delay, length);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }

private:
  IntegerAttr length;
  IntegerAttr delay;
};

class CirctLTLConcatConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "lhs") || namedPort(1, "rhs") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           sizedPort<UIntType>(1, 1) || sizedPort<UIntType>(2, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLConcatIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLNotConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "in") || namedPort(1, "out") ||
           sizedPort<UIntType>(0, 1) || sizedPort<UIntType>(1, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto input =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(input);
    auto out = builder.create<LTLNotIntrinsicOp>(input.getType(), input);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLImplicationConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "lhs") || namedPort(1, "rhs") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           sizedPort<UIntType>(1, 1) || sizedPort<UIntType>(2, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out =
        builder.create<LTLImplicationIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLEventuallyConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "in") || namedPort(1, "out") ||
           sizedPort<UIntType>(0, 1) || sizedPort<UIntType>(1, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto input =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(input);
    auto out = builder.create<LTLEventuallyIntrinsicOp>(input.getType(), input);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLClockConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "in") || namedPort(1, "clock") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           typedPort<ClockType>(1) || sizedPort<UIntType>(2, 1) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto clock =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(clock);
    auto out = builder.create<LTLClockIntrinsicOp>(in.getType(), in, clock);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLDisableConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "in") || namedPort(1, "condition") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           sizedPort<UIntType>(1, 1) || sizedPort<UIntType>(2, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
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
    return success();
  }
};

template <class Op>
class CirctVerifConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(1) || namedPort(0, "property") ||
           sizedPort<UIntType>(0, 1) || hasNParam(0, 1) ||
           namedParam("label", true);
  }

  LogicalResult convert(InstanceOp inst) override {
    auto params = mod.getParameters();
    StringAttr label;
    if (!params.empty())
      if (auto labelDecl = cast<ParamDeclAttr>(params[0]))
        label = cast<StringAttr>(labelDecl.getValue());

    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto property =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(property);
    builder.create<Op>(property, label);
    inst.erase();
    return success();
  }
};

class CirctHasBeenResetConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "clock") || namedPort(1, "reset") ||
           namedPort(2, "out") || typedPort<ClockType>(0) || resetPort(1) ||
           sizedPort<UIntType>(2, 1) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto clock =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto reset =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(clock);
    inst.getResult(1).replaceAllUsesWith(reset);
    auto out = builder.create<HasBeenResetIntrinsicOp>(clock, reset);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctProbeConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "data") || namedPort(1, "clock") ||
           typedPort<ClockType>(1) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto clock =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto input =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(clock);
    inst.getResult(1).replaceAllUsesWith(input);
    builder.create<FPGAProbeIntrinsicOp>(clock, input);
    inst.erase();
    return success();
  }
};

} // namespace

// Replace range of values with new wires and return them.
template <typename R>
static SmallVector<Value> replaceResults(OpBuilder &b, R &&range) {
  return llvm::map_to_vector(range, [&b](auto v) {
    auto w = b.create<WireOp>(v.getLoc(), v.getType()).getResult();
    v.replaceAllUsesWith(w);
    return w;
  });
}

// Check ports are all inputs, emit diagnostic if not.
static ParseResult allInputs(ArrayRef<PortInfo> ports) {
  for (auto &p : ports) {
    if (p.direction != Direction::In)
      return mlir::emitError(p.loc, "expected input port");
  }
  return success();
}

// Get parameter by the given name.  Null if not found.
static ParamDeclAttr getNamedParam(ArrayAttr params, StringRef name) {
  for (auto param : params.getAsRange<ParamDeclAttr>())
    if (param.getName().getValue().equals(name))
      return param;
  return {};
}

namespace {

template <class OpTy, bool ifElseFatal = false>
class CirctAssertAssumeConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return namedPort(0, "clock") || typedPort<ClockType>(0) ||
           namedPort(1, "predicate") || sizedPort<UIntType>(1, 1) ||
           namedPort(2, "enable") || sizedPort<UIntType>(2, 1) ||
           namedParam("format", /*optional=*/true) ||
           namedParam("label", /*optional=*/true) ||
           namedParam("guards", /*optional=*/true) || allInputs(mod.getPorts());
    // TODO: Check all parameters accounted for.
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto params = mod.getParameters();
    auto format = getNamedParam(params, "format");
    auto label = getNamedParam(params, "label");
    auto guards = getNamedParam(params, "guards");

    auto wires = replaceResults(builder, inst.getResults());

    auto clock = wires[0];
    auto predicate = wires[1];
    auto enable = wires[2];

    auto substitutions = ArrayRef(wires).drop_front(3);
    auto name = label ? cast<StringAttr>(label.getValue()).strref() : "";
    // Message is not optional, so provide empty string if not present.
    auto message = format ? cast<StringAttr>(format.getValue())
                          : builder.getStringAttr("");
    auto op = builder.template create<OpTy>(clock, predicate, enable, message,
                                            substitutions, name,
                                            /*isConcurrent=*/true);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      cast<StringAttr>(guards.getValue()).strref().split(guardStrings, ';');
      // TODO: Legalize / sanity-check?
      op->setAttr("guards", builder.getStrArrayAttr(guardStrings));
    }

    if constexpr (ifElseFatal)
      op->setAttr("format", builder.getStringAttr("ifElseFatal"));

    inst.erase();
    return success();
  }

private:
};

class CirctCoverConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return namedPort(0, "clock") || typedPort<ClockType>(0) ||
           namedPort(1, "predicate") || sizedPort<UIntType>(1, 1) ||
           namedPort(2, "enable") || sizedPort<UIntType>(2, 1) ||
           hasNPorts(3) || allInputs(mod.getPorts()) ||
           namedParam("label", /*optional=*/true) ||
           namedParam("guards", /*optional=*/true);
    // TODO: Check all parameters accounted for.
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto params = mod.getParameters();
    auto label = getNamedParam(params, "label");
    auto guards = getNamedParam(params, "guards");

    auto wires = replaceResults(builder, inst.getResults());

    auto clock = wires[0];
    auto predicate = wires[1];
    auto enable = wires[2];

    auto name = label ? cast<StringAttr>(label.getValue()).strref() : "";

    // Empty message string for cover, only 'name' / label.
    auto op = builder.create<CoverOp>(clock, predicate, enable,
                                      builder.getStringAttr(""), ValueRange{},
                                      name, /*isConcurrent=*/true);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      cast<StringAttr>(guards.getValue()).strref().split(guardStrings, ';');
      // TODO: Legalize / sanity-check?
      op->setAttr("guards", builder.getStrArrayAttr(guardStrings));
    }

    inst.erase();
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass : public LowerIntrinsicsBase<LowerIntrinsicsPass> {
  void runOnOperation() override;
  using LowerIntrinsicsBase::fixupEICGWrapper;
};
} // namespace

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  IntrinsicLowerings lowering(&getContext(), getAnalysis<InstanceGraph>());
  lowering.add<CirctSizeofConverter>("circt.sizeof", "circt_sizeof");
  lowering.add<CirctIsXConverter>("circt.isX", "circt_isX");
  lowering.add<CirctPlusArgTestConverter>("circt.plusargs.test",
                                          "circt_plusargs_test");
  lowering.add<CirctPlusArgValueConverter>("circt.plusargs.value",
                                           "circt_plusargs_value");
  lowering.add<CirctClockGateConverter>("circt.clock_gate", "circt_clock_gate");
  lowering.add<CirctClockInverterConverter>("circt.clock_inv",
                                            "circt_clock_inv");
  lowering.add<CirctLTLAndConverter>("circt.ltl.and", "circt_ltl_and");
  lowering.add<CirctLTLOrConverter>("circt.ltl.or", "circt_ltl_or");
  lowering.add<CirctLTLDelayConverter>("circt.ltl.delay", "circt_ltl_delay");
  lowering.add<CirctLTLConcatConverter>("circt.ltl.concat", "circt_ltl_concat");
  lowering.add<CirctLTLNotConverter>("circt.ltl.not", "circt_ltl_not");
  lowering.add<CirctLTLImplicationConverter>("circt.ltl.implication",
                                             "circt_ltl_implication");
  lowering.add<CirctLTLEventuallyConverter>("circt.ltl.eventually",
                                            "circt_ltl_eventually");
  lowering.add<CirctLTLClockConverter>("circt.ltl.clock", "circt_ltl_clock");
  lowering.add<CirctLTLDisableConverter>("circt.ltl.disable",
                                         "circt_ltl_disable");
  lowering.add<CirctVerifConverter<VerifAssertIntrinsicOp>>(
      "circt.verif.assert", "circt_verif_assert");
  lowering.add<CirctVerifConverter<VerifAssumeIntrinsicOp>>(
      "circt.verif.assume", "circt_verif_assume");
  lowering.add<CirctVerifConverter<VerifCoverIntrinsicOp>>("circt.verif.cover",
                                                           "circt_verif_cover");
  lowering.add<CirctMuxCellConverter<true>>("circt.mux2cell", "circt_mux2cell");
  lowering.add<CirctMuxCellConverter<false>>("circt.mux4cell",
                                             "circt_mux4cell");
  lowering.add<CirctHasBeenResetConverter>("circt.has_been_reset",
                                           "circt_has_been_reset");
  lowering.add<CirctProbeConverter>("circt.fpga_probe", "circt_fpga_probe");
  lowering.add<CirctAssertAssumeConverter<AssertOp>>(
      "circt.chisel_assert_assume", "circt_chisel_assert_assume");
  lowering.add<CirctAssertAssumeConverter<AssertOp, /*ifElseFatal=*/true>>(
      "circt.chisel_ifelsefatal", "circt_chisel_ifelsefatal");
  lowering.add<CirctAssertAssumeConverter<AssumeOp>>("circt.chisel_assume",
                                                     "circt_chisel_assume");
  lowering.add<CirctCoverConverter>("circt.chisel_cover", "circt_chisel_cover");

  // Remove this once `EICG_wrapper` is no longer special-cased by firtool.
  if (fixupEICGWrapper)
    lowering.addExtmod<EICGWrapperToClockGateConverter>("EICG_wrapper");

  if (failed(lowering.lower(getOperation())))
    return signalPassFailure();
  if (!lowering.getNumConverted())
    markAllAnalysesPreserved();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerIntrinsicsPass(bool fixupEICGWrapper) {
  auto pass = std::make_unique<LowerIntrinsicsPass>();
  pass->fixupEICGWrapper = fixupEICGWrapper;
  return pass;
}
