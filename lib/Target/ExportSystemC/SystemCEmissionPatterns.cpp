//===- SystemCEmissionPatterns.cpp - SystemC Dialect Emission Patterns ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the systemc dialect.
//
//===----------------------------------------------------------------------===//

#include "SystemCEmissionPatterns.h"
#include "EmissionPattern.h"
#include "EmissionPrinter.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::systemc;
using namespace circt::ExportSystemC;

static StringRef getSignalTypeString(Type type) {
  return TypeSwitch<Type, StringRef>(type)
      .Case<InputType>([](auto ty) { return "sc_in"; })
      .Case<OutputType>([](auto ty) { return "sc_out"; })
      .Case<InOutType>([](auto ty) { return "sc_inout"; })
      .Case<SignalType>([](auto ty) { return "sc_signal"; });
  llvm_unreachable("invalid type!");
}

static LogicalResult emitType(Type type, EmissionPrinter &p) {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<InputType, OutputType, InOutType, SignalType>([&](auto ty) {
        p << getSignalTypeString(ty) << "<";
        if (failed(emitType(ty.getBaseType(), p)))
          return failure();
        p << ">";
        return success();
      })
      .Case<IntegerType>([&](auto ty) {
        p << "sc_uint<" << ty.getIntOrFloatBitWidth() << ">";
        return success();
      })
      .Default([](auto ty) { return failure(); });
}

namespace {
struct SCModuleEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<SCModuleOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    auto module = value.getParentRegion()->getParentOfType<SCModuleOp>();
    for (size_t i = 0, e = module.getNumArguments(); i < e; ++i) {
      if (module.getArgument(i) == value)
        return EmissionResult(
            module.getPortNames()[i].cast<StringAttr>().getValue(),
            Precedence::VAR);
    }
    return EmissionResult();
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto module = cast<SCModuleOp>(op);
    p.emitIndent();
    p << "\nSC_MODULE(" << module.getModuleName() << ") {\n";
    p.increaseIndent();
    for (size_t i = 0, e = module.getNumArguments(); i < e; ++i) {
      p.emitIndent();
      if (failed(emitType(module.getArgument(i).getType(), p)))
        return mlir::emitError(module.getArgument(i).getLoc(),
                               "unsupported type");

      p << " " << module.getPortNames()[i].cast<StringAttr>().getValue()
        << ";\n";
    }
    p.decreaseIndent();

    if (failed(p.emitRegion(module.getRegion())))
      return failure();

    p.emitIndent();
    p << "}\n";
    return success();
  }
};

struct SignalWriteEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<SignalWriteOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    return EmissionResult();
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto signalWrite = cast<SignalWriteOp>(op);
    p.emitIndent();
    p << p.getExpression(signalWrite.getDest()).getExpressionString();

    if (config.getEmitExplicitReadWrite())
      p << ".write(";
    else
      p << " = ";

    p << p.getExpression(signalWrite.getSrc()).getExpressionString();

    if (config.getEmitExplicitReadWrite())
      p << ")";

    p << ";\n";

    return success();
  }
};

struct SignalReadEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<SignalReadOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    SignalReadOp readOp = value.getDefiningOp<SignalReadOp>();
    std::string input =
        p.getExpression(readOp.getInput()).getExpressionString();
    if (config.getEmitExplicitReadWrite())
      input += ".read()";
    return EmissionResult(input, Precedence::VAR);
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    return success();
  }
};

struct CtorEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<CtorOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    return EmissionResult();
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto ctorOp = cast<CtorOp>(op);
    p << "\n";
    p.emitIndent();
    p << "SC_CTOR(\"" << ctorOp->getParentOfType<SCModuleOp>().getModuleName()
      << "\") {\n";

    if (failed(p.emitRegion(ctorOp.getRegion())))
      return failure();

    p.emitIndent();
    p << "}\n";
    return success();
  }
};

struct SCFuncEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<SCFuncOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    SCFuncOp funcOp = value.getDefiningOp<SCFuncOp>();
    return EmissionResult(funcOp.getName(), Precedence::VAR);
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto funcOp = cast<SCFuncOp>(op);
    p << "\n";
    p.emitIndent();
    p << "void " << funcOp.getName() << "() {\n";

    if (failed(p.emitRegion(funcOp.getRegion())))
      return failure();

    p.emitIndent();
    p << "}\n";

    return success();
  }
};

struct MethodEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<MethodOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    return EmissionResult();
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto methodOp = cast<MethodOp>(op);
    p.emitIndent();
    p << "SC_METHOD("
      << p.getExpression(methodOp.getFuncHandle()).getExpressionString()
      << ");\n";

    return success();
  }
};

struct ThreadEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<ThreadOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    return EmissionResult();
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto threadOp = cast<ThreadOp>(op);
    p.emitIndent();
    p << "SC_THREAD("
      << p.getExpression(threadOp.getFuncHandle()).getExpressionString()
      << ");\n";

    return success();
  }
};

struct SignalEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<SignalOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    auto signalOp = value.getDefiningOp<SignalOp>();
    return EmissionResult(signalOp.getName(), Precedence::VAR);
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto signalOp = cast<SignalOp>(op);
    p.emitIndent();
    if (failed(emitType(signalOp.getSignal().getType(), p)))
      return failure();
    p << " " << signalOp.getName() << ";\n";

    return success();
  }
};

struct BuiltinModuleEmitter : EmissionPattern {
  using EmissionPattern::EmissionPattern;

  bool match(Operation *op, EmissionConfig &config) override {
    return isa<ModuleOp>(op);
  }

  EmissionResult getExpression(Value value, EmissionConfig &config,
                               EmissionPrinter &p) override {
    return EmissionResult();
  }

  LogicalResult emitStatement(Operation *op, EmissionConfig &config,
                              EmissionPrinter &p) override {
    auto moduleOp = cast<ModuleOp>(op);

    if (failed(p.emitRegion(moduleOp.getRegion(), false)))
      return failure();

    return success();
  }
};
} // namespace

void circt::ExportSystemC::populateSystemCEmitters(
    EmissionPatternSet &patterns) {
  patterns.add<BuiltinModuleEmitter, SCModuleEmitter, SignalWriteEmitter,
               SignalReadEmitter, CtorEmitter, SCFuncEmitter, MethodEmitter,
               ThreadEmitter, SignalEmitter>();
}
