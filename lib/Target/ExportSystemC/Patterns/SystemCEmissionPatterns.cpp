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
#include "../EmissionPrinter.h"
#include "circt/Dialect/SystemC/SystemCOps.h"

using namespace circt;
using namespace circt::systemc;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit a SystemC module using the SC_MODULE macro and emit all ports as fields
/// of the module. Users of the ports request an expression to be inlined and we
/// simply return the name of the port.
struct SCModuleEmitter : OpEmissionPattern<SCModuleOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value, EmissionConfig &config) override {
    if (value.isa<BlockArgument>() &&
        value.getParentRegion()->getParentOfType<SCModuleOp>())
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionConfig &config,
                   EmissionPrinter &p) override {
    auto module = value.getParentRegion()->getParentOfType<SCModuleOp>();
    for (size_t i = 0, e = module.getNumArguments(); i < e; ++i) {
      if (module.getArgument(i) == value) {
        p << module.getPortNames()[i].cast<StringAttr>().getValue();
        return;
      }
    }
  }

  void emitStatement(SCModuleOp module, EmissionConfig &config,
                     EmissionPrinter &p) override {
    p << "\nSC_MODULE(" << module.getModuleName() << ") ";
    auto scope = p.getOstream().scope("{\n", "};\n");
    for (size_t i = 0, e = module.getNumArguments(); i < e; ++i) {
      p.emitType(module.getArgument(i).getType());

      auto portName = module.getPortNames()[i].cast<StringAttr>().getValue();
      p << " " << portName << ";\n";
    }

    p.emitRegion(module.getRegion(), scope);
  }
};

/// Emit the builtin module op by emitting all children in sequence. As a
/// result, we don't have to hard-code the behavior in ExportSytemC.
struct BuiltinModuleEmitter : OpEmissionPattern<ModuleOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(ModuleOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    auto scope = p.getOstream().scope("", "", false);
    p.emitRegion(op.getRegion(), scope);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateSystemCOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<BuiltinModuleEmitter, SCModuleEmitter>(context);
}

void circt::ExportSystemC::populateSystemCTypeEmitters(
    TypeEmissionPatternSet &patterns) {}
