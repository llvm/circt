//===- SVExtractTestCode.cpp - SV Simulation Extraction Pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass extracts simulation constructs to submodules.  It
// will take simulation operations, write, finish, assert, assume, and cover and
// extract them and the dataflow into them into a separate module.  This module
// is then instantiated in the original module.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace circt;
using namespace sv;

//===----------------------------------------------------------------------===//
// StubExternalModules Pass
//===----------------------------------------------------------------------===//

namespace {

struct SVEmitHeadersPass : public SVEmitHeaderBase<SVEmitHeadersPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void SVEmitHeadersPass::runOnOperation() {
  // Intentionally pass an UnknownLoc here so we don't get line number
  // comments on the output of this boilerplate in generated Verilog.
  ImplicitLocOpBuilder b(UnknownLoc::get(&getContext()), getOperation());
  b.setInsertionPointToStart(getOperation().getBody());

  DenseSet<StringAttr> macros;
  getOperation().walk([&](Operation *op) {
    for (NamedAttribute attr : op->getAttrs()) {
      if (auto ident = attr.getValue().dyn_cast<MacroIdentAttr>()) {
        macros.insert(ident.getIdent());
      }
    }
  });

  bool needsAnyDefinition = false;
  auto defined = [&](StringRef name) {
    if (macros.count(b.getStringAttr(name))) {
      needsAnyDefinition = true;
      return true;
    }
    return false;
  };

  bool usedRandomizeGarbageAssign = defined("RANDOMIZE_GARBAGE_ASSIGN");
  bool usedRandomizeRegInit = defined("RANDOMIZE_REG_INIT");
  bool usedRandomizeMemInit = defined("RANDOMIZE_MEM_INIT");
  bool usedAssertVerboseCond = defined("ASSERT_VERBOSE_COND_");
  bool usedPrintfCond = defined("PRINTF_COND_");
  bool usedStopCond = defined("STOP_COND_");
  if (!needsAnyDefinition)
    return;

  // TODO: We could have an operation for macros and uses of them, and
  // even turn them into symbols so we can DCE unused macro definitions.
  auto emitString = [&](StringRef verilogString) {
    b.create<sv::VerbatimOp>(verilogString);
  };

  // Helper function to emit a "#ifdef guard" with a `define in the then and
  // optionally in the else branch.
  auto emitGuardedDefine = [&](const char *guard, const char *defineTrue,
                               const char *defineFalse = nullptr) {
    std::string define = "`define ";
    if (!defineFalse) {
      assert(defineTrue && "didn't define anything");
      b.create<sv::IfDefOp>(guard, [&]() { emitString(define + defineTrue); });
    } else {
      b.create<sv::IfDefOp>(
          guard,
          [&]() {
            if (defineTrue)
              emitString(define + defineTrue);
          },
          [&]() { emitString(define + defineFalse); });
    }
  };

  emitString("// Standard header to adapt well known macros to our needs.");

  bool needRandom = false;
  if (usedRandomizeGarbageAssign) {
    emitGuardedDefine("RANDOMIZE_GARBAGE_ASSIGN", "RANDOMIZE");
    needRandom = true;
  }
  if (usedRandomizeRegInit) {
    emitGuardedDefine("RANDOMIZE_REG_INIT", "RANDOMIZE");
    needRandom = true;
  }
  if (usedRandomizeMemInit) {
    emitGuardedDefine("RANDOMIZE_MEM_INIT", "RANDOMIZE");
    needRandom = true;
  }

  if (needRandom) {
    emitString("\n// RANDOM may be set to an expression that produces a 32-bit "
               "random unsigned value.");
    emitGuardedDefine("RANDOM", nullptr, "RANDOM $random");
  }

  if (usedPrintfCond) {
    emitString("\n// Users can define 'PRINTF_COND' to add an extra gate to "
               "prints.");
    emitGuardedDefine("PRINTF_COND", "PRINTF_COND_ (`PRINTF_COND)",
                      "PRINTF_COND_ 1");
  }

  if (usedAssertVerboseCond) {
    emitString("\n// Users can define 'ASSERT_VERBOSE_COND' to add an extra "
               "gate to assert error printing.");
    emitGuardedDefine("ASSERT_VERBOSE_COND",
                      "ASSERT_VERBOSE_COND_ (`ASSERT_VERBOSE_COND)",
                      "ASSERT_VERBOSE_COND_ 1");
  }

  if (usedStopCond) {
    emitString("\n// Users can define 'STOP_COND' to add an extra gate "
               "to stop conditions.");
    emitGuardedDefine("STOP_COND", "STOP_COND_ (`STOP_COND)", "STOP_COND_ 1");
  }

  if (needRandom) {
    emitString("\n// Users can define INIT_RANDOM as general code that gets "
               "injected "
               "into the\n// initializer block for modules with registers.");
    emitGuardedDefine("INIT_RANDOM", nullptr, "INIT_RANDOM");

    emitString(
        "\n// If using random initialization, you can also define "
        "RANDOMIZE_DELAY to\n// customize the delay used, otherwise 0.002 "
        "is used.");
    emitGuardedDefine("RANDOMIZE_DELAY", nullptr, "RANDOMIZE_DELAY 0.002");

    emitString("\n// Define INIT_RANDOM_PROLOG_ for use in our modules below.");
    b.create<sv::IfDefOp>(
        "RANDOMIZE",
        [&]() {
          emitGuardedDefine(
              "VERILATOR", "INIT_RANDOM_PROLOG_ `INIT_RANDOM",
              "INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end");
        },
        [&]() { emitString("`define INIT_RANDOM_PROLOG_"); });
  }

  if (usedRandomizeGarbageAssign) {
    emitString("\n// RANDOMIZE_GARBAGE_ASSIGN enable range checks for mem "
               "assignments.");
    b.create<sv::IfDefOp>(
        "RANDOMIZE_GARBAGE_ASSIGN",
        [&]() {
          emitString(
              "`define RANDOMIZE_GARBAGE_ASSIGN_BOUND_CHECK(INDEX, VALUE, "
              "SIZE) \\");
          emitString("  ((INDEX) < (SIZE) ? (VALUE) : {`RANDOM})");
        },
        [&]() {
          emitString("`define RANDOMIZE_GARBAGE_ASSIGN_BOUND_CHECK(INDEX, "
                     "VALUE, SIZE) (VALUE)");
        });
  }

  // Blank line to separate the header from the modules.
  emitString("");
}

std::unique_ptr<Pass> circt::sv::createSVEmitHeaderPass() {
  return std::make_unique<SVEmitHeadersPass>();
}
