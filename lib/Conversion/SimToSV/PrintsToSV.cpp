//===- PrintsToSV.cpp - Sim to SV lowering --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/PrintsToSV.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "lower-prints-to-sv"

namespace circt {
#define GEN_PASS_DEF_LOWERPRINTSTOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace sim;

static void cleanUpFormatStringTree(ArrayRef<PrintFormattedProcOp> deadFmts) {
  SmallVector<Operation *> cleanupList;
  SmallVector<Operation *> cleanupNextList;
  SmallPtrSet<Operation *, 8> erasedOps;

  for (auto deadFmt : deadFmts) {
    cleanupNextList.push_back(deadFmt.getInput().getDefiningOp());
    erasedOps.insert(deadFmt);
    deadFmt.erase();
  }

  bool hasChanged = true;
  while (hasChanged && !cleanupNextList.empty()) {
    cleanupList = std::move(cleanupNextList);
    cleanupNextList.clear();
    hasChanged = false;

    for (auto op : cleanupList) {
      if (!op || erasedOps.contains(op))
        continue;

      if (auto concat = dyn_cast<FormatStringConcatOp>(op)) {
        if (!concat->use_empty()) {
          cleanupNextList.push_back(concat);
          continue;
        }
        for (auto arg : concat.getInputs())
          cleanupNextList.emplace_back(arg.getDefiningOp());
        hasChanged = true;
        erasedOps.insert(concat);
        concat.erase();
        continue;
      }

      if (isa<FormatBinOp, FormatHexOp, FormatDecOp, FormatCharOp, FormatLitOp>(
              op)) {
        if (op->use_empty()) {
          erasedOps.insert(op);
          op->erase();
        } else {
          cleanupNextList.push_back(op);
        }
        continue;
      }
    }
  }
}
struct PrintsToSVPass
    : public circt::impl::LowerPrintsToSVBase<PrintsToSVPass> {

  using circt::impl::LowerPrintsToSVBase<PrintsToSVPass>::LowerPrintsToSVBase;

  void runOnOperation() override {
    SmallVector<PrintFormattedProcOp> printCleanupList;

    bool hasFailed = false;
    getOperation().getBodyBlock()->walk([&](PrintFormattedProcOp printOp) {
      OpBuilder builder(printOp);
      if (failed(lowerProcPrint(builder, printOp)))
        hasFailed = true;
      printCleanupList.push_back(printOp);
    });

    if (hasFailed) {
      signalPassFailure();
      return;
    }
    cleanUpFormatStringTree(printCleanupList);
  };

private:
  LogicalResult lowerProcPrint(OpBuilder &builder,
                               PrintFormattedProcOp printOp);
};

LogicalResult PrintsToSVPass::lowerProcPrint(OpBuilder &builder,
                                             PrintFormattedProcOp printOp) {
  SmallVector<Value, 4> flatString;
  if (auto concat = printOp.getInput().getDefiningOp<FormatStringConcatOp>()) {
    auto isAcyclic = concat.getFlattenedInputs(flatString);
    if (failed(isAcyclic))
      return printOp.emitOpError("Format string is cyclic.");
  } else {
    flatString.push_back(printOp.getInput());
  }

  SmallString<64> fmtString;
  SmallVector<Value> substitutions;
  SmallVector<Location> locs;
  for (auto fmt : flatString) {
    auto defOp = fmt.getDefiningOp();
    if (!defOp)
      return printOp.emitError(
          "Formatting tokens must not be passed as arguments.");
    bool ok =
        llvm::TypeSwitch<Operation *, bool>(defOp)
            .Case<FormatLitOp>([&](auto literal) {
              fmtString.reserve(fmtString.size() + literal.getLiteral().size());
              for (auto c : literal.getLiteral()) {
                fmtString.push_back(c);
                if (c == '%')
                  fmtString.push_back('%');
              }
              return true;
            })
            .Case<FormatBinOp>([&](auto bin) {
              fmtString.push_back('%');
              fmtString.push_back('b');
              substitutions.push_back(bin.getValue());
              return true;
            })
            .Case<FormatDecOp>([&](auto dec) {
              fmtString.push_back('%');
              fmtString.push_back('d');
              Type ty = dec.getValue().getType();
              Value conv = builder.createOrFold<sv::SystemFunctionOp>(
                  dec.getLoc(), ty, dec.getIsSigned() ? "signed" : "unsigned",
                  dec.getValue());
              substitutions.push_back(conv);
              return true;
            })
            .Case<FormatHexOp>([&](auto hex) {
              fmtString.push_back('%');
              fmtString.push_back('x');
              substitutions.push_back(hex.getValue());
              return true;
            })
            .Case<FormatCharOp>([&](auto c) {
              fmtString.push_back('%');
              fmtString.push_back('c');
              substitutions.push_back(c.getValue());
              return true;
            })
            .Default([&](Operation *op) {
              op->emitError("Unsupported format specifier op.");
              return false;
            });
    if (!ok)
      return failure();
    locs.push_back(defOp->getLoc());
  }
  locs.push_back(printOp.getLoc());
  if (fmtString.empty())
    return success();

  auto fusedLoc = FusedLoc::get(builder.getContext(), locs);
  if (printToStdErr) {
    Value stdErr = builder.createOrFold<hw::ConstantOp>(
        printOp.getLoc(), builder.getI32IntegerAttr(0x80000002));
    builder.create<sv::FWriteOp>(
        fusedLoc, stdErr, builder.getStringAttr(fmtString), substitutions);
  } else {
    bool implicitNewline = (fmtString.back() == '\n');
    if (implicitNewline)
      fmtString.pop_back();
    builder.create<sv::DisplayOp>(fusedLoc, builder.getStringAttr(fmtString),
                                  substitutions, !implicitNewline);
  }
  return success();
}
