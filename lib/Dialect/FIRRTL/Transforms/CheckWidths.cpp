//===- CheckWidths.cpp - Check that width of types are known ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CheckWidths pass.
//
// Modeled after: <https://github.com/chipsalliance/firrtl/blob/master/src/main/
// scala/firrtl/passes/CheckWidths.scala>
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"

using namespace circt;
using namespace firrtl;

namespace {
/// A simple pass that emits errors for any occurrences of `uint`, `sint`, or
/// `analog` without a known width.
class CheckWidthsPass : public CheckWidthsBase<CheckWidthsPass> {
  void runOnOperation() override;
  void checkType(Type type, Type reportType, Location loc,
                 llvm::function_ref<void(InFlightDiagnostic &)> addContext);

  InFlightDiagnostic emitError(Location loc, const Twine &message) {
    anyFailed = true;
    signalPassFailure();
    return mlir::emitError(loc, message);
  }

  /// A set of already-checked types, to avoid unnecessary work.
  llvm::DenseSet<Type> checkedTypes;

  /// Whether any checks have failed.
  bool anyFailed;
};
} // namespace

void CheckWidthsPass::runOnOperation() {
  FModuleOp module = getOperation();
  anyFailed = false;

  // Check the port types. Unfortunately we don't have an exact location of the
  // port name, so we just point at the module with the port name mentioned in
  // the error.
  for (auto &port : module.getPorts()) {
    checkType(
        port.type, port.type, module.getLoc(), [&](InFlightDiagnostic &error) {
          error.attachNote() << "in port `" << port.getName() << "` of module `"
                             << module.getName() << "`";
        });
    if (anyFailed)
      break;
  }

  // Check the results of each operation.
  module->walk([&](Operation *op) {
    for (auto type : op->getResultTypes())
      checkType(type, type, op->getLoc(), [&](InFlightDiagnostic &error) {
        error.attachNote() << "in result of `" << op->getName() << "`";
      });
    if (anyFailed)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  // Cleanup.
  checkedTypes.clear();
  markAllAnalysesPreserved();
}

/// Check a type and its children for any occurrences of uninferred widths.
/// Specifically look for `uint`, `sint`, and `analog` types. The argument
/// `type` is the one being checked, `reportType` is the one included in any
/// errors, `loc` is the location to report errors to, and `addContext` gives
/// the caller the chance to add additional context information to the error.
void CheckWidthsPass::checkType(
    Type type, Type reportType, Location loc,
    llvm::function_ref<void(InFlightDiagnostic &)> addContext) {
  // Add the type to the already-checked set. If an entry was already there,
  // don't do the work twice.
  if (!checkedTypes.insert(type).second)
    return;

  TypeSwitch<Type>(type)
      // Primarily complain about integer and analog types not having a proper
      // width.
      .template Case<IntType, AnalogType>([&](auto type) {
        if (!type.hasWidth()) {
          auto error = emitError(loc, "uninferred width: type ")
                       << reportType << " has no known width";
          addContext(error);
        }
      })
      // Look into `flip<...>` and vector types, but report the entire flip or
      // vector type upon failure. So `flip<uint>` will have the error mention
      // `flip<uint>` instead of just `uint`.
      .template Case<FlipType, FVectorType>([&](auto type) {
        checkType(type.getElementType(), reportType, loc, addContext);
      })
      // Look into bundle types. Report only the type of the offending field
      // upon error, and add a note indicating which field was problematic.
      .template Case<BundleType>([&](auto type) {
        for (auto &bundleElement : type.getElements())
          checkType(bundleElement.type, bundleElement.type, loc,
                    [&](InFlightDiagnostic &error) {
                      addContext(error);
                      error.attachNote()
                          << "in bundle field `"
                          << bundleElement.name.getValue() << "`";
                    });
      });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckWidthsPass() {
  return std::make_unique<CheckWidthsPass>();
}
