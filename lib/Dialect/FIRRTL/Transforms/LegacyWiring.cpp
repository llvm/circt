//===- LegacyWiring- legacy Wiring annotation resolver ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the legacy Wiring annotation resolver.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LogicalResult.h"

using namespace circt;
using namespace firrtl;

/// Consume SourceAnnotation and SinkAnnotation, storing into state
LogicalResult circt::firrtl::applyWiring(const AnnoPathValue &target,
                                         DictionaryAttr anno,
                                         ApplyState &state) {
  auto *context = anno.getContext();
  ImplicitLocOpBuilder builder(target.ref.getOp()->getLoc(), context);

  // Convert target to Value
  SmallVector<Value> targetsValues;
  if (auto portTarget = target.ref.dyn_cast<PortAnnoTarget>()) {
    auto portNum = portTarget.getImpl().getPortNo();
    if (auto module = dyn_cast<FModuleOp>(portTarget.getOp())) {
      builder.setInsertionPointToEnd(module.getBodyBlock());
      targetsValues.push_back(getValueByFieldID(
          builder, module.getArgument(portNum), target.fieldIdx));
    } else if (auto ext = dyn_cast<FExtModuleOp>(portTarget.getOp())) {
      if (target.instances.empty()) {
        auto paths = state.instancePathCache.getAbsolutePaths(ext);
        if (paths.size() > 1) {
          mlir::emitError(state.circuit.getLoc())
              << "cannot resolve a unique instance path from the "
                 "external module target"
              << target.ref;
          return failure();
        }
        for (auto path : paths.back()) {
          auto inst = cast<InstanceOp>(path);
          builder.setInsertionPointAfter(inst);
          targetsValues.push_back(getValueByFieldID(
              builder, inst->getResult(portNum), target.fieldIdx));
        }
      } else {
        for (auto &inst : target.instances) {
          builder.setInsertionPointAfter(inst);
          targetsValues.push_back(getValueByFieldID(
              builder, inst->getResult(portNum), target.fieldIdx));
        }
      }
    } else {
      return mlir::emitError(state.circuit.getLoc())
             << "Annotation has invalid target: " << anno;
    }
  } else if (auto opResult = target.ref.dyn_cast<OpAnnoTarget>()) {
    if (target.isOpOfType<WireOp, RegOp, RegResetOp>()) {
      auto module = cast<FModuleOp>(opResult.getModule());
      builder.setInsertionPointToEnd(module.getBodyBlock());
      if (opResult.getOp()->getNumResults() != 1) {
        // Unsure about what ops emit more/less than 1 result, perhaps the type
        // check is enough
        return failure();
      }
      targetsValues.push_back(getValueByFieldID(
          builder, opResult.getOp()->getResult(0), target.fieldIdx));
    } else {
      return mlir::emitError(state.circuit.getLoc())
             << "Annotation targets non-wireable operation: " << anno;
    }
  } else {
    return mlir::emitError(state.circuit.getLoc())
           << "Annotation has invalid target: " << anno;
  }

  // Get pin field
  StringRef pin;
  if (auto pinName = anno.getNamed("pin")) {
    pin = pinName->getValue().cast<StringAttr>().getValue();
  } else {
    return mlir::emitError(state.circuit.getLoc())
           << "Annotation does not have an associated pin name: " << anno;
  }

  // Handle difference between sinks and sources
  auto clazz = anno.getAs<StringAttr>("class").getValue();
  if (clazz == wiringSourceAnnoClass) {
    // Record any new SourceAnnotations
    if (state.legacyWiringSources.find(pin) !=
            state.legacyWiringSources.end() ||
        targetsValues.size() != 1) {
      return mlir::emitError(state.circuit.getLoc())
             << "More than one " << wiringSourceAnnoClass << " defined for pin "
             << pin;
    }
    state.legacyWiringSources[pin] = targetsValues[0];
  } else if (clazz == wiringSinkAnnoClass) {
    // Record any new SinkAnnotations
    state.legacyWiringSinks[pin].insert(state.legacyWiringSinks[pin].end(),
                                        targetsValues.begin(),
                                        targetsValues.end());
  }

  return success();
}
