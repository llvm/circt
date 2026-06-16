//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::synth;
using namespace mlir;

//===----------------------------------------------------------------------===//
// MappingCostAttr
//===----------------------------------------------------------------------===//

static bool isNamedArcArray(ArrayAttr arcs) {
  return llvm::all_of(
      arcs, [](Attribute attr) { return isa<LinearTimingArcAttr>(attr); });
}

static bool isNamelessArcArray(ArrayAttr arcs) {
  return llvm::all_of(arcs, [](Attribute outputArcsAttr) {
    auto outputArcs = dyn_cast<ArrayAttr>(outputArcsAttr);
    if (!outputArcs)
      return false;

    return llvm::all_of(outputArcs, [](Attribute arcAttr) {
      auto arc = dyn_cast<ArrayAttr>(arcAttr);
      return arc && arc.size() == 3 && isa<IntegerAttr>(arc[0]) &&
             isa<IntegerAttr>(arc[1]) && isa<PolarityAttr>(arc[2]);
    });
  });
}

LogicalResult
MappingCostAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        FloatAttr area, ArrayAttr arcs,
                        DictionaryAttr inputCaps) {
  if (arcs.empty())
    return emitError() << "expected arcs to be non-empty";

  if (!isNamedArcArray(arcs) && !isNamelessArcArray(arcs))
    return emitError()
           << "expected arcs to contain either synth.linear_timing_arc "
              "attributes or nameless arc rows";

  if (inputCaps)
    for (auto entry : inputCaps)
      if (!isa<FloatAttr>(entry.getValue()))
        return emitError()
               << "expected input_caps values to be floating-point attributes";

  return success();
}
