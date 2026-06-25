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

LogicalResult
MappingCostAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        FloatAttr area, ArrayAttr arcs,
                        DictionaryAttr inputCaps) {
  if (arcs.empty())
    return emitError() << "expected arcs to be non-empty";

  for (auto attr : arcs)
    if (!isa<LinearTimingArcAttr>(attr))
      return emitError() << "expected arcs to contain synth.linear_timing_arc";

  if (inputCaps)
    for (auto entry : inputCaps)
      if (!isa<FloatAttr>(entry.getValue()))
        return emitError()
               << "expected input_caps values to be floating-point attributes";

  return success();
}
