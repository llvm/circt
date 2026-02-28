//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_COMBTOSYNTH_H
#define CIRCT_CONVERSION_COMBTOSYNTH_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <string>

namespace circt {

enum CombToSynthTargetIR {
  // Lower to And-Inverter
  AIG,
  // Lower to Majority-Inverter
  MIG
};

#define GEN_PASS_DECL_CONVERTCOMBTOSYNTH
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CIRCT_CONVERSION_COMBTOSYNTH_H
