//===- RegisterAllEmitters.h - Register all emitters to ExportSystemC -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This registers the all the emitters of various dialects to the
// ExportSystemC pass.
//
//===----------------------------------------------------------------------===//

#ifndef REGISTERALLEMITTERS_H
#define REGISTERALLEMITTERS_H

#include "EmissionPattern.h"
#include "Patterns/CombEmissionPatterns.h"
#include "Patterns/HWEmissionPatterns.h"
#include "Patterns/SCFEmissionPatterns.h"
#include "Patterns/SystemCEmissionPatterns.h"

namespace circt {
namespace ExportSystemC {

inline void registerAllEmitters(EmissionPatternSet &patterns) {
  populateCombEmitters(patterns);
  populateHWEmitters(patterns);
  populateSCFEmitters(patterns);
  populateSystemCEmitters(patterns);
}

} // namespace ExportSystemC
} // namespace circt

#endif // REGISTERALLEMITTERS_H
