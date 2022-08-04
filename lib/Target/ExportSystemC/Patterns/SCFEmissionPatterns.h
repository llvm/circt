//===- SCFEmissionPatterns.h - SCF Dialect Emission Patterns --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This exposes the emission patterns of the SCF dialect for registration.
//
//===----------------------------------------------------------------------===//

#ifndef SCFEMISSIONPATTERNS_H
#define SCFEMISSIONPATTERNS_H

#include "../EmissionPattern.h"

namespace circt {
namespace ExportSystemC {
void populateSCFEmitters(EmissionPatternSet &patterns);
} // namespace ExportSystemC
} // namespace circt

#endif // SCFEMISSIONPATTERNS_H
