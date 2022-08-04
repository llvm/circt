//===- HWEmissionPatterns.h - HW Dialect Emission Patterns ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This exposes the emission patterns of the HW dialect for registration.
//
//===----------------------------------------------------------------------===//

#ifndef HWEMISSIONPATTERNS_H
#define HWEMISSIONPATTERNS_H

#include "../EmissionPattern.h"

namespace circt {
namespace ExportSystemC {
void populateHWEmitters(EmissionPatternSet &patterns);
} // namespace ExportSystemC
} // namespace circt

#endif // HWEMISSIONPATTERNS_H
