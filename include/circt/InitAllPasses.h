//===- InitAllPasses.h - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes to the
// system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_INITALLPASSES_H_
#define CIRCT_INITALLPASSES_H_

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"

namespace circt {

inline void registerAllPasses() {
  // Conversion Passes
  registerConversionPasses();

  // Standard Passes
  calyx::registerPasses();
  esi::registerESIPasses();
  firrtl::registerPasses();
  llhd::initLLHDTransformationPasses();
  seq::registerSeqPasses();
  sv::registerPasses();

  // Analysis passes
  comb::registerCombPasses();
}

} // namespace circt

#endif // CIRCT_INITALLPASSES_H_
