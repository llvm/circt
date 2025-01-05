//===- ICE40ToBLIF.h - ICE40 to BLIF dialect conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_ICE40TOBLIF_H
#define CIRCT_CONVERSION_ICE40TOBLIF_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

std::unique_ptr<mlir::Pass> createICE40ToBLIFPass();

#define GEN_PASS_DECL_ICE40TOBLIF
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CIRCT_CONVERSION_ICE40TOBLIF_H
