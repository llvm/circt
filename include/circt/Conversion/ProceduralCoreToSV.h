//===- ProceduralCoreToSV.h - Porcedural Core to SV pass entry point ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the declarations for the
// procedural core to SV lowering pass.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PROCEDURALCORETOSV_H
#define CIRCT_CONVERSION_PROCEDURALCORETOSV_H

#include "circt/Support/LLVM.h"
#include <memory>
namespace circt {

#define GEN_PASS_DECL_PROCEDURALCORETOSV
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CIRCT_CONVERSION_PROCEDURALCORETOSV_H
