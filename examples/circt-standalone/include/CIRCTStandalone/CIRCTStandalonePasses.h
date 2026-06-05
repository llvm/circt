//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STANDALONE_CIRCTSTANDALONEPASSES_H
#define CIRCT_STANDALONE_CIRCTSTANDALONEPASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {
namespace standalone {

#define GEN_PASS_DECL
#include "CIRCTStandalone/CIRCTStandalonePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "CIRCTStandalone/CIRCTStandalonePasses.h.inc"

} // namespace standalone
} // namespace circt

#endif // CIRCT_STANDALONE_CIRCTSTANDALONEPASSES_H
