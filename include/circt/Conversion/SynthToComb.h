//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_SYNTHTOCOMB_H
#define CIRCT_CONVERSION_SYNTHTOCOMB_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

#define GEN_PASS_DECL_CONVERTSYNTHTOCOMB
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CIRCT_CONVERSION_SYNTHTOCOMB_H
