//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_DATAPATHPASSES_H
#define CIRCT_DIALECT_DATAPATH_DATAPATHPASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
namespace datapath {

/// Generate the code for registering passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Datapath/DatapathPasses.h.inc"

} // namespace datapath
} // namespace circt

#endif // CIRCT_DIALECT_DATAPATH_DATAPATHPASSES_H
