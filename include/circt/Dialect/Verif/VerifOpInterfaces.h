//===- VerifOpInterfaces.h - TODO ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model operation declarations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VERIF_VERIFOPINTERFACES_H
#define CIRCT_DIALECT_VERIF_VERIFOPINTERFACES_H

#include "circt/Dialect/LTL/LTLTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace circt {
namespace verif {
class RequireLike;
} // namespace verif
} // namesapce circt

#include "circt/Dialect/Verif/VerifOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_VERIF_VERIFOPINTERFACES_H
