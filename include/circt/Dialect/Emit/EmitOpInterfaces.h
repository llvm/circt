//===- EmitOpInterfaces.h - Declare Emit op interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the Emit dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_EMIT_EMITOPINTERFACES_H
#define CIRCT_DIALECT_EMIT_EMITOPINTERFACES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"

namespace circt {
namespace emit {

template <typename ConcreteType>
class Emittable : public OpTrait::TraitBase<ConcreteType, Emittable> {};

} // namespace emit
} // namespace circt

#include "circt/Dialect/Emit/EmitOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_EMIT_EMITOPINTERFACES_H
