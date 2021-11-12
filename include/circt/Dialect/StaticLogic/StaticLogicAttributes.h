//===- StaticLogicAttributes.h - StaticLogic attributes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STATICLOGICATTRIBUTES_H
#define CIRCT_STATICLOGICATTRIBUTES_H

#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/StaticLogic/StaticLogicAttributes.h.inc"

#endif // CIRCT_STATICLOGICATTRIBUTES_H
