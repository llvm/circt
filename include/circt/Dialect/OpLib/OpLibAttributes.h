//===- OpLibAttributes.h - OpLib attribute definitions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpLib dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OPLIB_OPLIBATTRIBUTES_H
#define CIRCT_DIALECT_OPLIB_OPLIBATTRIBUTES_H

#include "circt/Dialect/OpLib/OpLibDialect.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/OpLib/OpLibAttributes.h.inc"

namespace circt {
namespace oplib {

/// Parse an array of attributes while recognizing the properties of the 
/// OpLib dialect even without a `#oplib.` prefix. Any attributes 
/// supplied in \p alreadyParsed are prepended to the parsed ones.
mlir::OptionalParseResult
parseOptionalPropertyArray(ArrayAttr &attr, AsmParser &parser,
                           ArrayRef<Attribute> alreadyParsed = {});

/// Print an array attribute, suppressing the `#oplib.` prefix for properties
/// defined in the OpLib dialect. Attributes mentioned in \p alreadyPrinted
/// are skipped.
void printPropertyArray(ArrayAttr attr, AsmPrinter &p,
                        ArrayRef<Attribute> alreadyPrinted = {});

} // namespace oplib
} // namespace circt

#endif // CIRCT_DIALECT_OPLIB_OPLIBATTRIBUTES_H
