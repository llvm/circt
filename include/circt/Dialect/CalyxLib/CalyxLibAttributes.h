//===- CalyxLibAttributes.h - CalyxLib attribute definitions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CalyxLib dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYXLIB_CALYXLIBATTRIBUTES_H
#define CIRCT_DIALECT_CALYXLIB_CALYXLIBATTRIBUTES_H

#include "circt/Dialect/CalyxLib/CalyxLibDialect.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/CalyxLib/CalyxLibAttributes.h.inc"

namespace circt {
namespace calyxlib {

/// Parse an array of attributes while recognizing the properties of the 
/// CalyxLib dialect even without a `#calyxlib.` prefix. Any attributes 
/// supplied in \p alreadyParsed are prepended to the parsed ones.
mlir::OptionalParseResult
parseOptionalPropertyArray(ArrayAttr &attr, AsmParser &parser,
                           ArrayRef<Attribute> alreadyParsed = {});

/// Print an array attribute, suppressing the `#calyxlib.` prefix for properties
/// defined in the CalyxLib dialect. Attributes mentioned in \p alreadyPrinted
/// are skipped.
void printPropertyArray(ArrayAttr attr, AsmPrinter &p,
                        ArrayRef<Attribute> alreadyPrinted = {});

} // namespace calyxlib
} // namespace circt

#endif // CIRCT_DIALECT_CALYXLIB_CALYXLIBATTRIBUTES_H
