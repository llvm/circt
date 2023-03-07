//===- CustomDirectiveImpl.h - Custom TableGen directives -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides common custom directives for table-gen assembly formats.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_CUSTOMDIRECTIVEIMPL_H
#define CIRCT_SUPPORT_CUSTOMDIRECTIVEIMPL_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

namespace circt {

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

/// Parse an attribute dictionary and ensure that it contains a `name` field by
/// inferring its value from the SSA name of the operation's first result if
/// necessary.
ParseResult parseImplicitSSAName(OpAsmParser &parser, NamedAttrList &attrs);

/// Ensure that `attrs` contains a `name` attribute by inferring its value from
/// the SSA name of the operation's first result if necessary. Returns true if a
/// name was inferred, false if `attrs` already contained a `name`.
bool inferImplicitSSAName(OpAsmParser &parser, NamedAttrList &attrs);

/// Print an attribute dictionary and elide the `name` field if its value
/// matches the SSA name of the operation's first result.
void printImplicitSSAName(OpAsmPrinter &p, Operation *op, DictionaryAttr attrs,
                          ArrayRef<StringRef> extraElides = {});

/// Check if the `name` attribute in `attrs` matches the SSA name of the
/// operation's first result. If it does, add `name` to `elides`. This is
/// helpful during printing of attribute dictionaries in order to determine if
/// the inclusion of the `name` field would be redundant.
void elideImplicitSSAName(OpAsmPrinter &printer, Operation *op,
                          DictionaryAttr attrs,
                          SmallVectorImpl<StringRef> &elides);

} // namespace circt

#endif // CIRCT_SUPPORT_CUSTOMDIRECTIVEIMPL_H
