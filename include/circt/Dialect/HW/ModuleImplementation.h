//===- ModuleImplementation.h - Module-like Op utilities --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility functions for implementing module-like
// operations, in particular, parsing, and printing common to module-like
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_MODULEIMPLEMENTATION_H
#define CIRCT_DIALECT_HW_MODULEIMPLEMENTATION_H

#include "circt/Support/LLVM.h"

#include "mlir/IR/DialectImplementation.h"

namespace circt {
namespace hw {

namespace module_like_impl {

/// This is a variant of mlor::parseFunctionSignature that allows names on
/// result arguments.
ParseResult parseModuleFunctionSignature(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<NamedAttrList> &argAttrs,
    bool &isVariadic, SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<NamedAttrList> &resultAttrs,
    SmallVectorImpl<Attribute> &resultNames);

/// Parse a function result list with named results.
ParseResult parseFunctionResultList(OpAsmParser &parser,
                                    SmallVectorImpl<Type> &resultTypes,
                                    SmallVectorImpl<NamedAttrList> &resultAttrs,
                                    SmallVectorImpl<Attribute> &resultNames);

/// Print a module signature with named results.
void printModuleSignature(OpAsmPrinter &p, Operation *op,
                          ArrayRef<Type> argTypes, bool isVariadic,
                          ArrayRef<Type> resultTypes, bool &needArgNamesAttr);

} // namespace module_like_impl
} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_MODULEIMPLEMENTATION_H
