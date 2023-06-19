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
/// Get the portname from an SSA value string, if said value name is not a
/// number.
StringAttr getPortNameAttr(MLIRContext *context, StringRef name);

/// This is a variant of mlir::parseFunctionSignature that allows names on
/// result arguments.
ParseResult parseModuleFunctionSignature(
    OpAsmParser &parser, bool &isVariadic,
    SmallVectorImpl<OpAsmParser::Argument> &args,
    SmallVectorImpl<Attribute> &argNames, SmallVectorImpl<Attribute> &argLocs,
    SmallVectorImpl<Attribute> &resultNames,
    SmallVectorImpl<DictionaryAttr> &resultAttrs,
    SmallVectorImpl<Attribute> &resultLocs, TypeAttr &type);

/// Print a module signature with named results.
void printModuleSignature(OpAsmPrinter &p, Operation *op,
                          ArrayRef<Type> argTypes, bool isVariadic,
                          ArrayRef<Type> resultTypes, bool &needArgNamesAttr);

// Creates index mappings of a modules ports based on the provided arg
// and result names.
void updateModuleIndexMappings(OperationState &result,
                               ArrayRef<Attribute> argNames,
                               ArrayRef<Attribute> resultNames);

// Refreshes the index mappings of the module's ports based on the current
// arg and result names.
LogicalResult updateModuleIndexMappings(Operation *op);

// Verifies that a modules arg and result index maps are consistent with
// the current arg and result names.
LogicalResult verifyModuleIdxMap(Operation *mod);

// Returns the index of the module's argument with the given name, using
// the module's arg index map.
FailureOr<size_t> getModuleArgIndex(Operation *op, StringRef name);

// Returns the index of the module's result with the given name, using
// the module's result index map.
FailureOr<size_t> getModuleResIndex(Operation *op, StringRef name);

} // namespace module_like_impl
} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_MODULEIMPLEMENTATION_H
