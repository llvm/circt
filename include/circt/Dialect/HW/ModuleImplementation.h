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

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"

namespace circt {
namespace hw {

class HWModuleLike;

namespace module_like_impl {

struct PortParse : OpAsmParser::Argument {
  ModulePort::Direction direction;
  std::string rawName;
};

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

/// New Style parsing
ParseResult parseModuleSignature(OpAsmParser &parser,
                                 SmallVectorImpl<PortParse> &args,
                                 TypeAttr &modType);

void printModuleSignatureNew(OpAsmPrinter &p, Region &body,
                             hw::ModuleType modType,
                             ArrayRef<Attribute> portAttrs,
                             ArrayRef<Location> locAttrs);
void printModuleSignatureNew(OpAsmPrinter &p, HWModuleLike op);
void getAsmBlockArgumentNamesImpl(mlir::Region &region,
                                  OpAsmSetValueNameFn setNameFn);

SmallVector<Location> getAllPortLocsImpl(hw::ModuleType modType);

} // namespace module_like_impl
} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_MODULEIMPLEMENTATION_H
