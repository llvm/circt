//===- RTLTypes.cpp - RTL types code defs ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation logic for RTL data types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/RTL/RTLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt::rtl;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTL/RTLTypes.cpp.inc"

/// Return true if the specified type can be used as an RTL value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType.
bool circt::rtl::isRTLValueType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.isSignless();

  if (auto array = type.dyn_cast<ArrayType>())
    return isRTLValueType(array.getElementType());

  if (auto array = type.dyn_cast<UnpackedArrayType>())
    return isRTLValueType(array.getElementType());

  return false;
}

/// Return the element type of an InOutType or null if the operand isn't an
/// InOut type.
mlir::Type circt::rtl::getInOutElementType(mlir::Type type) {
  if (auto inout = type.dyn_cast_or_null<InOutType>())
    return inout.getElementType();
  return {};
}

/// Return the element type of an ArrayType or UnpackedArrayType, or null if the
/// operand isn't an array.
Type circt::rtl::getAnyRTLArrayElementType(Type type) {
  if (!type)
    return {};
  if (auto array = type.dyn_cast<ArrayType>())
    return array.getElementType();
  if (auto array = type.dyn_cast<UnpackedArrayType>())
    return array.getElementType();
  return {};
}

/// Parse and print nested RTL types nicely.  These helper methods allow eliding
/// the "rtl." prefix on array, inout, and other types when in a context that
/// expects RTL subelement types.
static ParseResult parseRTLElementType(Type &result, DialectAsmParser &p) {
  // If this is an RTL dialect type, then we don't need/want the !rtl. prefix
  // redundantly specified.
  auto fullString = p.getFullSymbolSpec();
  auto *curPtr = p.getCurrentLocation().getPointer();
  auto typeString =
      StringRef(curPtr, fullString.size() - (curPtr - fullString.data()));

  if (typeString.startswith("array<") || typeString.startswith("inout<") ||
      typeString.startswith("uarray<")) {
    llvm::StringRef mnemonic;
    if (p.parseKeyword(&mnemonic))
      llvm_unreachable("should have an array or inout keyword here");
    result = generatedTypeParser(p.getBuilder().getContext(), p, mnemonic);
    return result ? success() : failure();
  }

  return p.parseType(result);
}

static void printRTLElementType(Type element, DialectAsmPrinter &p) {
  if (succeeded(generatedTypePrinter(element, p)))
    return;
  p.printType(element);
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

Type ArrayType::parse(MLIRContext *ctxt, DialectAsmParser &p) {
  SmallVector<int64_t, 2> dims;
  Type inner;
  if (p.parseLess() || p.parseDimensionList(dims, /* allowDynamic */ false) ||
      parseRTLElementType(inner, p) || p.parseGreater())
    return Type();
  if (dims.size() != 1) {
    p.emitError(p.getNameLoc(), "rtl.array only supports one dimension");
    return Type();
  }

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verifyConstructionInvariants(loc, inner, dims[0])))
    return Type();

  return get(ctxt, inner, dims[0]);
}

void ArrayType::print(DialectAsmPrinter &p) const {
  p << "array<" << getSize() << "x";
  printRTLElementType(getElementType(), p);
  p << '>';
}

LogicalResult ArrayType::verifyConstructionInvariants(Location loc,
                                                      Type innerType,
                                                      size_t size) {
  if (!isRTLValueType(innerType))
    return emitError(loc, "invalid element for rtl.array type");
  return success();
}

//===----------------------------------------------------------------------===//
// UnpackedArrayType
//===----------------------------------------------------------------------===//

Type UnpackedArrayType::parse(MLIRContext *ctxt, DialectAsmParser &p) {
  SmallVector<int64_t, 2> dims;
  Type inner;
  if (p.parseLess() || p.parseDimensionList(dims, /* allowDynamic */ false) ||
      parseRTLElementType(inner, p) || p.parseGreater())
    return Type();

  if (dims.size() != 1) {
    p.emitError(p.getNameLoc(), "sv.uarray only supports one dimension");
    return Type();
  }

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verifyConstructionInvariants(loc, inner, dims[0])))
    return Type();

  return get(ctxt, inner, dims[0]);
}

void UnpackedArrayType::print(DialectAsmPrinter &p) const {
  p << "uarray<" << getSize() << "x";
  printRTLElementType(getElementType(), p);
  p << '>';
}

LogicalResult UnpackedArrayType::verifyConstructionInvariants(Location loc,
                                                              Type innerType,
                                                              size_t size) {
  if (!isRTLValueType(innerType))
    return emitError(loc, "invalid element for sv.uarray type");
  return success();
}

//===----------------------------------------------------------------------===//
// InOutType
//===----------------------------------------------------------------------===//

Type InOutType::parse(MLIRContext *ctxt, DialectAsmParser &p) {
  Type inner;
  if (p.parseLess() || parseRTLElementType(inner, p) || p.parseGreater())
    return Type();

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verifyConstructionInvariants(loc, inner)))
    return Type();

  return get(ctxt, inner);
}

void InOutType::print(DialectAsmPrinter &p) const {
  p << "inout<";
  printRTLElementType(getElementType(), p);
  p << '>';
}

LogicalResult InOutType::verifyConstructionInvariants(Location loc,
                                                      Type innerType) {
  if (!isRTLValueType(innerType))
    return emitError(loc, "invalid element for rtl.inout type");
  return success();
}

/// Parses a type registered to this dialect. Parse out the mnemonic then invoke
/// the tblgen'd type parser dispatcher.
Type RTLDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();
  return generatedTypeParser(getContext(), parser, mnemonic);
}

/// Print a type registered to this dialect. Try the tblgen'd type printer
/// dispatcher then fail since all RTL types are defined via ODS.
void RTLDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unexpected 'rtl' type");
}
