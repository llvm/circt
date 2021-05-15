//===- HWTypes.cpp - HW types code defs -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation logic for HW data types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::rtl;
using namespace circt::rtl::detail;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/HW/HWTypes.cpp.inc"

FieldInfo FieldInfo::allocateInto(mlir::TypeStorageAllocator &alloc) const {
  return FieldInfo{alloc.copyInto(name), type};
}

//===----------------------------------------------------------------------===//
// Type Helpers
//===----------------------------------------------------------------------===/

/// Return true if the specified type is a value RTL Integer type.  This checks
/// that it is a signless standard dialect type, that it isn't zero bits.
bool circt::rtl::isRTLIntegerType(mlir::Type type) {
  auto intType = type.dyn_cast<IntegerType>();
  if (!intType || !intType.isSignless())
    return false;

  return intType.getWidth() != 0;
}

/// Return true if the specified type can be used as an RTL value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType.
bool circt::rtl::isRTLValueType(Type type) {
  // Signless and signed integer types are both valid.
  if (type.isa<IntegerType>())
    return true;

  if (auto array = type.dyn_cast<ArrayType>())
    return isRTLValueType(array.getElementType());

  if (auto array = type.dyn_cast<UnpackedArrayType>())
    return isRTLValueType(array.getElementType());

  if (auto t = type.dyn_cast<StructType>()) {
    return std::all_of(t.getElements().begin(), t.getElements().end(),
                       [](const auto &f) { return isRTLValueType(f.type); });
  }

  if (auto t = type.dyn_cast<UnionType>()) {
    return std::all_of(t.getElements().begin(), t.getElements().end(),
                       [](const auto &f) { return isRTLValueType(f.type); });
  }

  return false;
}

/// Return the hardware bit width of a type. Does not reflect any encoding,
/// padding, or storage scheme, just the bit (and wire width) of a
/// statically-size type. Reflects the number of wires needed to transmit a
/// value of this type. Returns -1 if the type is not known or cannot be
/// statically computed.
int64_t circt::rtl::getBitWidth(mlir::Type type) {
  return llvm::TypeSwitch<::mlir::Type, size_t>(type)
      .Case<IntegerType>(
          [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
      .Case<ArrayType>([](ArrayType a) {
        int64_t elementBitWidth = getBitWidth(a.getElementType());
        if (elementBitWidth < 0)
          return elementBitWidth;
        return (int64_t)a.getSize() * elementBitWidth;
      })
      .Case<StructType>([](StructType s) {
        int64_t total = 0;
        for (auto field : s.getElements()) {
          int64_t fieldSize = getBitWidth(field.type);
          if (fieldSize < 0)
            return fieldSize;
          total += fieldSize;
        }
        return total;
      })
      .Case<UnionType>([](UnionType u) {
        int64_t maxSize = 0;
        for (auto field : u.getElements()) {
          int64_t fieldSize = getBitWidth(field.type);
          if (fieldSize > maxSize)
            maxSize = fieldSize;
        }
        return maxSize;
      })
      .Default([](Type) { return -1; });
}

/// Return true if the specified type contains known marker types like
/// InOutType.  Unlike isRTLValueType, this is not conservative, it only returns
/// false on known InOut types, rather than any unknown types.
bool circt::rtl::hasRTLInOutType(Type type) {
  if (auto array = type.dyn_cast<ArrayType>())
    return hasRTLInOutType(array.getElementType());

  if (auto array = type.dyn_cast<UnpackedArrayType>())
    return hasRTLInOutType(array.getElementType());

  if (auto t = type.dyn_cast<StructType>()) {
    return std::any_of(t.getElements().begin(), t.getElements().end(),
                       [](const auto &f) { return hasRTLInOutType(f.type); });
  }
  return type.isa<InOutType>();
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
      typeString.startswith("uarray<") || typeString.startswith("struct<")) {
    llvm::StringRef mnemonic;
    if (p.parseKeyword(&mnemonic))
      llvm_unreachable("should have an array or inout keyword here");
    auto parseResult =
        generatedTypeParser(p.getBuilder().getContext(), p, mnemonic, result);
    return parseResult.hasValue() ? success() : failure();
  }

  return p.parseType(result);
}

static void printRTLElementType(Type element, DialectAsmPrinter &p) {
  if (succeeded(generatedTypePrinter(element, p)))
    return;
  p.printType(element);
}

//===----------------------------------------------------------------------===//
// Struct Type
//===----------------------------------------------------------------------===//
namespace circt {
namespace rtl {
namespace detail {
bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}
llvm::hash_code hash_value(const FieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type);
}
} // namespace detail
} // namespace rtl
} // namespace circt

/// Parse a list of field names and types within <>. E.g.:
/// <foo: i7, bar: i8>
static ParseResult parseFields(MLIRContext *ctxt, DialectAsmParser &p,
                               SmallVectorImpl<FieldInfo> &parameters) {
  StringRef name;
  if (p.parseLess())
    return failure();
  while (mlir::succeeded(p.parseOptionalKeyword(&name))) {
    Type type;
    if (p.parseColon() || p.parseType(type))
      return failure();
    parameters.push_back(FieldInfo{name, type});
    if (p.parseOptionalComma())
      break;
  }
  if (p.parseGreater())
    return failure();
  return success();
}

/// Print out a list of named fields surrounded by <>.
static void printFields(DialectAsmPrinter &p, ArrayRef<FieldInfo> fields) {
  p << '<';
  llvm::interleaveComma(fields, p, [&](const FieldInfo &field) {
    p << field.name << ": " << field.type;
  });
  p << ">";
}

Type StructType::parse(MLIRContext *ctxt, DialectAsmParser &p) {
  llvm::SmallVector<FieldInfo, 4> parameters;
  if (parseFields(ctxt, p, parameters))
    return Type();
  return get(ctxt, parameters);
}

void StructType::print(DialectAsmPrinter &p) const {
  p << "struct";
  printFields(p, getElements());
}

Type StructType::getFieldType(mlir::StringRef fieldName) {
  for (const auto &field : getElements())
    if (field.name == fieldName)
      return field.type;
  return Type();
}

void StructType::getInnerTypes(SmallVectorImpl<Type> &types) {
  for (const auto &field : getElements())
    types.push_back(field.type);
}

//===----------------------------------------------------------------------===//
// Union Type
//===----------------------------------------------------------------------===//

Type UnionType::parse(MLIRContext *ctxt, DialectAsmParser &p) {
  llvm::SmallVector<FieldInfo, 4> parameters;
  if (parseFields(ctxt, p, parameters))
    return Type();
  return get(ctxt, parameters);
}

void UnionType::print(DialectAsmPrinter &p) const {
  p << "union";
  printFields(p, getElements());
}

Type UnionType::getFieldType(mlir::StringRef fieldName) {
  for (const auto &field : getElements())
    if (field.name == fieldName)
      return field.type;
  return Type();
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
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner,
                    dims[0])))
    return Type();

  return get(ctxt, inner, dims[0]);
}

void ArrayType::print(DialectAsmPrinter &p) const {
  p << "array<" << getSize() << "x";
  printRTLElementType(getElementType(), p);
  p << '>';
}

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type innerType, size_t size) {
  if (hasRTLInOutType(innerType))
    return emitError() << "rtl.array cannot contain InOut types";
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
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner,
                    dims[0])))
    return Type();

  return get(ctxt, inner, dims[0]);
}

void UnpackedArrayType::print(DialectAsmPrinter &p) const {
  p << "uarray<" << getSize() << "x";
  printRTLElementType(getElementType(), p);
  p << '>';
}

LogicalResult
UnpackedArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                          Type innerType, size_t size) {
  if (!isRTLValueType(innerType))
    return emitError() << "invalid element for sv.uarray type";
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
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner)))
    return Type();

  return get(ctxt, inner);
}

void InOutType::print(DialectAsmPrinter &p) const {
  p << "inout<";
  printRTLElementType(getElementType(), p);
  p << '>';
}

LogicalResult InOutType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type innerType) {
  if (!isRTLValueType(innerType))
    return emitError() << "invalid element for rtl.inout type " << innerType;
  return success();
}

/// Parses a type registered to this dialect. Parse out the mnemonic then invoke
/// the tblgen'd type parser dispatcher.
Type RTLDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();
  Type type;
  auto parseResult = generatedTypeParser(getContext(), parser, mnemonic, type);
  if (parseResult.hasValue())
    return type;
  return Type();
}

/// Print a type registered to this dialect. Try the tblgen'd type printer
/// dispatcher then fail since all RTL types are defined via ODS.
void RTLDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unexpected 'rtl' type");
}

void RTLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/HW/HWTypes.cpp.inc"
      >();
}
