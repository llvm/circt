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
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::hw;
using namespace circt::hw::detail;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/HW/HWTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type Helpers
//===----------------------------------------------------------------------===/

mlir::Type circt::hw::getCanonicalType(mlir::Type type) {
  Type canonicalType;
  if (auto typeAlias = dyn_cast<TypeAliasType>(type))
    canonicalType = typeAlias.getCanonicalType();
  else
    canonicalType = type;
  return canonicalType;
}

/// Return true if the specified type is a value HW Integer type.  This checks
/// that it is a signless standard dialect type or a hw::IntType.
bool circt::hw::isHWIntegerType(mlir::Type type) {
  Type canonicalType = getCanonicalType(type);

  if (isa<hw::IntType>(canonicalType))
    return true;

  auto intType = dyn_cast<IntegerType>(canonicalType);
  if (!intType || !intType.isSignless())
    return false;

  return true;
}

bool circt::hw::isHWEnumType(mlir::Type type) {
  return isa<hw::EnumType>(getCanonicalType(type));
}

/// Return true if the specified type can be used as an HW value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType.
bool circt::hw::isHWValueType(Type type) {
  // Signless and signed integer types are both valid.
  if (isa<IntegerType, IntType, EnumType>(type))
    return true;

  if (auto array = dyn_cast<ArrayType>(type))
    return isHWValueType(array.getElementType());

  if (auto array = dyn_cast<UnpackedArrayType>(type))
    return isHWValueType(array.getElementType());

  if (auto t = dyn_cast<StructType>(type))
    return llvm::all_of(t.getElements(),
                        [](auto f) { return isHWValueType(f.type); });

  if (auto t = dyn_cast<UnionType>(type))
    return llvm::all_of(t.getElements(),
                        [](auto f) { return isHWValueType(f.type); });

  if (auto t = dyn_cast<TypeAliasType>(type))
    return isHWValueType(t.getCanonicalType());

  return false;
}

/// Return the hardware bit width of a type. Does not reflect any encoding,
/// padding, or storage scheme, just the bit (and wire width) of a
/// statically-size type. Reflects the number of wires needed to transmit a
/// value of this type. Returns -1 if the type is not known or cannot be
/// statically computed.
int64_t circt::hw::getBitWidth(mlir::Type type) {
  return llvm::TypeSwitch<::mlir::Type, size_t>(type)
      .Case<IntegerType>(
          [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
      .Case<ArrayType, UnpackedArrayType>([](auto a) {
        int64_t elementBitWidth = getBitWidth(a.getElementType());
        if (elementBitWidth < 0)
          return elementBitWidth;
        int64_t dimBitWidth = a.getNumElements();
        if (dimBitWidth < 0)
          return static_cast<int64_t>(-1L);
        return (int64_t)a.getNumElements() * elementBitWidth;
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
          int64_t fieldSize = getBitWidth(field.type) + field.offset;
          if (fieldSize > maxSize)
            maxSize = fieldSize;
        }
        return maxSize;
      })
      .Case<EnumType>([](EnumType e) { return e.getBitWidth(); })
      .Case<TypeAliasType>(
          [](TypeAliasType t) { return getBitWidth(t.getCanonicalType()); })
      .Default([](Type) { return -1; });
}

/// Return true if the specified type contains known marker types like
/// InOutType.  Unlike isHWValueType, this is not conservative, it only returns
/// false on known InOut types, rather than any unknown types.
bool circt::hw::hasHWInOutType(Type type) {
  if (auto array = dyn_cast<ArrayType>(type))
    return hasHWInOutType(array.getElementType());

  if (auto array = dyn_cast<UnpackedArrayType>(type))
    return hasHWInOutType(array.getElementType());

  if (auto t = dyn_cast<StructType>(type)) {
    return std::any_of(t.getElements().begin(), t.getElements().end(),
                       [](const auto &f) { return hasHWInOutType(f.type); });
  }

  if (auto t = dyn_cast<TypeAliasType>(type))
    return hasHWInOutType(t.getCanonicalType());

  return isa<InOutType>(type);
}

/// Parse and print nested HW types nicely.  These helper methods allow eliding
/// the "hw." prefix on array, inout, and other types when in a context that
/// expects HW subelement types.
static ParseResult parseHWElementType(Type &result, AsmParser &p) {
  // If this is an HW dialect type, then we don't need/want the !hw. prefix
  // redundantly specified.
  auto fullString = static_cast<DialectAsmParser &>(p).getFullSymbolSpec();
  auto *curPtr = p.getCurrentLocation().getPointer();
  auto typeString =
      StringRef(curPtr, fullString.size() - (curPtr - fullString.data()));

  if (typeString.starts_with("array<") || typeString.starts_with("inout<") ||
      typeString.starts_with("uarray<") || typeString.starts_with("struct<") ||
      typeString.starts_with("typealias<") || typeString.starts_with("int<") ||
      typeString.starts_with("enum<")) {
    llvm::StringRef mnemonic;
    auto parseResult = generatedTypeParser(p, &mnemonic, result);
    return parseResult.has_value() ? success() : failure();
  }

  return p.parseType(result);
}

static void printHWElementType(Type element, AsmPrinter &p) {
  if (succeeded(generatedTypePrinter(element, p)))
    return;
  p.printType(element);
}

//===----------------------------------------------------------------------===//
// Int Type
//===----------------------------------------------------------------------===//

Type IntType::get(mlir::TypedAttr width) {
  // The width expression must always be a 32-bit wide integer type itself.
  auto widthWidth = llvm::dyn_cast<IntegerType>(width.getType());
  assert(widthWidth && widthWidth.getWidth() == 32 &&
         "!hw.int width must be 32-bits");
  (void)widthWidth;

  if (auto cstWidth = llvm::dyn_cast<IntegerAttr>(width))
    return IntegerType::get(width.getContext(),
                            cstWidth.getValue().getZExtValue());

  return Base::get(width.getContext(), width);
}

Type IntType::parse(AsmParser &p) {
  // The bitwidth of the parameter size is always 32 bits.
  auto int32Type = p.getBuilder().getIntegerType(32);

  mlir::TypedAttr width;
  if (p.parseLess() || p.parseAttribute(width, int32Type) || p.parseGreater())
    return Type();
  return get(width);
}

void IntType::print(AsmPrinter &p) const {
  p << "<";
  p.printAttributeWithoutType(getWidth());
  p << '>';
}

//===----------------------------------------------------------------------===//
// Struct Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace hw {
namespace detail {
bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}
llvm::hash_code hash_value(const FieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type);
}
} // namespace detail
} // namespace hw
} // namespace circt

/// Parse a list of unique field names and types within <>. E.g.:
/// <foo: i7, bar: i8>
static ParseResult parseFields(AsmParser &p,
                               SmallVectorImpl<FieldInfo> &parameters) {
  llvm::StringSet<> nameSet;
  bool hasDuplicateName = false;
  auto parseResult = p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
        std::string name;
        Type type;

        auto fieldLoc = p.getCurrentLocation();
        if (p.parseKeywordOrString(&name) || p.parseColon() ||
            p.parseType(type))
          return failure();

        if (!nameSet.insert(name).second) {
          p.emitError(fieldLoc, "duplicate field name \'" + name + "\'");
          // Continue parsing to print all duplicates, but make sure to error
          // eventually
          hasDuplicateName = true;
        }

        parameters.push_back(
            FieldInfo{StringAttr::get(p.getContext(), name), type});
        return success();
      });

  if (hasDuplicateName)
    return failure();
  return parseResult;
}

/// Print out a list of named fields surrounded by <>.
static void printFields(AsmPrinter &p, ArrayRef<FieldInfo> fields) {
  p << '<';
  llvm::interleaveComma(fields, p, [&](const FieldInfo &field) {
    p.printKeywordOrString(field.name.getValue());
    p << ": " << field.type;
  });
  p << ">";
}

Type StructType::parse(AsmParser &p) {
  llvm::SmallVector<FieldInfo, 4> parameters;
  if (parseFields(p, parameters))
    return Type();
  return get(p.getContext(), parameters);
}

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<StructType::FieldInfo> elements) {
  llvm::SmallDenseSet<StringAttr> fieldNameSet;
  LogicalResult result = success();
  fieldNameSet.reserve(elements.size());
  for (const auto &elt : elements)
    if (!fieldNameSet.insert(elt.name).second) {
      result = failure();
      emitError() << "duplicate field name '" << elt.name.getValue()
                  << "' in hw.struct type";
    }
  return result;
}

void StructType::print(AsmPrinter &p) const { printFields(p, getElements()); }

Type StructType::getFieldType(mlir::StringRef fieldName) {
  for (const auto &field : getElements())
    if (field.name == fieldName)
      return field.type;
  return Type();
}

std::optional<uint32_t> StructType::getFieldIndex(mlir::StringRef fieldName) {
  ArrayRef<hw::StructType::FieldInfo> elems = getElements();
  for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
    if (elems[idx].name == fieldName)
      return idx;
  return {};
}

std::optional<uint32_t> StructType::getFieldIndex(mlir::StringAttr fieldName) {
  ArrayRef<hw::StructType::FieldInfo> elems = getElements();
  for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
    if (elems[idx].name == fieldName)
      return idx;
  return {};
}

static std::pair<uint64_t, SmallVector<uint64_t>>
getFieldIDsStruct(const StructType &st) {
  uint64_t fieldID = 0;
  auto elements = st.getElements();
  SmallVector<uint64_t> fieldIDs;
  fieldIDs.reserve(elements.size());
  for (auto &element : elements) {
    auto type = element.type;
    fieldID += 1;
    fieldIDs.push_back(fieldID);
    // Increment the field ID for the next field by the number of subfields.
    fieldID += hw::FieldIdImpl::getMaxFieldID(type);
  }
  return {fieldID, fieldIDs};
}

void StructType::getInnerTypes(SmallVectorImpl<Type> &types) {
  for (const auto &field : getElements())
    types.push_back(field.type);
}

uint64_t StructType::getMaxFieldID() const {
  uint64_t fieldID = 0;
  for (const auto &field : getElements())
    fieldID += 1 + hw::FieldIdImpl::getMaxFieldID(field.type);
  return fieldID;
}

std::pair<Type, uint64_t>
StructType::getSubTypeByFieldID(uint64_t fieldID) const {
  if (fieldID == 0)
    return {*this, 0};
  auto [maxId, fieldIDs] = getFieldIDsStruct(*this);
  auto *it = std::prev(llvm::upper_bound(fieldIDs, fieldID));
  auto subfieldIndex = std::distance(fieldIDs.begin(), it);
  auto subfieldType = getElements()[subfieldIndex].type;
  auto subfieldID = fieldID - fieldIDs[subfieldIndex];
  return {subfieldType, subfieldID};
}

std::pair<uint64_t, bool>
StructType::projectToChildFieldID(uint64_t fieldID, uint64_t index) const {
  auto [maxId, fieldIDs] = getFieldIDsStruct(*this);
  auto childRoot = fieldIDs[index];
  auto rangeEnd =
      index + 1 >= getElements().size() ? maxId : (fieldIDs[index + 1] - 1);
  return std::make_pair(fieldID - childRoot,
                        fieldID >= childRoot && fieldID <= rangeEnd);
}

uint64_t StructType::getFieldID(uint64_t index) const {
  auto [maxId, fieldIDs] = getFieldIDsStruct(*this);
  return fieldIDs[index];
}

uint64_t StructType::getIndexForFieldID(uint64_t fieldID) const {
  assert(!getElements().empty() && "Bundle must have >0 fields");
  auto [maxId, fieldIDs] = getFieldIDsStruct(*this);
  auto *it = std::prev(llvm::upper_bound(fieldIDs, fieldID));
  return std::distance(fieldIDs.begin(), it);
}

std::pair<uint64_t, uint64_t>
StructType::getIndexAndSubfieldID(uint64_t fieldID) const {
  auto index = getIndexForFieldID(fieldID);
  auto elementFieldID = getFieldID(index);
  return {index, fieldID - elementFieldID};
}

//===----------------------------------------------------------------------===//
// Union Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace hw {
namespace detail {
bool operator==(const OffsetFieldInfo &a, const OffsetFieldInfo &b) {
  return a.name == b.name && a.type == b.type && a.offset == b.offset;
}
// NOLINTNEXTLINE
llvm::hash_code hash_value(const OffsetFieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type, fi.offset);
}
} // namespace detail
} // namespace hw
} // namespace circt

Type UnionType::parse(AsmParser &p) {
  llvm::SmallVector<FieldInfo, 4> parameters;
  llvm::StringSet<> nameSet;
  bool hasDuplicateName = false;
  if (p.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            StringRef name;
            Type type;

            auto fieldLoc = p.getCurrentLocation();
            if (p.parseKeyword(&name) || p.parseColon() || p.parseType(type))
              return failure();

            if (!nameSet.insert(name).second) {
              p.emitError(fieldLoc, "duplicate field name \'" + name +
                                        "\' in hw.union type");
              // Continue parsing to print all duplicates, but make sure to
              // error eventually
              hasDuplicateName = true;
            }

            size_t offset = 0;
            if (succeeded(p.parseOptionalKeyword("offset")))
              if (p.parseInteger(offset))
                return failure();
            parameters.push_back(UnionType::FieldInfo{
                StringAttr::get(p.getContext(), name), type, offset});
            return success();
          }))
    return Type();

  if (hasDuplicateName)
    return Type();

  return get(p.getContext(), parameters);
}

void UnionType::print(AsmPrinter &odsPrinter) const {
  odsPrinter << '<';
  llvm::interleaveComma(
      getElements(), odsPrinter, [&](const UnionType::FieldInfo &field) {
        odsPrinter << field.name.getValue() << ": " << field.type;
        if (field.offset)
          odsPrinter << " offset " << field.offset;
      });
  odsPrinter << ">";
}

LogicalResult UnionType::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<UnionType::FieldInfo> elements) {
  llvm::SmallDenseSet<StringAttr> fieldNameSet;
  LogicalResult result = success();
  fieldNameSet.reserve(elements.size());
  for (const auto &elt : elements)
    if (!fieldNameSet.insert(elt.name).second) {
      result = failure();
      emitError() << "duplicate field name '" << elt.name.getValue()
                  << "' in hw.union type";
    }
  return result;
}

std::optional<uint32_t> UnionType::getFieldIndex(mlir::StringAttr fieldName) {
  ArrayRef<hw::UnionType::FieldInfo> elems = getElements();
  for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
    if (elems[idx].name == fieldName)
      return idx;
  return {};
}

std::optional<uint32_t> UnionType::getFieldIndex(mlir::StringRef fieldName) {
  return getFieldIndex(StringAttr::get(getContext(), fieldName));
}

UnionType::FieldInfo UnionType::getFieldInfo(::mlir::StringRef fieldName) {
  if (auto fieldIndex = getFieldIndex(fieldName))
    return getElements()[*fieldIndex];
  return FieldInfo();
}

Type UnionType::getFieldType(mlir::StringRef fieldName) {
  return getFieldInfo(fieldName).type;
}

//===----------------------------------------------------------------------===//
// Enum Type
//===----------------------------------------------------------------------===//

Type EnumType::parse(AsmParser &p) {
  llvm::SmallVector<Attribute> fields;

  if (p.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&]() {
        StringRef name;
        if (p.parseKeyword(&name))
          return failure();
        fields.push_back(StringAttr::get(p.getContext(), name));
        return success();
      }))
    return Type();

  return get(p.getContext(), ArrayAttr::get(p.getContext(), fields));
}

void EnumType::print(AsmPrinter &p) const {
  p << '<';
  llvm::interleaveComma(getFields(), p, [&](Attribute enumerator) {
    p << llvm::cast<StringAttr>(enumerator).getValue();
  });
  p << ">";
}

bool EnumType::contains(mlir::StringRef field) {
  return indexOf(field).has_value();
}

std::optional<size_t> EnumType::indexOf(mlir::StringRef field) {
  for (auto it : llvm::enumerate(getFields()))
    if (llvm::cast<StringAttr>(it.value()).getValue() == field)
      return it.index();
  return {};
}

size_t EnumType::getBitWidth() {
  auto w = getFields().size();
  if (w > 1)
    return llvm::Log2_64_Ceil(getFields().size());
  return 1;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

static LogicalResult parseArray(AsmParser &p, Attribute &dim, Type &inner) {
  if (p.parseLess())
    return failure();

  uint64_t dimLiteral;
  auto int64Type = p.getBuilder().getIntegerType(64);

  if (auto res = p.parseOptionalInteger(dimLiteral); res.has_value())
    dim = p.getBuilder().getI64IntegerAttr(dimLiteral);
  else if (!p.parseOptionalAttribute(dim, int64Type).has_value())
    return failure();

  if (!isa<IntegerAttr, ParamExprAttr, ParamDeclRefAttr>(dim)) {
    p.emitError(p.getNameLoc(), "unsupported dimension kind in hw.array");
    return failure();
  }

  if (p.parseXInDimensionList() || parseHWElementType(inner, p) ||
      p.parseGreater())
    return failure();

  return success();
}

Type ArrayType::parse(AsmParser &p) {
  Attribute dim;
  Type inner;

  if (failed(parseArray(p, dim, inner)))
    return Type();

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner, dim)))
    return Type();

  return get(inner.getContext(), inner, dim);
}

void ArrayType::print(AsmPrinter &p) const {
  p << "<";
  p.printAttributeWithoutType(getSizeAttr());
  p << "x";
  printHWElementType(getElementType(), p);
  p << '>';
}

size_t ArrayType::getNumElements() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSizeAttr()))
    return intAttr.getInt();
  return -1;
}

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type innerType, Attribute size) {
  if (hasHWInOutType(innerType))
    return emitError() << "hw.array cannot contain InOut types";
  return success();
}

uint64_t ArrayType::getMaxFieldID() const {
  return getNumElements() *
         (hw::FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

std::pair<Type, uint64_t>
ArrayType::getSubTypeByFieldID(uint64_t fieldID) const {
  if (fieldID == 0)
    return {*this, 0};
  return {getElementType(), getIndexAndSubfieldID(fieldID).second};
}

std::pair<uint64_t, bool>
ArrayType::projectToChildFieldID(uint64_t fieldID, uint64_t index) const {
  auto childRoot = getFieldID(index);
  auto rangeEnd =
      index >= getNumElements() ? getMaxFieldID() : (getFieldID(index + 1) - 1);
  return std::make_pair(fieldID - childRoot,
                        fieldID >= childRoot && fieldID <= rangeEnd);
}

uint64_t ArrayType::getIndexForFieldID(uint64_t fieldID) const {
  assert(fieldID && "fieldID must be at least 1");
  // Divide the field ID by the number of fieldID's per element.
  return (fieldID - 1) / (hw::FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

std::pair<uint64_t, uint64_t>
ArrayType::getIndexAndSubfieldID(uint64_t fieldID) const {
  auto index = getIndexForFieldID(fieldID);
  auto elementFieldID = getFieldID(index);
  return {index, fieldID - elementFieldID};
}

uint64_t ArrayType::getFieldID(uint64_t index) const {
  return 1 + index * (hw::FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

//===----------------------------------------------------------------------===//
// UnpackedArrayType
//===----------------------------------------------------------------------===//

Type UnpackedArrayType::parse(AsmParser &p) {
  Attribute dim;
  Type inner;

  if (failed(parseArray(p, dim, inner)))
    return Type();

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner, dim)))
    return Type();

  return get(inner.getContext(), inner, dim);
}

void UnpackedArrayType::print(AsmPrinter &p) const {
  p << "<";
  p.printAttributeWithoutType(getSizeAttr());
  p << "x";
  printHWElementType(getElementType(), p);
  p << '>';
}

LogicalResult
UnpackedArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                          Type innerType, Attribute size) {
  if (!isHWValueType(innerType))
    return emitError() << "invalid element for uarray type";
  return success();
}

size_t UnpackedArrayType::getNumElements() const {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(getSizeAttr()))
    return intAttr.getInt();
  return -1;
}

uint64_t UnpackedArrayType::getMaxFieldID() const {
  return getNumElements() *
         (hw::FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

std::pair<Type, uint64_t>
UnpackedArrayType::getSubTypeByFieldID(uint64_t fieldID) const {
  if (fieldID == 0)
    return {*this, 0};
  return {getElementType(), getIndexAndSubfieldID(fieldID).second};
}

std::pair<uint64_t, bool>
UnpackedArrayType::projectToChildFieldID(uint64_t fieldID,
                                         uint64_t index) const {
  auto childRoot = getFieldID(index);
  auto rangeEnd =
      index >= getNumElements() ? getMaxFieldID() : (getFieldID(index + 1) - 1);
  return std::make_pair(fieldID - childRoot,
                        fieldID >= childRoot && fieldID <= rangeEnd);
}

uint64_t UnpackedArrayType::getIndexForFieldID(uint64_t fieldID) const {
  assert(fieldID && "fieldID must be at least 1");
  // Divide the field ID by the number of fieldID's per element.
  return (fieldID - 1) / (hw::FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

std::pair<uint64_t, uint64_t>
UnpackedArrayType::getIndexAndSubfieldID(uint64_t fieldID) const {
  auto index = getIndexForFieldID(fieldID);
  auto elementFieldID = getFieldID(index);
  return {index, fieldID - elementFieldID};
}

uint64_t UnpackedArrayType::getFieldID(uint64_t index) const {
  return 1 + index * (hw::FieldIdImpl::getMaxFieldID(getElementType()) + 1);
}

//===----------------------------------------------------------------------===//
// InOutType
//===----------------------------------------------------------------------===//

Type InOutType::parse(AsmParser &p) {
  Type inner;
  if (p.parseLess() || parseHWElementType(inner, p) || p.parseGreater())
    return Type();

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner)))
    return Type();

  return get(p.getContext(), inner);
}

void InOutType::print(AsmPrinter &p) const {
  p << "<";
  printHWElementType(getElementType(), p);
  p << '>';
}

LogicalResult InOutType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type innerType) {
  if (!isHWValueType(innerType))
    return emitError() << "invalid element for hw.inout type " << innerType;
  return success();
}

//===----------------------------------------------------------------------===//
// TypeAliasType
//===----------------------------------------------------------------------===//

static Type computeCanonicalType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case([](TypeAliasType t) {
        return computeCanonicalType(t.getCanonicalType());
      })
      .Case([](ArrayType t) {
        return ArrayType::get(computeCanonicalType(t.getElementType()),
                              t.getNumElements());
      })
      .Case([](UnpackedArrayType t) {
        return UnpackedArrayType::get(computeCanonicalType(t.getElementType()),
                                      t.getNumElements());
      })
      .Case([](StructType t) {
        SmallVector<StructType::FieldInfo> fieldInfo;
        for (auto field : t.getElements())
          fieldInfo.push_back(StructType::FieldInfo{
              field.name, computeCanonicalType(field.type)});
        return StructType::get(t.getContext(), fieldInfo);
      })
      .Default([](Type t) { return t; });
}

TypeAliasType TypeAliasType::get(SymbolRefAttr ref, Type innerType) {
  return get(ref.getContext(), ref, innerType, computeCanonicalType(innerType));
}

Type TypeAliasType::parse(AsmParser &p) {
  SymbolRefAttr ref;
  Type type;
  if (p.parseLess() || p.parseAttribute(ref) || p.parseComma() ||
      p.parseType(type) || p.parseGreater())
    return Type();

  return get(ref, type);
}

void TypeAliasType::print(AsmPrinter &p) const {
  p << "<" << getRef() << ", " << getInnerType() << ">";
}

/// Return the Typedecl referenced by this TypeAlias, given the module to look
/// in.  This returns null when the IR is malformed.
TypedeclOp TypeAliasType::getTypeDecl(const HWSymbolCache &cache) {
  SymbolRefAttr ref = getRef();
  auto typeScope = ::dyn_cast_or_null<TypeScopeOp>(
      cache.getDefinition(ref.getRootReference()));
  if (!typeScope)
    return {};

  return typeScope.lookupSymbol<TypedeclOp>(ref.getLeafReference());
}

////////////////////////////////////////////////////////////////////////////////
// ModuleType
////////////////////////////////////////////////////////////////////////////////

LogicalResult ModuleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<ModulePort> ports) {
  if (llvm::any_of(ports, [](const ModulePort &port) {
        return hasHWInOutType(port.type);
      }))
    return emitError() << "Ports cannot be inout types";
  return success();
}

size_t ModuleType::getPortIdForInputId(size_t idx) {
  assert(idx < getImpl()->inputToAbs.size() && "input port out of range");
  return getImpl()->inputToAbs[idx];
}

size_t ModuleType::getPortIdForOutputId(size_t idx) {
  assert(idx < getImpl()->outputToAbs.size() && " output port out of range");
  return getImpl()->outputToAbs[idx];
}

size_t ModuleType::getInputIdForPortId(size_t idx) {
  auto nIdx = getImpl()->absToInput[idx];
  assert(nIdx != ~0ULL);
  return nIdx;
}

size_t ModuleType::getOutputIdForPortId(size_t idx) {
  auto nIdx = getImpl()->absToOutput[idx];
  assert(nIdx != ~0ULL);
  return nIdx;
}

size_t ModuleType::getNumInputs() { return getImpl()->inputToAbs.size(); }

size_t ModuleType::getNumOutputs() { return getImpl()->outputToAbs.size(); }

size_t ModuleType::getNumPorts() { return getPorts().size(); }

SmallVector<Type> ModuleType::getInputTypes() {
  SmallVector<Type> retval;
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Input)
      retval.push_back(p.type);
    else if (p.dir == ModulePort::Direction::InOut) {
      retval.push_back(hw::InOutType::get(p.type));
    }
  }
  return retval;
}

SmallVector<Type> ModuleType::getOutputTypes() {
  SmallVector<Type> retval;
  for (auto &p : getPorts())
    if (p.dir == ModulePort::Direction::Output)
      retval.push_back(p.type);
  return retval;
}

SmallVector<Type> ModuleType::getPortTypes() {
  SmallVector<Type> retval;
  for (auto &p : getPorts())
    retval.push_back(p.type);
  return retval;
}

Type ModuleType::getInputType(size_t idx) {
  const auto &portInfo = getPorts()[getPortIdForInputId(idx)];
  if (portInfo.dir != ModulePort::InOut)
    return portInfo.type;
  return InOutType::get(portInfo.type);
}

Type ModuleType::getOutputType(size_t idx) {
  return getPorts()[getPortIdForOutputId(idx)].type;
}

SmallVector<Attribute> ModuleType::getInputNames() {
  SmallVector<Attribute> retval;
  for (auto &p : getPorts())
    if (p.dir != ModulePort::Direction::Output)
      retval.push_back(p.name);
  return retval;
}

SmallVector<Attribute> ModuleType::getOutputNames() {
  SmallVector<Attribute> retval;
  for (auto &p : getPorts())
    if (p.dir == ModulePort::Direction::Output)
      retval.push_back(p.name);
  return retval;
}

StringAttr ModuleType::getPortNameAttr(size_t idx) {
  return getPorts()[idx].name;
}

StringRef ModuleType::getPortName(size_t idx) {
  auto sa = getPortNameAttr(idx);
  if (sa)
    return sa.getValue();
  return {};
}

StringAttr ModuleType::getInputNameAttr(size_t idx) {
  return getPorts()[getPortIdForInputId(idx)].name;
}

StringRef ModuleType::getInputName(size_t idx) {
  auto sa = getInputNameAttr(idx);
  if (sa)
    return sa.getValue();
  return {};
}

StringAttr ModuleType::getOutputNameAttr(size_t idx) {
  return getPorts()[getPortIdForOutputId(idx)].name;
}

StringRef ModuleType::getOutputName(size_t idx) {
  auto sa = getOutputNameAttr(idx);
  if (sa)
    return sa.getValue();
  return {};
}

bool ModuleType::isOutput(size_t idx) {
  auto &p = getPorts()[idx];
  return p.dir == ModulePort::Direction::Output;
}

FunctionType ModuleType::getFuncType() {
  SmallVector<Type> inputs, outputs;
  for (auto p : getPorts())
    if (p.dir == ModulePort::Input)
      inputs.push_back(p.type);
    else if (p.dir == ModulePort::InOut)
      inputs.push_back(InOutType::get(p.type));
    else
      outputs.push_back(p.type);
  return FunctionType::get(getContext(), inputs, outputs);
}

ArrayRef<ModulePort> ModuleType::getPorts() const {
  return getImpl()->getPorts();
}

FailureOr<ModuleType> ModuleType::resolveParametricTypes(ArrayAttr parameters,
                                                         LocationAttr loc,
                                                         bool emitErrors) {
  SmallVector<ModulePort, 8> resolvedPorts;
  for (ModulePort port : getPorts()) {
    FailureOr<Type> resolvedType =
        evaluateParametricType(loc, parameters, port.type, emitErrors);
    if (failed(resolvedType))
      return failure();
    port.type = *resolvedType;
    resolvedPorts.push_back(port);
  }
  return ModuleType::get(getContext(), resolvedPorts);
}

static StringRef dirToStr(ModulePort::Direction dir) {
  switch (dir) {
  case ModulePort::Direction::Input:
    return "input";
  case ModulePort::Direction::Output:
    return "output";
  case ModulePort::Direction::InOut:
    return "inout";
  }
}

static ModulePort::Direction strToDir(StringRef str) {
  if (str == "input")
    return ModulePort::Direction::Input;
  if (str == "output")
    return ModulePort::Direction::Output;
  if (str == "inout")
    return ModulePort::Direction::InOut;
  llvm::report_fatal_error("invalid direction");
}

/// Parse a list of field names and types within <>. E.g.:
/// <input foo: i7, output bar: i8>
static ParseResult parsePorts(AsmParser &p,
                              SmallVectorImpl<ModulePort> &ports) {
  return p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
        StringRef dir;
        std::string name;
        Type type;
        if (p.parseKeyword(&dir) || p.parseKeywordOrString(&name) ||
            p.parseColon() || p.parseType(type))
          return failure();
        ports.push_back(
            {StringAttr::get(p.getContext(), name), type, strToDir(dir)});
        return success();
      });
}

/// Print out a list of named fields surrounded by <>.
static void printPorts(AsmPrinter &p, ArrayRef<ModulePort> ports) {
  p << '<';
  llvm::interleaveComma(ports, p, [&](const ModulePort &port) {
    p << dirToStr(port.dir) << " ";
    p.printKeywordOrString(port.name.getValue());
    p << " : " << port.type;
  });
  p << ">";
}

Type ModuleType::parse(AsmParser &odsParser) {
  llvm::SmallVector<ModulePort, 4> ports;
  if (parsePorts(odsParser, ports))
    return Type();
  return get(odsParser.getContext(), ports);
}

void ModuleType::print(AsmPrinter &odsPrinter) const {
  printPorts(odsPrinter, getPorts());
}

ModuleType circt::hw::detail::fnToMod(Operation *op,
                                      ArrayRef<Attribute> inputNames,
                                      ArrayRef<Attribute> outputNames) {
  return fnToMod(
      cast<FunctionType>(cast<mlir::FunctionOpInterface>(op).getFunctionType()),
      inputNames, outputNames);
}

ModuleType circt::hw::detail::fnToMod(FunctionType fnty,
                                      ArrayRef<Attribute> inputNames,
                                      ArrayRef<Attribute> outputNames) {
  SmallVector<ModulePort> ports;
  if (!inputNames.empty()) {
    for (auto [t, n] : llvm::zip_equal(fnty.getInputs(), inputNames))
      if (auto iot = dyn_cast<hw::InOutType>(t))
        ports.push_back({cast<StringAttr>(n), iot.getElementType(),
                         ModulePort::Direction::InOut});
      else
        ports.push_back({cast<StringAttr>(n), t, ModulePort::Direction::Input});
  } else {
    for (auto t : fnty.getInputs())
      if (auto iot = dyn_cast<hw::InOutType>(t))
        ports.push_back(
            {{}, iot.getElementType(), ModulePort::Direction::InOut});
      else
        ports.push_back({{}, t, ModulePort::Direction::Input});
  }
  if (!outputNames.empty()) {
    for (auto [t, n] : llvm::zip_equal(fnty.getResults(), outputNames))
      ports.push_back({cast<StringAttr>(n), t, ModulePort::Direction::Output});
  } else {
    for (auto t : fnty.getResults())
      ports.push_back({{}, t, ModulePort::Direction::Output});
  }
  return ModuleType::get(fnty.getContext(), ports);
}

detail::ModuleTypeStorage::ModuleTypeStorage(ArrayRef<ModulePort> inPorts)
    : ports(inPorts) {
  size_t nextInput = 0;
  size_t nextOutput = 0;
  for (auto [idx, p] : llvm::enumerate(ports)) {
    if (p.dir == ModulePort::Direction::Output) {
      outputToAbs.push_back(idx);
      absToOutput.push_back(nextOutput);
      absToInput.push_back(~0ULL);
      ++nextOutput;
    } else {
      inputToAbs.push_back(idx);
      absToInput.push_back(nextInput);
      absToOutput.push_back(~0ULL);
      ++nextInput;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// BoilerPlate
////////////////////////////////////////////////////////////////////////////////

void HWDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/HW/HWTypes.cpp.inc"
      >();
}
