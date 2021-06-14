//===- LLHDDialect.cpp - Implement the LLHD dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;
using namespace circt::llhd;

using mlir::TypeStorageAllocator;

//===----------------------------------------------------------------------===//
// LLHDDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with LLHD operations.
struct LLHDInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All operations within LLHD can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *src, bool,
                       BlockAndValueMapping &) const final {
    // Don't inline processes and entities
    return !isa<llhd::ProcOp>(src->getParentOp()) &&
           !isa<llhd::EntityOp>(src->getParentOp());
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LLHD Dialect
//===----------------------------------------------------------------------===//

void LLHDDialect::initialize() {
  addTypes<SigType, TimeType, ArrayType, PtrType>();
  addAttributes<TimeAttr>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LLHD/IR/LLHD.cpp.inc"
      >();
  addInterfaces<LLHDInlinerInterface>();
}

Operation *LLHDDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return builder.create<llhd::ConstOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// Type parsing
//===----------------------------------------------------------------------===//

/// Parse a nested type, enclosed in angle brackts (`<...>`).
static Type parseNestedType(DialectAsmParser &parser) {
  Type underlyingType;
  if (parser.parseLess())
    return Type();

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(underlyingType)) {
    parser.emitError(loc, "No signal type found. Signal needs an underlying "
                          "type.");
    return nullptr;
  }

  if (parser.parseGreater())
    return Type();

  return underlyingType;
}

/// Parse a signal type.
/// Syntax: sig ::= !llhd.sig<type>
static Type parseSigType(DialectAsmParser &parser) {
  return SigType::get(parseNestedType(parser));
}

/// Parse a pointer type.
/// Syntax: ptr ::= !llhd.ptr<type
static Type parsePtrType(DialectAsmParser &parser) {
  return PtrType::get(parseNestedType(parser));
}

/// Parse an array type.
/// Syntax: array ::= !llhd.array<{length}x{elementType}>
static Type parseArrayType(DialectAsmParser &parser) {
  Type elementType;
  SmallVector<int64_t, 1> length;
  if (parser.parseLess() || parser.parseDimensionList(length, false) ||
      parser.parseType(elementType) || parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(),
                     "No array type found. Array needs a length and an element "
                     "type.");
    return nullptr;
  }
  if (length.size() != 1) {
    parser.emitError(parser.getCurrentLocation(),
                     "Array must have exactly one dimension");
    return nullptr;
  }
  return ArrayType::get(length[0], elementType);
}

Type LLHDDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef typeKeyword;
  // parse the type keyword first
  if (parser.parseKeyword(&typeKeyword))
    return Type();
  if (typeKeyword == SigType::getKeyword()) {
    return parseSigType(parser);
  }
  if (typeKeyword == TimeType::getKeyword())
    return TimeType::get(getContext());
  if (typeKeyword == ArrayType::getKeyword())
    return parseArrayType(parser);
  if (typeKeyword == PtrType::getKeyword())
    return parsePtrType(parser);

  emitError(parser.getEncodedSourceLoc(parser.getCurrentLocation()),
            "Invalid LLHD type!");
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Type printing
//===----------------------------------------------------------------------===//

/// Print a signal type with custom syntax:
/// type ::= !sig.type<underlying-type>
static void printSigType(SigType sig, DialectAsmPrinter &printer) {
  printer << sig.getKeyword() << "<" << sig.getUnderlyingType() << ">";
}

void LLHDDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (SigType sig = type.dyn_cast<SigType>()) {
    printSigType(sig, printer);
  } else if (TimeType time = type.dyn_cast<TimeType>()) {
    printer << time.getKeyword();
  } else if (ArrayType array = type.dyn_cast<ArrayType>()) {
    printer << array.getKeyword() << "<" << array.getLength() << "x"
            << array.getElementType() << ">";
  } else if (PtrType ptr = type.dyn_cast<PtrType>()) {
    printer << ptr.getKeyword() << "<" << ptr.getUnderlyingType() << ">";
  } else {
    llvm_unreachable("Unknown LLHD type!");
  }
}

//===----------------------------------------------------------------------===//
// Attribute parsing
//===----------------------------------------------------------------------===//

/// Parse a time attribute with the custom syntax:
/// time ::= #llhd.time<time time_unit, delta d, epsilon e>
static Attribute parseTimeAttribute(DialectAsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();

  // values to parse
  llvm::SmallVector<unsigned, 3> values;
  llvm::StringRef timeUnit;
  unsigned time = 0;
  unsigned delta = 0;
  unsigned eps = 0;

  // parse the time value
  if (parser.parseInteger(time) || parser.parseKeyword(&timeUnit))
    return {};
  values.push_back(time);

  // parse the delta step value
  if (parser.parseComma() || parser.parseInteger(delta) ||
      parser.parseKeyword("d"))
    return {};
  values.push_back(delta);

  // parse the epsilon value
  if (parser.parseComma() || parser.parseInteger(eps) ||
      parser.parseKeyword("e") || parser.parseGreater())
    return Attribute();
  values.push_back(eps);

  // return a new instance of time attribute
  return TimeAttr::get(type, values, timeUnit);
}

/// Parse an LLHD attribute
Attribute LLHDDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  llvm::StringRef attrKeyword;
  // parse keyword first
  if (parser.parseKeyword(&attrKeyword))
    return Attribute();
  if (attrKeyword == TimeAttr::getKeyword()) {
    return parseTimeAttribute(parser, type);
  }

  emitError(parser.getEncodedSourceLoc(parser.getCurrentLocation()),
            "Invalid LLHD attribute!");
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Attribute printing
//===----------------------------------------------------------------------===//

/// Print an LLHD time attribute.
static void printTimeAttribute(TimeAttr attr, DialectAsmPrinter &printer) {
  printer << attr.getKeyword() << "<";
  printer << attr.getTime() << attr.getTimeUnit() << ", ";
  printer << attr.getDelta() << "d, ";
  printer << attr.getEps() << "e>";
}

void LLHDDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  if (TimeAttr time = attr.dyn_cast<TimeAttr>()) {
    printTimeAttribute(time, printer);
  } else {
    llvm_unreachable("Unknown LLHD attribute!");
  }
}

namespace circt {
namespace llhd {
namespace detail {

//===----------------------------------------------------------------------===//
// Type storage
//===----------------------------------------------------------------------===//

// Sig Type Storage

/// Storage struct implementation for LLHD's sig type. The sig type only
/// contains one underlying llhd type.
struct SigTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;
  /// construcor for sig type's storage.
  /// Takes the underlying type as the only argument
  SigTypeStorage(mlir::Type underlyingType) : underlyingType(underlyingType) {}

  /// compare sig type instances on the underlying type
  bool operator==(const KeyTy &key) const { return key == getUnderlyingType(); }

  /// return the KeyTy for sig type
  static KeyTy getKey(mlir::Type underlyingType) {
    return KeyTy(underlyingType);
  }

  /// construction method for creating a new instance of the sig type
  /// storage
  static SigTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<SigTypeStorage>()) SigTypeStorage(key);
  }

  /// get the underlying type
  mlir::Type getUnderlyingType() const { return underlyingType; }

private:
  mlir::Type underlyingType;
};

/// Array Type Storage and Uniquing.
struct ArrayTypeStorage : public TypeStorage {
  /// The hash key used for uniquing.
  using KeyTy = std::pair<unsigned, Type>;
  ArrayTypeStorage(unsigned length, Type elementTy)
      : length(length), elementType(elementTy) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(length, elementType);
  }

  static ArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>())
        ArrayTypeStorage(key.first, key.second);
  }

  unsigned getLength() const { return length; }
  Type getElementType() const { return elementType; }

private:
  unsigned length;
  Type elementType;
};

struct PtrTypeStorage : public mlir::TypeStorage {
  using KeyTy = Type;

  PtrTypeStorage(Type underlyingType) : underlyingType(underlyingType) {}

  bool operator==(const KeyTy &key) const { return key == getUnderlyingType(); }

  static KeyTy getKey(mlir::Type underlyingType) {
    return KeyTy(underlyingType);
  }

  static PtrTypeStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<PtrTypeStorage>()) PtrTypeStorage(key);
  }

  Type getUnderlyingType() const { return underlyingType; }

private:
  Type underlyingType;
};

//===----------------------------------------------------------------------===//
// Attribute storage
//===----------------------------------------------------------------------===//

struct TimeAttrStorage : public mlir::AttributeStorage {
public:
  // use the ArrayRef containign the timing attributes for uniquing
  using KeyTy = std::tuple<Type, llvm::ArrayRef<unsigned>, llvm::StringRef>;

  /// Construct a time attribute storage
  TimeAttrStorage(Type type, llvm::ArrayRef<unsigned> timeValues,
                  llvm::StringRef timeUnit)
      : AttributeStorage(type), timeValues(timeValues), timeUnit(timeUnit) {}

  /// Compare two istances of the time attribute. Hashing and equality are done
  /// only on the time values and time unit. The time type is implicitly always
  /// equal.
  bool operator==(const KeyTy &key) const {
    return (std::get<1>(key) == timeValues && std::get<2>(key) == timeUnit);
  }

  /// Generate hash key for uniquing.
  static unsigned hashKey(const KeyTy &key) {
    auto vals = std::get<1>(key);
    auto unit = std::get<2>(key);
    return llvm::hash_combine(vals, unit);
  }

  /// Construction method for llhd's time attribute
  static TimeAttrStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                    const KeyTy &key) {
    auto keyValues = std::get<1>(key);
    auto values = allocator.copyInto(keyValues);
    auto keyUnit = std::get<2>(key);
    auto unit = allocator.copyInto(keyUnit);

    return new (allocator.allocate<TimeAttrStorage>())
        TimeAttrStorage(std::get<0>(key), values, unit);
  }

  llvm::ArrayRef<unsigned> getValue() const { return timeValues; }

  unsigned getTime() { return timeValues[0]; }

  unsigned getDelta() { return timeValues[1]; }

  unsigned getEps() { return timeValues[2]; }

  llvm::StringRef getTimeUnit() { return timeUnit; }

private:
  llvm::ArrayRef<unsigned> timeValues;
  llvm::StringRef timeUnit;
};

} // namespace detail
} // namespace llhd
} // namespace circt

//===----------------------------------------------------------------------===//
// LLHD Types
//===----------------------------------------------------------------------===//

// Sig Type

SigType SigType::get(mlir::Type underlyingType) {
  return Base::get(underlyingType.getContext(), underlyingType);
}

mlir::Type SigType::getUnderlyingType() {
  return getImpl()->getUnderlyingType();
}

// Time Type

TimeType TimeType::get(MLIRContext *context) { return Base::get(context); }

// ArrayType

ArrayType ArrayType::get(unsigned length, Type elementType) {
  return Base::get(elementType.getContext(), length, elementType);
}

ArrayType ArrayType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                unsigned length, Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), length,
                          elementType);
}

LogicalResult ArrayType::verifyConstructionInvariants(Location loc,
                                                      unsigned length,
                                                      Type elementType) {
  if (length == 0)
    return emitError(loc, "array types must have at least one element");

  return success();
}

unsigned ArrayType::getLength() const { return getImpl()->getLength(); }
Type ArrayType::getElementType() const { return getImpl()->getElementType(); }

// Ptr Type

PtrType PtrType::get(mlir::Type underlyingType) {
  return Base::get(underlyingType.getContext(), underlyingType);
}

mlir::Type PtrType::getUnderlyingType() {
  return getImpl()->getUnderlyingType();
}
//===----------------------------------------------------------------------===//
// LLHD Attribtues
//===----------------------------------------------------------------------===//

// Time Attribute

TimeAttr TimeAttr::get(Type type, llvm::ArrayRef<unsigned> timeValues,
                       llvm::StringRef timeUnit) {
  return Base::get(type.getContext(), type, timeValues, timeUnit);
}

LogicalResult
TimeAttr::verifyConstructionInvariants(Location loc, Type type,
                                       llvm::ArrayRef<unsigned> timeValues,
                                       llvm::StringRef timeUnit) {
  // Check the attribute type is of TimeType.
  if (!type.isa<TimeType>())
    return emitError(loc) << "Time attribute type has to be TimeType, but got "
                          << type;

  // Check the time unit is a legal SI unit
  std::vector<std::string> legalUnits{"ys", "zs", "as", "fs", "ps",
                                      "ns", "us", "ms", "s"};
  if (std::find(legalUnits.begin(), legalUnits.end(), timeUnit) ==
      legalUnits.end())
    return emitError(loc) << "Illegal time unit.";

  // Check there are exactly 3 time values
  if (timeValues.size() != 3)
    return emitError(loc) << "Got a wrong number of time values. Expected "
                             "exactly 3, but got "
                          << timeValues.size();

  return success();
}

llvm::ArrayRef<unsigned> TimeAttr::getValue() const {
  return getImpl()->getValue();
}

unsigned TimeAttr::getTime() const { return getImpl()->getTime(); }

unsigned TimeAttr::getDelta() const { return getImpl()->getDelta(); }

unsigned TimeAttr::getEps() const { return getImpl()->getEps(); }

llvm::StringRef TimeAttr::getTimeUnit() const {
  return getImpl()->getTimeUnit();
}
