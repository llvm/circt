//===- RTGAttributes.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace llvm {
template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const DenseSet<T> &set) {
  // TODO: improve collision resistance
  unsigned hash = 0;
  for (auto element : set)
    hash ^= element;
  return hash;
}
} // namespace llvm

//===----------------------------------------------------------------------===//
// SetAttr
//===----------------------------------------------------------------------===//

LogicalResult
SetAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                rtg::SetType type, const DenseSet<TypedAttr> *elements) {

  // check that all elements have the right type
  // iterating over the set is fine here because the iteration order is not
  // visible to the outside (it would not be fine to print the earliest invalid
  // element)
  if (!llvm::all_of(*elements, [&](auto element) {
        return element.getType() == type.getElementType();
      })) {
    return emitError() << "all elements must be of the set element type "
                       << type.getElementType();
  }

  return success();
}

Attribute SetAttr::parse(AsmParser &odsParser, Type odsType) {
  DenseSet<TypedAttr> elements;
  Type elementType;
  if (odsParser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::LessGreater,
                                        [&]() {
                                          TypedAttr element;
                                          if (odsParser.parseAttribute(element))
                                            return failure();
                                          elements.insert(element);
                                          elementType = element.getType();
                                          return success();
                                        }))
    return {};

  auto setType = llvm::dyn_cast_or_null<SetType>(odsType);
  if (odsType && !setType) {
    odsParser.emitError(odsParser.getNameLoc())
        << "type must be a an '!rtg.set' type";
    return {};
  }

  if (!setType && elements.empty()) {
    odsParser.emitError(odsParser.getNameLoc())
        << "type must be explicitly provided: cannot infer set element type "
           "from empty set";
    return {};
  }

  if (!setType && !elements.empty())
    setType = SetType::get(elementType);

  return SetAttr::getChecked(
      odsParser.getEncodedSourceLoc(odsParser.getNameLoc()),
      odsParser.getContext(), setType, &elements);
}

void SetAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  // Sort elements lexicographically by their printed string representation
  SmallVector<std::string> sortedElements;
  for (auto element : *getElements()) {
    std::string &elementStr = sortedElements.emplace_back();
    llvm::raw_string_ostream elementOS(elementStr);
    element.print(elementOS);
  }
  llvm::sort(sortedElements);
  llvm::interleaveComma(sortedElements, odsPrinter);
  odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
// TupleAttr
//===----------------------------------------------------------------------===//

Type TupleAttr::getType() const {
  SmallVector<Type> elementTypes(llvm::map_range(
      getElements(), [](auto element) { return element.getType(); }));
  return TupleType::get(getContext(), elementTypes);
}

//===----------------------------------------------------------------------===//
// ImmediateAttr
//===----------------------------------------------------------------------===//

namespace circt {
namespace rtg {
namespace detail {
struct ImmediateAttrStorage : public mlir::AttributeStorage {
  using KeyTy = APInt;
  ImmediateAttrStorage(APInt value) : value(std::move(value)) {}

  KeyTy getAsKey() const { return value; }

  // NOTE: the implementation of this operator is the reason we need to define
  // the storage manually. The auto-generated version would just do the direct
  // equality check of the APInt, but that asserts the bitwidth of both to be
  // the same, leading to a crash. This implementation, therefore, checks for
  // matching bit-width beforehand.
  bool operator==(const KeyTy &key) const {
    return (value.getBitWidth() == key.getBitWidth() && value == key);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static ImmediateAttrStorage *
  construct(mlir::AttributeStorageAllocator &allocator, KeyTy &&key) {
    return new (allocator.allocate<ImmediateAttrStorage>())
        ImmediateAttrStorage(std::move(key));
  }

  APInt value;
};
} // namespace detail
} // namespace rtg
} // namespace circt

Type ImmediateAttr::getType() const {
  return ImmediateType::get(getContext(), getValue().getBitWidth());
}

APInt ImmediateAttr::getValue() const { return getImpl()->value; }

Attribute ImmediateAttr::parse(AsmParser &odsParser, Type odsType) {
  llvm::SMLoc loc = odsParser.getCurrentLocation();

  APInt val;
  uint32_t width; // NOTE: this integer type should match the 'width' parameter
                  // type in immediate type.
  if (odsParser.parseLess() || odsParser.parseInteger(width) ||
      odsParser.parseComma() || odsParser.parseInteger(val) ||
      odsParser.parseGreater())
    return {};

  // If the attribute type is explicitly given, check that the bit-widths match.
  if (auto immTy = llvm::dyn_cast_or_null<ImmediateType>(odsType)) {
    if (immTy.getWidth() != width) {
      odsParser.emitError(loc) << "explicit immediate type bit-width does not "
                                  "match attribute bit-width, "
                               << immTy.getWidth() << " vs " << width;
      return {};
    }
  }

  if (width > val.getBitWidth()) {
    // sext is always safe here, even for unsigned values, because the
    // parseOptionalInteger method will return something with a zero in the
    // top bits if it is a positive number.
    val = val.sext(width);
  } else if (width < val.getBitWidth()) {
    // The parser can return an unnecessarily wide result.
    // This isn't a problem, but truncating off bits is bad.
    unsigned neededBits =
        val.isNegative() ? val.getSignificantBits() : val.getActiveBits();
    if (width < neededBits) {
      odsParser.emitError(loc)
          << "integer value out-of-range for bit-width " << width;
      return {};
    }
    val = val.trunc(width);
  }

  return ImmediateAttr::get(odsParser.getContext(), val);
}

void ImmediateAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getValue().getBitWidth() << ", " << getValue() << ">";
}

Type VirtualRegisterConfigAttr::getType() const {
  return getAllowedRegs()[0].getType();
}

LogicalResult VirtualRegisterConfigAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ArrayRef<rtg::RegisterAttrInterface> allowedRegs) {
  if (allowedRegs.empty())
    return emitError() << "must have at least one allowed register";

  if (!llvm::all_of(allowedRegs, [&](auto reg) {
        return reg.getType() == allowedRegs[0].getType();
      })) {
    return emitError() << "all allowed registers must be of the same type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

void RTGDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/RTG/IR/RTGAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGAttributes.cpp.inc"
