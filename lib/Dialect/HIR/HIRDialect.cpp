//=========- HIR.cpp - Registration, Parser & Printer----------------------===//
//
// This file implements parsers and printers for Types and registers the types
// and operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace hir;

HIRDialect::HIRDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HIRDialect>()) {
  addTypes<TimeType, GroupType, ArrayType, ConstType, MemrefType, WireType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/HIR.cpp.inc"
      >();
}

namespace Helpers {
static ParseResult parsePortKind(DialectAsmParser &parser,
                                 Details::PortKind &port) {
  if (!parser.parseOptionalKeyword("rw"))
    port = Details::rw;
  else if (!parser.parseOptionalKeyword("r"))
    port = Details::r;
  else if (!parser.parseOptionalKeyword("w"))
    port = Details::w;
  else
    return failure();
  return success();
}

static OptionalParseResult
parseOptionalMemrefPacking(DialectAsmParser &parser,
                           SmallVectorImpl<unsigned> &packing) {
  if (parser.parseOptionalKeyword("packing"))
    return OptionalParseResult(None);
  if (parser.parseEqual() || parser.parseLSquare())
    return failure();
  if (!parser.parseOptionalRSquare())
    return success();

  unsigned dim;
  if (parser.parseInteger(dim))
    return failure();
  packing.push_back(dim);
  while (!parser.parseOptionalComma()) {
    if (parser.parseInteger(dim))
      return failure();
    packing.push_back(dim);
  }
  if (parser.parseRSquare())
    return failure();
  return success();
}

static ParseResult parseShapedType(DialectAsmParser &parser,
                                   SmallVectorImpl<unsigned> &shape,
                                   Type &elementType) {
  unsigned dim;

  if (parser.parseInteger(dim))
    return failure();
  shape.push_back(dim);
  if (parser.parseOptionalStar())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '*' followed by type"),
           failure();
  while (true) {
    auto optionalParseRes = parser.parseOptionalInteger(dim);
    if (!optionalParseRes.hasValue())
      break;
    if (optionalParseRes.getValue())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected integer literal or type"),
             failure();
    shape.push_back(dim);
    if (parser.parseOptionalStar())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected '*' followed by type"),
             failure();
  }
  if (parser.parseType(elementType))
    return failure();
  return success();
}
}; // namespace Helpers.

// Types.

// Memref Type.
static Type parseMemrefType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<unsigned, 4> shape;
  Type elementType;
  hir::Details::PortKind defaultPort = hir::Details::rw;
  if (parser.parseLess())
    return Type();

  if (Helpers::parseShapedType(parser, shape, elementType))
    return Type();

  SmallVector<unsigned, 4> defaultPacking;
  // default packing is [0,1,2,3] for shape = [d3,d2,d1,d0] i.e. d0 is fastest
  // changing dim in linear index and no distributed dimensions.
  for (size_t i = 0; i < shape.size(); i++)
    defaultPacking.push_back(i);

  if (!parser.parseOptionalGreater())
    return MemrefType::get(context, shape, elementType, defaultPacking,
                           defaultPort);

  if (parser.parseComma())
    return Type();

  SmallVector<unsigned, 4> packing;
  OptionalParseResult hasPacking =
      Helpers::parseOptionalMemrefPacking(parser, packing);

  if (hasPacking.hasValue()) {
    if (hasPacking.getValue())
      return Type();
    if (!parser.parseOptionalGreater())
      return MemrefType::get(context, shape, elementType, packing, defaultPort);
    if (parser.parseComma())
      return Type();
  }

  hir::Details::PortKind port;
  if (Helpers::parsePortKind(parser, port) || parser.parseGreater())
    return Type();

  return MemrefType::get(context, shape, elementType,
                         hasPacking.hasValue() ? packing : defaultPacking,
                         port);
}
static void printMemrefType(MemrefType memrefTy, DialectAsmPrinter &printer) {
  printer << memrefTy.getKeyword();
  printer << "<";
  auto shape = memrefTy.getShape();
  auto elementType = memrefTy.getElementType();
  auto packing = memrefTy.getPacking();
  auto port = memrefTy.getPort();
  for (auto dim : shape)
    printer << dim << "*";

  printer << elementType << ", packing = [";

  for (size_t i = 0; i < packing.size(); i++)
    printer << packing[i] << ((i == (packing.size() - 1)) ? "" : ", ");
  printer << "]";

  if (port == Details::r) {
    printer << ", r";
  } else if (port == Details::w) {
    printer << ", w";
  } else if (port == Details::rw) {
    printer << "";
  } else {
    printer << ", unknown";
  }
  printer << ">";
}

// Interface Type
// hir.interface< ((#in i32, #out f32,#call ()->i1),#in [4xi32], [2x(#in i1,
// #out i1, #in i8)])>
namespace helper {

// forward declarations.
static Type parseOptionalGroup(DialectAsmParser &parser, MLIRContext *context);
static Type parseOptionalArray(DialectAsmParser &parser, MLIRContext *context);
static Type parseOptionalInterfaceElementType(DialectAsmParser &parser,
                                              MLIRContext *context);
static std::pair<Type, Attribute>
parseInterfaceElementTypeWithOptionalAttr(DialectAsmParser &parser,
                                          MLIRContext *context);
static void printGroup(GroupType groupTy, DialectAsmPrinter &printer);
static void printArray(ArrayType groupTy, DialectAsmPrinter &printer);
void printInterfaceElementType(Type elementTy, DialectAsmPrinter &printer);

// Implementations.
static Type parseOptionalGroup(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<Type> elementTypes;
  SmallVector<Attribute> attrs;

  // if no starting LParen then this is not a group.
  if (failed(parser.parseOptionalLParen()))
    return Type();

  // parse the group members.
  do {
    auto typeAndAttr =
        parseInterfaceElementTypeWithOptionalAttr(parser, context);
    Type elementTy = std::get<0>(typeAndAttr);
    Attribute attr = std::get<1>(typeAndAttr);
    elementTypes.push_back(elementTy);
    attrs.push_back(attr);
  } while (succeeded(parser.parseOptionalComma()));

  // Finish parsing the group.
  if (parser.parseRParen())
    return Type();
  return GroupType::get(context, elementTypes, attrs);
}

static Type parseOptionalArray(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<int64_t, 4> dims;
  Type elementType;

  // if no starting LSquare then this is not an array.
  if (failed(parser.parseOptionalLSquare()))
    return Type();

  // parse the dimensions.
  if (parser.parseDimensionList(dims, false))
    return Type();

  // parse the element type.
  elementType = parseOptionalInterfaceElementType(parser, context);
  if (!elementType)
    return Type();
  // Finish parsing the array.
  if (parser.parseRSquare())
    return Type();
  return ArrayType::get(context, dims, elementType);
}

static Type parseOptionalInterfaceElementType(DialectAsmParser &parser,
                                              MLIRContext *context) {
  Type builtinType;
  Type groupTy = parseOptionalGroup(parser, context);
  Type arrayTy = parseOptionalArray(parser, context);
  OptionalParseResult result = parser.parseOptionalType(builtinType);
  if (result.hasValue() && succeeded(result.getValue()))
    return builtinType;
  if (groupTy)
    return groupTy;
  if (arrayTy)
    return arrayTy;
  return Type();
}

static std::pair<Type, Attribute>
parseInterfaceElementTypeWithOptionalAttr(DialectAsmParser &parser,
                                          MLIRContext *context) {
  Type elementType;
  Attribute attr = nullptr;
  elementType = parseOptionalInterfaceElementType(parser, context);
  if (!elementType) {
    parser.parseAttribute(attr);
    elementType = parseOptionalInterfaceElementType(parser, context);
  }
  return std::make_pair(elementType, attr);
}

static void printGroup(GroupType groupTy, DialectAsmPrinter &printer) {
  printer << "(";
  ArrayRef<Type> elementTypes = groupTy.getElementTypes();
  ArrayRef<Attribute> attrs = groupTy.getAttributes();
  for (unsigned i = 0; i < elementTypes.size(); i++) {
    auto attr = attrs[i];
    auto elementTy = elementTypes[i];
    if (attr)
      printer << attr;
    printInterfaceElementType(elementTy, printer);
  }
  printer << ")";
}

static void printArray(ArrayType arrayTy, DialectAsmPrinter &printer) {
  printer << "[";
  ArrayRef<int64_t> dims;
  Type elementTy = arrayTy.getElementType();
  for (auto dim : dims) {
    printer << dim << "x";
  }
  printInterfaceElementType(elementTy, printer);
  printer << "]";
}

void printInterfaceElementType(Type elementTy, DialectAsmPrinter &printer) {
  if (GroupType groupTy = elementTy.dyn_cast<GroupType>()) {
    helper::printGroup(groupTy, printer);
  } else if (ArrayType arrayTy = elementTy.dyn_cast<ArrayType>()) {
    helper::printArray(arrayTy, printer);
  } else
    printer << elementTy;
}
} // namespace helper

static Type parseInterfaceType(DialectAsmParser &parser, MLIRContext *context) {
  if (parser.parseLess())
    return Type();

  // Parse the element type and optional attribute.
  auto typeAndAttr =
      helper::parseInterfaceElementTypeWithOptionalAttr(parser, context);
  Type elementType = std::get<0>(typeAndAttr);
  Attribute attr = std::get<1>(typeAndAttr);
  if (!elementType)
    return Type();

  // Finish parsing.
  if (parser.parseGreater())
    return Type();

  return InterfaceType::get(context, elementType, attr);
}
static void printInterfaceType(InterfaceType interfaceTy,
                               DialectAsmPrinter &printer) {
  printer << interfaceTy.getKeyword();
  printer << "<";
  Type elementTy = interfaceTy.getElementType();
  Attribute attr = interfaceTy.getAttribute();
  if (attr)
    printer << attr;
  helper::printInterfaceElementType(elementTy, printer);
  printer << ">";
}

// WireType.
static Type parseWireType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<unsigned, 4> shape;
  Type elementType;
  hir::Details::PortKind defaultPort = hir::Details::rw;
  if (parser.parseLess())
    return Type();

  if (Helpers::parseShapedType(parser, shape, elementType))
    return Type();

  SmallVector<unsigned, 4> defaultPacking;
  // default packing is [0,1,2,3] for shape = [d3,d2,d1,d0] i.e. d0 is fastest
  // changing dim in linear index and no distributed dimensions.
  for (size_t i = 0; i < shape.size(); i++)
    defaultPacking.push_back(i);

  if (!parser.parseOptionalGreater())
    return WireType::get(context, shape, elementType, defaultPort);

  if (parser.parseComma())
    return Type();

  hir::Details::PortKind port;
  if (Helpers::parsePortKind(parser, port) || parser.parseGreater())
    return Type();

  return WireType::get(context, shape, elementType, port);
}

static void printWireType(WireType wireTy, DialectAsmPrinter &printer) {
  printer << wireTy.getKeyword();
  printer << "<";
  auto shape = wireTy.getShape();
  auto elementType = wireTy.getElementType();
  auto port = wireTy.getPort();
  for (auto dim : shape)
    printer << dim << "*";

  printer << elementType;

  if (port == Details::r) {
    printer << ", r";
  } else if (port == Details::w) {
    printer << ", w";
  } else if (port == Details::rw) {
    printer << "";
  } else {
    printer << ", unknown";
  }
  printer << ">";
}

// parseType and printType.
Type HIRDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  if (parser.parseKeyword(&typeKeyword))
    return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();

  if (typeKeyword == TimeType::getKeyword())
    return TimeType::get(getContext());

  if (typeKeyword == MemrefType::getKeyword())
    return parseMemrefType(parser, getContext());

  if (typeKeyword == InterfaceType::getKeyword())
    return parseInterfaceType(parser, getContext());

  if (typeKeyword == WireType::getKeyword())
    return parseWireType(parser, getContext());

  if (typeKeyword == ConstType::getKeyword())
    return ConstType::get(getContext());

  return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
}

void HIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (TimeType timeTy = type.dyn_cast<TimeType>()) {
    printer << timeTy.getKeyword();
    return;
  }
  if (MemrefType memrefTy = type.dyn_cast<MemrefType>()) {
    printMemrefType(memrefTy, printer);
    return;
  }
  if (InterfaceType interfaceTy = type.dyn_cast<InterfaceType>()) {
    printInterfaceType(interfaceTy, printer);
    return;
  }
  if (WireType wireTy = type.dyn_cast<WireType>()) {
    printWireType(wireTy, printer);
    return;
  }
  if (ConstType constTy = type.dyn_cast<ConstType>()) {
    printer << constTy.getKeyword();
    return;
  }
}
