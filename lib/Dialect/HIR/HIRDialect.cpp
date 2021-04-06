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
#include "parserHelper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace hir;

HIRDialect::HIRDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HIRDialect>()) {
  addTypes<TimeType, GroupType, ArrayType, InterfaceType, ConstType, ModuleType,
           MemrefType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/HIR.cpp.inc"
      >();
}

namespace Helpers {
static ParseResult parseShapedType(DialectAsmParser &parser,
                                   SmallVectorImpl<int64_t> &shape,
                                   Type &elementType) {

  if (parser.parseDimensionList(shape) || parser.parseType(elementType))
    return failure();
  return success();
}
}; // namespace Helpers.

// Types.

// Memref Type.
static Type parseMemrefType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (parser.parseLess())
    return Type();

  if (Helpers::parseShapedType(parser, shape, elementType))
    return Type();

  if (parser.parseComma())
    return Type();

  bool hasBankedDims = false;
  mlir::Attribute attr1, attr2;
  llvm::SMLoc loc1, loc2;
  ArrayAttr bankedDims;
  DictionaryAttr portAttr;
  loc1 = parser.getCurrentLocation();
  if (parser.parseAttribute(attr1))
    return Type();
  if (parser.parseOptionalGreater()) {
    if (parser.parseComma())
      return Type();
    hasBankedDims = true;
    loc2 = parser.getCurrentLocation();
    parser.parseAttribute(attr2);
  }
  if (parser.parseGreater())
    return Type();

  if (hasBankedDims) {
    bankedDims = attr1.dyn_cast<ArrayAttr>();
    if (!bankedDims) {
      parser.emitError(loc1, "expected banked dims!");
      return Type();
    }
    portAttr = attr2.dyn_cast<DictionaryAttr>();
    if (!portAttr) {
      parser.emitError(loc2, "expected port attributes!");
      return Type();
    }
  } else {
    portAttr = attr1.dyn_cast<DictionaryAttr>();
    if (!portAttr) {
      parser.emitError(loc1, "expected port attributes!");
      return Type();
    }
  }
  return MemrefType::get(context, shape, elementType, bankedDims, portAttr);
}

static void printMemrefType(MemrefType memrefTy, DialectAsmPrinter &printer) {
  printer << memrefTy.getKeyword();
  printer << "<";
  auto shape = memrefTy.getShape();
  auto elementType = memrefTy.getElementType();
  auto bankedDims = memrefTy.getBankedDims();
  auto portAttrs = memrefTy.getPortAttrs();
  for (auto dim : shape)
    printer << dim << "x";

  printer << elementType << ", ";
  if (!bankedDims.empty())
    printer << bankedDims << ", ";
  printer << portAttrs << ">";
}

// Interface Type
// hir.interface< ((#in i32, #out f32,#call ()->i1),#in [4xi32], [2x(#in i1,
// #out i1, #in i8)])>
namespace helper {

// forward declarations.
Type parseOptionalModule(DialectAsmParser &parser, MLIRContext *context);
static Type parseOptionalGroup(DialectAsmParser &parser, MLIRContext *context);
static Type parseOptionalArray(DialectAsmParser &parser, MLIRContext *context);
static Type parseOptionalInterfaceElementType(DialectAsmParser &parser,
                                              MLIRContext *context);
static std::pair<Type, Attribute>
parseInterfaceElementTypeWithOptionalAttr(DialectAsmParser &parser,
                                          MLIRContext *context);
static void printModule(ModuleType moduleTy, DialectAsmPrinter &printer);
static void printGroup(GroupType groupTy, DialectAsmPrinter &printer);
static void printArray(ArrayType groupTy, DialectAsmPrinter &printer);
void printInterfaceElementType(Type elementTy, DialectAsmPrinter &printer);
void printInterfaceElementTypeWithOptionalAttr(DialectAsmPrinter &printer,
                                               Type elementTy, Attribute attr);

// Implementations.
Type parseOptionalModule(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<Type, 4> argTypes;
  SmallVector<Attribute, 4> inputDelays;
  SmallVector<Attribute, 4> outputDelays;
  ArrayAttr inputDelayAttr, outputDelayAttr;
  if (parser.parseOptionalKeyword("func"))
    return Type();
  if (parser.parseLParen())
    return Type();
  int count = 0;
  if (parser.parseOptionalRParen()) {
    while (1) {
      Type type;
      IntegerAttr delayAttr;
      if (parser.parseType(type))
        return Type();
      argTypes.push_back(type);
      if (type.isa<IntegerType>() &&
          succeeded(parser.parseOptionalKeyword("delay"))) {
        if (parser.parseAttribute(
                delayAttr,
                getIntegerType(parser.getBuilder().getContext(), 64)))
          return Type();
        inputDelays.push_back(delayAttr);
      } else {
        // Default delay is 0.
        inputDelays.push_back(
            getIntegerAttr(parser.getBuilder().getContext(), 64, 0));
      }
      count++;
      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseRParen())
      return Type();
  }

  inputDelayAttr = parser.getBuilder().getArrayAttr(inputDelays);

  // Process output types and delays.
  SmallVector<Type, 4> resultTypes;
  if (!parser.parseOptionalArrow()) {
    if (parser.parseLParen())
      return Type();
    if (parser.parseOptionalRParen()) {
      while (1) {
        Type type;
        IntegerAttr delayAttr;
        if (parser.parseType(type))
          return Type();
        resultTypes.push_back(type);
        if (type.isa<IntegerType>() &&
            succeeded(parser.parseOptionalKeyword("delay"))) {
          if (parser.parseAttribute(
                  delayAttr,
                  getIntegerType(parser.getBuilder().getContext(), 64)))
            return Type();
          outputDelays.push_back(delayAttr);
        } else {
          // Default delay is 0.
          outputDelays.push_back(
              getIntegerAttr(parser.getBuilder().getContext(), 64, 0));
        }
        if (parser.parseOptionalComma())
          break;
      }

      if (parser.parseRParen())
        return Type();
    }
  }

  outputDelayAttr = parser.getBuilder().getArrayAttr(outputDelays);
  FunctionType functionTy =
      parser.getBuilder().getFunctionType(argTypes, resultTypes);
  return hir::ModuleType::get(parser.getBuilder().getContext(), functionTy,
                              inputDelayAttr, outputDelayAttr);
}

static Type parseOptionalGroup(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<Type> elementTypes;
  SmallVector<Attribute> attrs;

  // if not starting with "group" keyword then this is not a group.
  if (parser.parseOptionalKeyword("group"))
    return Type();
  if (parser.parseLParen())
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
  if (parser.parseOptionalKeyword("array"))
    return Type();
  if (parser.parseLSquare())
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
  Type simpleType;
  auto locType = parser.getCurrentLocation();
  OptionalParseResult result = parser.parseOptionalType(simpleType);
  if (result.hasValue() && succeeded(result.getValue())) {
    if (simpleType.isa<IntegerType>() || simpleType.isa<FloatType>() ||
        simpleType.isa<hir::TimeType>())
      return simpleType;
    return parser.emitError(
               locType,
               "only integer, float, !hir.time, group, array and func types "
               "are supported inside interface!"),
           Type();
  }
  Type moduleTy = parseOptionalModule(parser, context);
  if (moduleTy)
    return moduleTy;
  Type groupTy = parseOptionalGroup(parser, context);
  if (groupTy)
    return groupTy;
  Type arrayTy = parseOptionalArray(parser, context);
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
  if (!elementType)
    parser.emitError(parser.getNameLoc(), "Could not parse element type!");
  return std::make_pair(elementType, attr);
}

static void printModule(ModuleType moduleTy, DialectAsmPrinter &printer) {
  printer << "func(";
  FunctionType functionTy = moduleTy.getFunctionType();
  ArrayAttr inputDelays = moduleTy.getInputDelays();
  ArrayAttr outputDelays = moduleTy.getOutputDelays();
  auto inputTypes = functionTy.getInputs();
  auto outputTypes = functionTy.getResults();
  for (size_t i = 0; i < inputTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << inputTypes[i];
    auto delay = inputDelays[i].dyn_cast<IntegerAttr>().getInt();
    if (delay != 0)
      printer << " delay " << delay;
  }
  printer << ") -> (";

  for (size_t i = 0; i < outputTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << outputTypes[i];
    auto delay = outputDelays[i].dyn_cast<IntegerAttr>().getInt();
    if (delay != 0)
      printer << " delay " << delay;
  }
  printer << ")";
}

static void printGroup(GroupType groupTy, DialectAsmPrinter &printer) {
  printer << "group(";
  ArrayRef<Type> elementTypes = groupTy.getElementTypes();
  ArrayRef<Attribute> attrs = groupTy.getAttributes();
  for (unsigned i = 0; i < elementTypes.size(); i++) {
    auto attr = attrs[i];
    auto elementTy = elementTypes[i];
    if (i > 0)
      printer << ", ";
    printInterfaceElementTypeWithOptionalAttr(printer, elementTy, attr);
  }
  printer << ")";
}

static void printArray(ArrayType arrayTy, DialectAsmPrinter &printer) {
  printer << "array[";
  ArrayRef<int64_t> dims = arrayTy.getDimensions();
  Type elementTy = arrayTy.getElementType();
  for (auto dim : dims) {
    printer << dim << "x";
  }
  printInterfaceElementType(elementTy, printer);
  printer << "]";
}

void printInterfaceElementType(Type elementTy, DialectAsmPrinter &printer) {
  if (ModuleType moduleTy = elementTy.dyn_cast<hir::ModuleType>()) {
    helper::printModule(moduleTy, printer);
  } else if (GroupType groupTy = elementTy.dyn_cast<GroupType>()) {
    helper::printGroup(groupTy, printer);
  } else if (ArrayType arrayTy = elementTy.dyn_cast<ArrayType>()) {
    helper::printArray(arrayTy, printer);
  } else
    printer << elementTy;
}

void printInterfaceElementTypeWithOptionalAttr(DialectAsmPrinter &printer,
                                               Type elementTy, Attribute attr) {
  if (attr)
    printer << attr << " ";
  helper::printInterfaceElementType(elementTy, printer);
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
  helper::printInterfaceElementTypeWithOptionalAttr(printer, elementTy, attr);
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
  if (ConstType constTy = type.dyn_cast<ConstType>()) {
    printer << constTy.getKeyword();
    return;
  }
}
