#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
using namespace circt;
using namespace hir;

// Types.
// Memref Type.
ParseResult parseBankedDimensionList(DialectAsmParser &parser,
                                     SmallVectorImpl<int64_t> &shape,
                                     SmallVectorImpl<hir::DimKind> &dimKinds) {
  while (true) {
    int dimSize;
    // try to parse an ADDR dimension.
    mlir::OptionalParseResult result = parser.parseOptionalInteger(dimSize);
    if (!result.hasValue())
      return failure();
    if (succeeded(result.getValue())) {
      if (parser.parseXInDimensionList())
        return failure();
      shape.push_back(dimSize);
      dimKinds.push_back(ADDR);
      continue;
    }

    // try to parse a BANK dimension.
    if (succeeded(parser.parseOptionalLParen())) {
      if (parser.parseKeyword("bank") || parser.parseInteger(dimSize) ||
          parser.parseRParen())
        return failure();
      if (parser.parseXInDimensionList())
        return failure();
      continue;
    }

    // else the dimension list is over.
    break;
  }
  return success();
}

static Type parseMemrefType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (parser.parseLess())
    return Type();
  SmallVector<hir::DimKind, 4> dimKinds;
  if (parseBankedDimensionList(parser, shape, dimKinds) ||
      parser.parseType(elementType))
    return Type();

  return MemrefType::get(context, shape, elementType, dimKinds);
}

static void printMemrefType(MemrefType memrefTy, DialectAsmPrinter &printer) {
  printer << memrefTy.getKeyword();
  printer << "<";
  auto shape = memrefTy.getShape();
  auto elementType = memrefTy.getElementType();
  auto dimKinds = memrefTy.getDimKinds();
  for (size_t i = 0; i < shape.size(); i++) {
    if (dimKinds[i] == ADDR)
      printer << shape[i] << "x";
    else
      printer << "(bank " + std::to_string(shape[i]) + ")x";
  }
  printer << elementType;
}

/// FuncType, GroupType, ArrayType

static Type parseFuncType(DialectAsmParser &parser, MLIRContext *context);
static Type parseGroupType(DialectAsmParser &parser, MLIRContext *context);
static Type parseArrayType(DialectAsmParser &parser, MLIRContext *context);
static void printGroupType(GroupType groupTy, DialectAsmPrinter &printer);
static void printFuncType(FuncType groupTy, DialectAsmPrinter &printer);
static void printArrayType(ArrayType groupTy, DialectAsmPrinter &printer);

namespace helper {

// forward declarations.
static Type parseInnerFuncType(DialectAsmParser &parser, MLIRContext *context);
static Type parseInnerGroupType(DialectAsmParser &parser, MLIRContext *context);
static Type parseInnerArrayType(DialectAsmParser &parser, MLIRContext *context);

static Type parseOptionalElementType(DialectAsmParser &parser,
                                     MLIRContext *context);
static std::pair<Type, Attribute>
parseElementTypeWithOptionalAttr(DialectAsmParser &parser,
                                 MLIRContext *context);

void printElementType(Type elementTy, DialectAsmPrinter &printer);
void printElementTypeWithOptionalAttr(DialectAsmPrinter &printer,
                                      Type elementTy, Attribute attr);

// Implementations.

static bool isValidElementType(Type t) {
  if (t.isa<IntegerType>() || t.isa<mlir::FloatType>() ||
      t.isa<hir::TimeType>() || t.isa<hir::MemrefType>())
    return true;
  return false;
}

static Type parseOptionalElementType(DialectAsmParser &parser,
                                     MLIRContext *context) {
  auto locType = parser.getCurrentLocation();

  if (!parser.parseOptionalKeyword("func"))
    return parseFuncType(parser, context);
  if (!parser.parseOptionalKeyword("group"))
    return parseGroupType(parser, context);
  if (!parser.parseOptionalKeyword("array"))
    return parseArrayType(parser, context);

  Type simpleTy; // non-container types.
  auto result = parser.parseOptionalType(simpleTy);

  if (result.hasValue()) {
    if (result.getValue())
      return parser.emitError(locType, "could not parse type!"), Type();
    if (!isValidElementType(simpleTy))
      return parser.emitError(locType,
                              "only integer, float, !hir.time, !hir.memref, "
                              "group, array and func types "
                              "are supported inside interface!"),
             Type();
    return simpleTy;
  }

  return Type();
}

static std::pair<Type, Attribute>
parseElementTypeWithOptionalAttr(DialectAsmParser &parser,
                                 MLIRContext *context) {
  Type elementType;
  Attribute attr = nullptr;
  elementType = parseOptionalElementType(parser, context);
  if (!elementType) {
    if (!parser.parseKeyword("send"))
      attr = StringAttr::get(context, "send");
    elementType = parseOptionalElementType(parser, context);
  }
  if (!elementType)
    parser.emitError(parser.getNameLoc(), "Could not parse element type!");
  return std::make_pair(elementType, attr);
}

static Type parseInnerFuncType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<Type, 4> argTypes;
  SmallVector<Attribute, 4> inputAttrs;
  SmallVector<Attribute, 4> outputDelays;
  auto builder = parser.getBuilder();
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
      if (helper::isPrimitiveType(type)) {
        if (succeeded(parser.parseOptionalKeyword("delay"))) {
          if (parser.parseAttribute(delayAttr, getIntegerType(context, 64)))
            return Type();
        } else {
          delayAttr = helper::getIntegerAttr(context, 64, 0);
        }

        inputAttrs.push_back(DictionaryAttr::get(
            context, SmallVector<NamedAttribute, 1>(
                         {builder.getNamedAttr("delay", delayAttr)})));
      } else if (type.dyn_cast<hir::MemrefType>()) {
        ArrayAttr portsAttr;
        if (parser.parseAttribute(portsAttr))
          return Type();
        inputAttrs.push_back(DictionaryAttr::get(
            context, SmallVector<NamedAttribute, 1>(
                         {builder.getNamedAttr("ports", portsAttr)})));
      } else if (type.dyn_cast<hir::BusType>()) {
        std::string busPortKind;
        if (parser.parseLSquare() || parser.parseKeyword(busPortKind) ||
            parser.parseRSquare())
          return Type();

        inputAttrs.push_back(DictionaryAttr::get(
            context, SmallVector<NamedAttribute, 1>({builder.getNamedAttr(
                         "ports", StringAttr::get(context, busPortKind))})));
      }
      count++;
      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseRParen())
      return Type();
  }

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

  FunctionType functionTy =
      parser.getBuilder().getFunctionType(argTypes, resultTypes);
  return hir::FuncType::get(parser.getBuilder().getContext(), functionTy,
                            builder.getArrayAttr(inputAttrs),
                            parser.getBuilder().getArrayAttr(outputDelays));
}

static Type parseInnerGroupType(DialectAsmParser &parser,
                                MLIRContext *context) {
  SmallVector<Type> elementTypes;
  SmallVector<Attribute> attrs;

  do {
    auto typeAndAttr =
        helper::parseElementTypeWithOptionalAttr(parser, context);
    Type elementTy = std::get<0>(typeAndAttr);
    Attribute attr = std::get<1>(typeAndAttr);
    elementTypes.push_back(elementTy);
    attrs.push_back(attr);
  } while (succeeded(parser.parseOptionalComma()));

  return GroupType::get(context, elementTypes, attrs);
}

static Type parseInnerArrayType(DialectAsmParser &parser,
                                MLIRContext *context) {
  SmallVector<int64_t, 4> dims;
  StringAttr attr;
  Type elementType;
  if (!parser.parseOptionalKeyword("send"))
    attr = StringAttr::get(context, "send");

  // parse the dimensions.
  if (parser.parseDimensionList(dims, false))
    return Type();

  // parse the element type.
  elementType = helper::parseOptionalElementType(parser, context);
  if (!elementType)
    return Type();

  return ArrayType::get(context, dims, elementType, attr);
}

void printElementType(Type elementTy, DialectAsmPrinter &printer) {
  if (FuncType moduleTy = elementTy.dyn_cast<hir::FuncType>()) {
    printFuncType(moduleTy, printer);
  } else if (GroupType groupTy = elementTy.dyn_cast<GroupType>()) {
    printGroupType(groupTy, printer);
  } else if (ArrayType arrayTy = elementTy.dyn_cast<ArrayType>()) {
    printArrayType(arrayTy, printer);
  } else
    printer << elementTy;
}

void printElementTypeWithOptionalAttr(DialectAsmPrinter &printer,
                                      Type elementTy, Attribute attr) {
  if (attr)
    printer << attr << " ";
  helper::printElementType(elementTy, printer);
}
} // namespace helper

static Type parseFuncType(DialectAsmParser &parser, MLIRContext *context) {
  if (parser.parseLess())
    return Type();
  Type funcTy = helper::parseInnerFuncType(parser, context);
  if (parser.parseGreater())
    return Type();
  return funcTy;
}

static Type parseGroupType(DialectAsmParser &parser, MLIRContext *context) {
  if (parser.parseLess())
    return Type();
  Type groupTy = helper::parseInnerGroupType(parser, context);
  // Finish parsing the group.
  if (parser.parseGreater())
    return Type();
  return groupTy;
}

static Type parseArrayType(DialectAsmParser &parser, MLIRContext *context) {
  // if no starting LSquare then this is not an array.
  if (parser.parseLess())
    return Type();
  Type arrayTy = helper::parseInnerArrayType(parser, context);
  // Finish parsing the array.
  if (parser.parseGreater())
    return Type();

  return arrayTy;
}

static Type parseBusType(DialectAsmParser &parser, MLIRContext *context) {

  SmallVector<Type> elementTypes;
  SmallVector<BusDirection> directions;

  // start parsing
  if (parser.parseLess())
    return Type();
  // parse comma separated list of bus-directions and types.
  do {
    Type ty;

    if (succeeded(parser.parseOptionalKeyword("flip")))
      directions.push_back(FLIP);
    else
      directions.push_back(SAME);
    if (parser.parseType(ty))
      return Type();
    elementTypes.push_back(ty);
  } while (!parser.parseOptionalComma());

  // Finish parsing the group.
  if (parser.parseGreater())
    return Type();

  return hir::BusType::get(context, elementTypes, directions);
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

  if (typeKeyword == FuncType::getKeyword())
    return parseFuncType(parser, getContext());

  if (typeKeyword == GroupType::getKeyword())
    return parseGroupType(parser, getContext());

  if (typeKeyword == ArrayType::getKeyword())
    return parseArrayType(parser, getContext());

  if (typeKeyword == BusType::getKeyword())
    return parseBusType(parser, getContext());

  return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
}

static void printFuncType(FuncType moduleTy, DialectAsmPrinter &printer) {
  printer << "func<(";
  FunctionType functionTy = moduleTy.getFunctionType();
  ArrayAttr inputAttrs = moduleTy.getInputAttrs();
  ArrayAttr outputDelays = moduleTy.getOutputDelays();
  auto inputTypes = functionTy.getInputs();
  auto outputTypes = functionTy.getResults();
  for (size_t i = 0; i < inputTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << inputTypes[i];
    if (helper::isPrimitiveType(inputTypes[i])) {
      auto delay = inputAttrs[i]
                       .dyn_cast<DictionaryAttr>()
                       .getNamed("delay")
                       .getValue()
                       .second.dyn_cast<IntegerAttr>()
                       .getInt();
      if (delay != 0)
        printer << " delay " << delay;
    } else if (inputTypes[i].dyn_cast<hir::MemrefType>()) {
      auto ports = inputAttrs[i]
                       .dyn_cast<DictionaryAttr>()
                       .getNamed("ports")
                       .getValue()
                       .second;
      printer << " ports " << ports;
    } else if (inputTypes[i].dyn_cast<hir::BusType>()) {
      auto busPortKind = inputAttrs[i]
                             .dyn_cast<DictionaryAttr>()
                             .getNamed("ports")
                             .getValue()
                             .second.dyn_cast<StringAttr>()
                             .getValue()
                             .str();
      printer << " ports [" << busPortKind << "]";
    }
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
  printer << ")>";
}

static void printGroupType(GroupType groupTy, DialectAsmPrinter &printer) {
  printer << "group<";
  ArrayRef<Type> elementTypes = groupTy.getElementTypes();
  ArrayRef<Attribute> attrs = groupTy.getAttributes();
  for (unsigned i = 0; i < elementTypes.size(); i++) {
    auto attr = attrs[i];
    auto elementTy = elementTypes[i];
    if (i > 0)
      printer << ", ";
    helper::printElementTypeWithOptionalAttr(printer, elementTy, attr);
  }
  printer << ">";
}

static void printArrayType(ArrayType arrayTy, DialectAsmPrinter &printer) {
  printer << "array<";
  ArrayRef<int64_t> dims = arrayTy.getDimensions();
  Type elementTy = arrayTy.getElementType();
  for (auto dim : dims) {
    printer << dim << "x";
  }
  helper::printElementType(elementTy, printer);
  printer << ">";
}

static void printBusType(BusType busTy, DialectAsmPrinter &printer) {
  ArrayRef<Type> elementTypes = busTy.getElementTypes();
  ArrayRef<BusDirection> directions = busTy.getElementDirections();

  printer << "bus<";
  for (int i = 0; i < (int)elementTypes.size(); i++) {
    if (i > 0)
      printer << ", ";

    if (directions[i] == BusDirection::FLIP)
      printer << "flip ";
    printer << elementTypes[i];
  }
  printer << ">";
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
  if (FuncType funcTy = type.dyn_cast<FuncType>()) {
    printFuncType(funcTy, printer);
    return;
  }
  if (GroupType groupTy = type.dyn_cast<GroupType>()) {
    printGroupType(groupTy, printer);
    return;
  }
  if (ArrayType arrayTy = type.dyn_cast<ArrayType>()) {
    printArrayType(arrayTy, printer);
    return;
  }
  if (BusType busTy = type.dyn_cast<BusType>()) {
    printBusType(busTy, printer);
    return;
  }
}
