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

//------------------------------------------------------------------------------
// Helper functions.
//------------------------------------------------------------------------------
namespace {
ParseResult parseBankedDimensionList(DialectAsmParser &parser,
                                     SmallVectorImpl<int64_t> &shape,
                                     SmallVectorImpl<hir::DimKind> &dimKinds) {
  while (true) {
    int dimSize;
    // try to parse an ADDR dimension.
    mlir::OptionalParseResult result = parser.parseOptionalInteger(dimSize);
    if (result.hasValue()) {
      if (failed(result.getValue()))
        return failure();
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
      shape.push_back(dimSize);
      dimKinds.push_back(BANK);
      continue;
    }

    // else the dimension list is over.
    break;
  }
  return success();
}

mlir::OptionalParseResult
parseOptionalDelayAttr(DialectAsmParser &parser,
                       SmallVectorImpl<DictionaryAttr> &attrsList) {
  IntegerAttr delayAttr;
  auto *context = parser.getBuilder().getContext();
  if (succeeded(parser.parseOptionalKeyword("delay"))) {
    if (parser.parseAttribute(delayAttr, IntegerType::get(context, 64)))
      return failure();
  } else {
    return llvm::None;
  }
  attrsList.push_back(
      helper::getDictionaryAttr(parser.getBuilder(), "hir.delay", delayAttr));

  return success();
}

ParseResult parseMemrefPortsAttr(DialectAsmParser &parser,
                                 SmallVectorImpl<DictionaryAttr> &attrsList) {
  ArrayAttr memrefPortsAttr;
  if (parser.parseKeyword("ports") || parser.parseAttribute(memrefPortsAttr))
    return failure();
  attrsList.push_back(helper::getDictionaryAttr(
      parser.getBuilder(), "hir.memref.ports", memrefPortsAttr));

  return success();
}

ParseResult parseBusPortsAttr(DialectAsmParser &parser,
                              SmallVectorImpl<DictionaryAttr> &attrsList) {

  auto *context = parser.getBuilder().getContext();
  if (parser.parseKeyword("ports") || parser.parseLSquare())
    return failure();
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword("send")))
    keyword = "send";
  else if (succeeded(parser.parseOptionalKeyword("recv")))
    keyword = "recv";
  else
    return parser.emitError(parser.getCurrentLocation())
           << "Expected 'send' or 'recv' keyword.";

  if (parser.parseRSquare())
    return failure();

  attrsList.push_back(helper::getDictionaryAttr(
      parser.getBuilder(), "hir.bus.ports",
      ArrayAttr::get(context,
                     SmallVector<Attribute>({StringAttr::get(
                         parser.getBuilder().getContext(), keyword)}))));
  return success();
}

} // namespace

//------------------------------------------------------------------------------
// Type parsers.
//------------------------------------------------------------------------------
// Types.
// Memref Type.
static Type parseMemrefType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  SmallVector<hir::DimKind, 4> dimKinds;
  if (parser.parseLess() || parseBankedDimensionList(parser, shape, dimKinds) ||
      parser.parseType(elementType) || parser.parseGreater())
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
  printer << elementType << ">";
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
    if (failed(result.getValue()))
      return Type();
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
  SmallVector<Type, 4> inputTypes;
  SmallVector<DictionaryAttr, 4> inputAttrs;
  SmallVector<DictionaryAttr, 4> resultAttrs;
  if (parser.parseLParen())
    return Type();
  if (failed(parser.parseOptionalRParen())) {
    while (1) {
      Type type;
      auto typeLoc = parser.getCurrentLocation();
      if (parser.parseType(type))
        return Type();
      inputTypes.push_back(type);

      if (helper::isBuiltinSizedType(type)) {
        auto parseResult = parseOptionalDelayAttr(parser, inputAttrs);
        if (parseResult.hasValue()) {
          if (parseResult.getValue())
            return Type();
        } else {
          inputAttrs.push_back(helper::getDictionaryAttr(
              parser.getBuilder(), "hir.delay",
              parser.getBuilder().getI64IntegerAttr(0)));
        }
      } else if (type.dyn_cast<hir::MemrefType>()) {
        if (parseMemrefPortsAttr(parser, inputAttrs))
          return Type();
      } else if (helper::isBusType(type)) {
        if (parseBusPortsAttr(parser, inputAttrs))
          return Type();
      } else
        return parser.emitError(typeLoc)
                   << "hir.func type does not support input arguments of type "
                   << type,
               Type();
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
        Type resultTy;
        if (parser.parseType(resultTy))
          return Type();
        resultTypes.push_back(resultTy);
        if (!(helper::isBuiltinSizedType(resultTy) ||
              resultTy.isa<hir::TimeType>()))
          return parser.emitError(parser.getCurrentLocation(),
                                  "Only mlir builtin types are supported in "
                                  "function results."),
                 Type();
        auto parseResult = parseOptionalDelayAttr(parser, resultAttrs);

        if (parseResult.hasValue()) {
          if (failed(parseResult.getValue()))
            return Type();
        } else {
          resultAttrs.push_back(DictionaryAttr());
        }

        if (parser.parseOptionalComma())
          break;
      }

      if (parser.parseRParen())
        return Type();
    }
  }

  return hir::FuncType::get(parser.getBuilder().getContext(), inputTypes,
                            inputAttrs, resultTypes, resultAttrs);
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
  ArrayRef<DictionaryAttr> inputAttrs = moduleTy.getInputAttrs();
  ArrayRef<DictionaryAttr> resultAttrs = moduleTy.getResultAttrs();
  auto inputTypes = functionTy.getInputs();
  auto resultTypes = functionTy.getResults();
  for (size_t i = 0; i < inputTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << inputTypes[i];
    if (helper::isBuiltinSizedType(inputTypes[i])) {
      auto delay = helper::extractDelayFromDict(inputAttrs[i]);
      if (delay != 0)
        printer << " delay " << delay;
    } else if (inputTypes[i].dyn_cast<hir::MemrefType>()) {
      auto ports = helper::extractMemrefPortsFromDict(inputAttrs[i]);
      printer << " ports " << ports;
    } else if (helper::isBusType(inputTypes[i])) {
      auto busPortStr = helper::extractBusPortFromDict(inputAttrs[i]);
      printer << " ports [" << busPortStr << "]";
    }
  }
  printer << ") -> (";

  for (size_t i = 0; i < resultTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << resultTypes[i];
    if (!helper::isBuiltinSizedType(inputTypes[i]))
      continue;
    auto delay = helper::extractDelayFromDict(resultAttrs[i]);
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
