#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"

#include "circt/Dialect/HIR/helper.h"
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

// Types.
// Memref Type.
static Type parseMemrefType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (parser.parseLess())
    return Type();

  if (parser.parseDimensionList(shape) || parser.parseType(elementType))
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
    if (parser.parseGreater())
      return Type();
  }

  if (hasBankedDims) {
    bankedDims = attr1.dyn_cast<ArrayAttr>();
    portAttr = attr2.dyn_cast<DictionaryAttr>();
    if (!bankedDims) {
      parser.emitError(loc1, "expected banked dims!");
      return Type();
    }
    if (!portAttr) {
      parser.emitError(loc2, "expected port attributes!");
      return Type();
    }
  } else {
    portAttr = attr1.dyn_cast<DictionaryAttr>();
    bankedDims = ArrayAttr::get(context, SmallVector<Attribute>());
    if (!portAttr) {
      parser.emitError(loc1, "expected port attributes!");
      return Type();
    }
  }
  assert(bankedDims);
  assert(portAttr);
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
  if (t.isa<IntegerType>() || t.isa<FloatType>() || t.isa<hir::TimeType>() ||
      t.isa<hir::MemrefType>())
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
  SmallVector<Attribute, 4> inputDelays;
  SmallVector<Attribute, 4> outputDelays;
  ArrayAttr inputDelayAttr, outputDelayAttr;
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
      if (helper::isPrimitiveType(type) &&
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
  return hir::FuncType::get(parser.getBuilder().getContext(), functionTy,
                            inputDelayAttr, outputDelayAttr);
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
  if (parser.parseLess())
    return Type();

  SmallVector<Type> elementTypes;
  SmallVector<PortKind> directions;
  DictionaryAttr proto;

  do {
    Type ty;

    if (!parser.parseOptionalKeyword("rd")) {
      if (parser.parseType(ty))
        return Type();
      directions.push_back(PortKind::rd);
      elementTypes.push_back(ty);
    } else if (!parser.parseOptionalKeyword("wr")) {
      if (parser.parseType(ty))
        return Type();
      directions.push_back(PortKind::wr);
      elementTypes.push_back(ty);
    } else if (!parser.parseOptionalKeyword("proto")) {
      if (parser.parseAttribute(proto))
        return Type();
      break;
    } else {
      if (parser.parseType(ty))
        return Type();
      directions.push_back(PortKind::rw);
      elementTypes.push_back(ty);
    }
  } while (!parser.parseOptionalComma());

  // Finish parsing the group.
  if (parser.parseGreater())
    return Type();

  if (!proto)
    proto = DictionaryAttr::get(context, SmallVector<NamedAttribute, 4>());

  return hir::BusType::get(context, elementTypes, directions, proto);
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

  if (typeKeyword == ConstType::getKeyword())
    return ConstType::get(getContext());

  return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
}

static void printFuncType(FuncType moduleTy, DialectAsmPrinter &printer) {
  printer << "func<(";
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
  ArrayRef<PortKind> directions = busTy.getElementDirections();
  DictionaryAttr proto = busTy.getProto();

  printer << "bus<";
  for (int i = 0; i < (int)elementTypes.size(); i++) {
    Type elementTy = elementTypes[i];
    PortKind direction = directions[i];

    if (i > 0)
      printer << ", ";

    if (direction == PortKind::rd)
      printer << "rd ";
    else if (direction == PortKind::wr)
      printer << "wr ";

    printer << elementTy;
  }
  if (!proto.empty())
    printer << ", proto " << proto;
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
  if (ConstType constTy = type.dyn_cast<ConstType>()) {
    printer << constTy.getKeyword();
    return;
  }
}
