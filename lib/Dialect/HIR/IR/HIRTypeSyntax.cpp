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

// Implementations.
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
          resultAttrs.push_back(helper::getDictionaryAttr(
              parser.getBuilder(), "hir.delay",
              parser.getBuilder().getI64IntegerAttr(0)));
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

static Type parseFuncType(DialectAsmParser &parser, MLIRContext *context) {
  if (parser.parseLess())
    return Type();
  Type funcTy = parseInnerFuncType(parser, context);
  if (parser.parseGreater())
    return Type();
  return funcTy;
}

static Type parseBusType(DialectAsmParser &parser, MLIRContext *context) {

  Type elementTy;
  if (parser.parseLess() || parser.parseType(elementTy) ||
      parser.parseGreater())
    return Type();
  return hir::BusType::get(context, elementTy);
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
    if (!helper::isBuiltinSizedType(resultTypes[i]))
      continue;
    auto delay = helper::extractDelayFromDict(resultAttrs[i]);
    if (delay != 0)
      printer << " delay " << delay;
  }
  printer << ")>";
}

static void printBusType(BusType busTy, DialectAsmPrinter &printer) {
  printer << busTy.getKeyword() << "<" << busTy.getElementType() << ">";
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
  if (BusType busTy = type.dyn_cast<BusType>()) {
    printBusType(busTy, printer);
    return;
  }
}
