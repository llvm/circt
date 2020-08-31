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
    : Dialect(getDialectNamespace(), context) {
  addTypes<TimeType, ConstType, MemrefType, WireType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/HIR.cpp.inc"
      >();
}

static Type parseConstType(DialectAsmParser &parser, MLIRContext *context) {
  Type elementType;
  if (parser.parseLess() || parser.parseType(elementType) ||
      parser.parseGreater())
    return Type();

  // TODO:Construct the ConstType with type.
  auto constTy = ConstType::get(context);
  return constTy;
}

static ParseResult parseMemrefPortKind(DialectAsmParser &parser,
                                       MemrefDetails::PortKind &port) {
  if (!parser.parseOptionalKeyword("rw"))
    port = MemrefDetails::rw;
  else if (!parser.parseOptionalKeyword("r"))
    port = MemrefDetails::r;
  else if (!parser.parseOptionalKeyword("w"))
    port = MemrefDetails::w;
  else
    return failure();
  return success();
}

static OptionalParseResult
parseOptionalMemrefPacking(DialectAsmParser &parser,
                           SmallVectorImpl<unsigned> &packing) {
  if (parser.parseOptionalKeyword("packing"))
    return OptionalParseResult(llvm::None);
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

static Type parseMemrefType(DialectAsmParser &parser, MLIRContext *context) {
  llvm::SmallVector<unsigned, 4> shape;
  Type elementType;
  hir::MemrefDetails::PortKind default_port = hir::MemrefDetails::rw;
  if (parser.parseLess())
    return Type();

  unsigned dim;
  if (parser.parseInteger(dim))
    return Type();
  shape.push_back(dim);
  if (parser.parseOptionalStar())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '*' followed by type"),
           Type();
  while (true) {
    auto optionalParseRes = parser.parseOptionalInteger(dim);
    if (!optionalParseRes.hasValue())
      break;
    if (optionalParseRes.getValue())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected integer literal or type"),
             Type();
    shape.push_back(dim);
    if (parser.parseOptionalStar())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected '*' followed by type"),
             Type();
  }
  if (parser.parseType(elementType))
    return Type();
  
  llvm::SmallVector<unsigned, 4> default_packing;
  // default packing is [0,1,2,3] for shape = [d3,d2,d1,d0] i.e. d0 is fastest
  // changing dim in linear index and no distributed dimensions.
  for (int i = 0; i < shape.size(); i++)
    default_packing.push_back(i);

  if (!parser.parseOptionalGreater())
    return MemrefType::get(context, shape, elementType, default_packing,
                           default_port);

  if (parser.parseComma())
    return Type();

  llvm::SmallVector<unsigned, 4> packing;
  OptionalParseResult hasPacking = parseOptionalMemrefPacking(parser, packing);

  if (hasPacking.hasValue()) {
    if (hasPacking.getValue())
      return Type();
    if (!parser.parseOptionalGreater())
      return MemrefType::get(context, shape, elementType, packing,
                             default_port);
    if (parser.parseComma())
      return Type();
  }

  hir::MemrefDetails::PortKind port;
  if (parseMemrefPortKind(parser, port) || parser.parseGreater())
    return Type();

  return MemrefType::get(context, shape, elementType, packing, port);
}

// Types
Type HIRDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  if (parser.parseKeyword(&typeKeyword))
    return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();

  if (typeKeyword == TimeType::getKeyword())
    return TimeType::get(getContext());

  if (typeKeyword == MemrefType::getKeyword())
    return parseMemrefType(parser,
                           getContext()); // MemrefType::get(getContext());

  if (typeKeyword == WireType::getKeyword())
    return WireType::get(getContext());

  if (typeKeyword == ConstType::getKeyword())
    return parseConstType(parser, getContext());

  return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
}

static void printMemrefType(MemrefType memrefTy, DialectAsmPrinter &printer){
    printer << memrefTy.getKeyword();
    printer << "<";
    auto shape = memrefTy.getShape();
    auto elementType = memrefTy.getElementType();
    auto packing = memrefTy.getPacking();
    auto port = memrefTy.getPort();
    for (auto dim:shape)
      printer << dim << "*";

    printer << elementType << ", packing = [";

    for (int i=0;i<packing.size();i++)
      printer << packing[i] << ((i==(packing.size()-1))?"" : ", ");
    printer << "], ";

    if(port == MemrefDetails::r){
      printer << "r";
    }
    else if (port == MemrefDetails::w){
      printer << "w";
    }
    else if (port == MemrefDetails::rw){
      printer << "rw";
    }
    else{
      printer << "unknown";
    }
    printer << ">";
}

void HIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (TimeType timeTy = type.dyn_cast<TimeType>()) {
    printer << timeTy.getKeyword();
    return;
  }
  if (MemrefType memrefTy = type.dyn_cast<MemrefType>()) {
    printMemrefType(memrefTy,printer);
    return;
  }
  if (WireType wireTy = type.dyn_cast<WireType>()) {
    printer << wireTy.getKeyword();
    return;
  }
  if (ConstType constTy = type.dyn_cast<ConstType>()) {
    printer << constTy.getKeyword();
    return;
  }
}
