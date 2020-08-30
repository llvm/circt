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
  addTypes<TimeType,ConstType, MemrefType, WireType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/HIR.cpp.inc"
      >();
}

static Type parseConstType(DialectAsmParser &parser, MLIRContext *context ){
  Type elementType;
  if(parser.parseLess() || parser.parseType(elementType) || parser.parseGreater())
    return Type();

  //TODO:Construct the ConstType with type.
  auto constTy =  ConstType::get(context);
  return constTy;
}

static Type parseMemrefType(DialectAsmParser &parser, MLIRContext *context ){
  llvm::SmallVector<int,4> dims;
  llvm::SmallVector<int,4> dist_dims;
  Type elementType;

  //TODO: Construct MemrefType with dims, elementType, dist_dims and permission(r,w,rw)
  auto memrefTy =  MemrefType::get(context);

  if(parser.parseLess())
    return  Type();

  int dim;
  if(parser.parseInteger(dim))
    return  Type();
  dims.push_back(dim); 
  if(parser.parseOptionalStar())
    return parser.emitError(parser.getCurrentLocation(), "expected '*' followed by type"),Type();
  while(true){
    auto optionalParseRes = parser.parseOptionalInteger(dim);
    if(!optionalParseRes.hasValue())
      break;
    if(optionalParseRes.getValue())
      return parser.emitError(parser.getCurrentLocation(), "expected integer literal or type"), Type();
    dims.push_back(dim); 
    if(parser.parseOptionalStar())
      return parser.emitError(parser.getCurrentLocation(), "expected '*' followed by type"),Type();
  }
  if(parser.parseType(elementType))
    return Type();
  if(!parser.parseOptionalGreater())
    return memrefTy; 
  if(parser.parseComma())
    return Type();
  if(!parser.parseOptionalKeyword("dist_dims")){
    if(parser.parseEqual() || parser.parseLSquare())
      return Type();
    int dim;
    if(parser.parseInteger(dim))
      return Type();
    dist_dims.push_back(dim);
    while(!parser.parseOptionalComma()){
      if(parser.parseInteger(dim))
        return  Type();
      dist_dims.push_back(dim);
    }
    if(parser.parseOptionalRSquare())
      return parser.emitError(parser.getCurrentLocation(), "expected ']' or ','"), Type();
  }
  //TODO: Add r,w,rw keywords
  if(parser.parseOptionalGreater())
    return parser.emitError(parser.getCurrentLocation(), "expected '>'"), Type(); 
  return memrefTy;
}

// Types
Type HIRDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  if (parser.parseKeyword(&typeKeyword))
    return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();

  if (typeKeyword == TimeType::getKeyword())
    return TimeType::get(getContext());

  if (typeKeyword == MemrefType::getKeyword())
    return parseMemrefType(parser,getContext());//MemrefType::get(getContext());

  if (typeKeyword == WireType::getKeyword())
    return WireType::get(getContext());

  if (typeKeyword == ConstType::getKeyword())
    return parseConstType(parser,getContext());

  return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
}

void HIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (TimeType timeTy = type.dyn_cast<TimeType>()) {
    printer << timeTy.getKeyword();
    return;
  }
  if (MemrefType memrefTy =
          type.dyn_cast<MemrefType>()) {
    printer << memrefTy.getKeyword();
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
