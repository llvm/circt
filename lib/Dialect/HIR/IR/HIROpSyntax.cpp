#include "HIROpSyntax.h"
#include "circt/Dialect/HIR/IR/helper.h"

// parser and printer for Time and offset.
ParseResult parseTimeAndOffset(
    mlir::OpAsmParser &parser,
    llvm::Optional<mlir::OpAsmParser::OperandType> &tstartOptional,
    llvm::Optional<mlir::OpAsmParser::OperandType> &offsetOptional) {
  OpAsmParser::OperandType tstart;
  OpAsmParser::OperandType offset;

  if (parser.parseKeyword("at"))
    return failure();

  // early exit if no schedule provided.
  if (parser.parseOptionalQuestion())
    return success();

  // parse tstart.
  if (parser.parseOperand(tstart))
    return failure();
  tstartOptional = tstart;

  // early exit if no offsets.
  if (parser.parseOptionalPlus())
    return success();

  // Parse offset
  if (parser.parseOperand(offset))
    return failure();
  offsetOptional = offset;

  return success();
}

void printTimeAndOffset(OpAsmPrinter &printer, Operation *op, Value tstart,
                        Value offset) {
  printer << "at ";
  if (!tstart) {
    printer << "?";
    return;
  }
  printer << tstart;

  if (!offset)
    return;
  printer << " + " << offset;
}

// parser and printer for array address indices.
ParseResult
parseOptionalArrayAccess(OpAsmParser &parser,
                         SmallVectorImpl<OpAsmParser::OperandType> &varAddrs,
                         ArrayAttr &constAddrs) {
  // If constAddr[i] is -ve then its an operand and the operand is at
  // varAddrs[-constAddr[i]-1], i.e. constAddr[i]>=0 is const and <0 is operand.
  llvm::SmallVector<Attribute, 4> tempConstAddrs;
  // early exit.
  if (parser.parseOptionalLSquare())
    return success();
  do {
    int val;
    auto *context = parser.getBuilder().getContext();
    OpAsmParser::OperandType var;
    mlir::OptionalParseResult result = parser.parseOptionalInteger(val);
    if (result.hasValue() && !result.getValue()) {
      tempConstAddrs.push_back(helper::getIntegerAttr(context, val));
    } else if (!parser.parseOperand(var)) {
      varAddrs.push_back(var);
      tempConstAddrs.push_back(helper::getIntegerAttr(
          parser.getBuilder().getContext(), -varAddrs.size()));
    } else
      return failure();
  } while (!parser.parseOptionalComma());
  constAddrs = ArrayAttr::get(parser.getBuilder().getContext(), tempConstAddrs);
  if (parser.parseRSquare())
    return failure();
  return success();
}

void printOptionalArrayAccess(OpAsmPrinter &printer, Operation *op,
                              OperandRange varAddrs, ArrayAttr constAddrs) {
  if (constAddrs.size() == 0)
    return;
  printer << "[";
  for (size_t i = 0; i < constAddrs.size(); i++) {
    int idx = constAddrs[i].dyn_cast<IntegerAttr>().getInt();
    if (i > 0)
      printer << ", ";
    if (idx >= 0)
      printer << idx;
    else
      printer << varAddrs[-idx - 1];
  }
  printer << "]";
}

// parser and printer for array address types.
ParseResult parseOptionalArrayAccessTypes(mlir::OpAsmParser &parser,
                                          ArrayAttr &constAddrs,
                                          SmallVectorImpl<Type> &varAddrTypes) {
  // Early exit if no address types.
  if (parser.parseOptionalLSquare())
    return success();
  // Parse the types list.
  int i = 0;
  do {
    int idx = constAddrs[i].dyn_cast<IntegerAttr>().getInt();
    i++;
    Type t;
    if (!parser.parseOptionalKeyword("const"))
      t = IndexType::get(parser.getBuilder().getContext());
    else if (parser.parseType(t))
      return failure();

    if (idx < 0)
      varAddrTypes.push_back(t);
    else if (!t.isa<IndexType>())
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected index type");
  } while (!parser.parseOptionalComma());

  // Finish parsing.
  if (parser.parseRSquare())
    return failure();

  return success();
}

void printOptionalArrayAccessTypes(OpAsmPrinter &printer, Operation *op,
                                   ArrayAttr constAddrs,
                                   TypeRange varAddrTypes) {
  if (constAddrs.size() == 0)
    return;
  printer << "[";
  for (size_t i = 0; i < constAddrs.size(); i++) {
    if (i > 0)
      printer << ", ";
    int idx = constAddrs[i].dyn_cast<IntegerAttr>().getInt();
    if (idx >= 0)
      printer << "const";
    else if (varAddrTypes[-idx - 1].isa<IndexType>())
      printer << "const";
    else
      printer << varAddrTypes[-idx - 1];
  }
  printer << "]";
}

ParseResult parseMemrefAndElementType(OpAsmParser &parser, Type &memrefTy,
                                      SmallVectorImpl<Type> &idxTypes,
                                      Type &elementTy) {
  auto memTyLoc = parser.getCurrentLocation();
  if (parser.parseType(memrefTy))
    return failure();
  auto memTy = memrefTy.dyn_cast<hir::MemrefType>();
  if (!memTy)
    return parser.emitError(memTyLoc, "Expected hir.memref type!");
  auto builder = parser.getBuilder();
  auto *context = builder.getContext();
  auto shape = memTy.getShape();
  auto dimKinds = memTy.getDimKinds();

  for (int i = 0; i < (int)dimKinds.size(); i++) {
    if (dimKinds[i] == hir::BANK)
      idxTypes.push_back(IndexType::get(context));
    else {
      idxTypes.push_back(IntegerType::get(context, helper::clog2(shape[i])));
    }
  }

  elementTy = memTy.getElementType();
  return success();
}

void printMemrefAndElementType(OpAsmPrinter &printer, Operation *,
                               Type memrefTy, TypeRange, Type) {
  printer << memrefTy;
}
