#include "circt/Dialect/HIR/IR/HIROpSyntax.h"
#include "circt/Dialect/HIR/IR/helper.h"
#define min(x, y) x > y ? x : y

// parser and printer for Time and offset.

ParseResult parseTimeAndOffset(
    mlir::OpAsmParser &parser,
    llvm::Optional<mlir::OpAsmParser::OperandType> &tstartOptional,
    IntegerAttr &offsetOptional) {
  OpAsmParser::OperandType tstart;

  // early exit if no schedule provided.
  if (succeeded(parser.parseOptionalQuestion()))
    return success();

  // parse tstart.
  if (parser.parseOperand(tstart))
    return failure();
  tstartOptional = tstart;

  // set offset = 0 and early exit if no offsets.
  if (parser.parseOptionalPlus()) {
    offsetOptional = parser.getBuilder().getI64IntegerAttr(0);
    return success();
  }

  // Parse offset
  int64_t offset;
  if (parser.parseInteger(offset))
    return failure();
  offsetOptional = parser.getBuilder().getI64IntegerAttr(offset);

  return success();
}

ParseResult parseTimeAndOffset(mlir::OpAsmParser &parser,
                               mlir::OpAsmParser::OperandType &tstart,
                               IntegerAttr &delay) {

  // parse tstart.
  if (parser.parseOperand(tstart))
    return failure();

  // early exit if no offsets.
  if (parser.parseOptionalPlus()) {
    delay = parser.getBuilder().getI64IntegerAttr(0);
    return success();
  }

  // Parse offset
  int64_t offset;
  if (parser.parseInteger(offset))
    return failure();
  delay = parser.getBuilder().getI64IntegerAttr(offset);

  return success();
}

void printTimeAndOffset(OpAsmPrinter &printer, Operation *op, Value tstart,
                        IntegerAttr offset) {
  if (!tstart) {
    printer << "?";
    return;
  }
  printer << tstart;

  if (!offset || (offset.getInt() == 0))
    return;
  printer << " + " << offset.getInt();
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
      tempConstAddrs.push_back(helper::getI64IntegerAttr(context, val));
    } else if (!parser.parseOperand(var)) {
      varAddrs.push_back(var);
      tempConstAddrs.push_back(helper::getI64IntegerAttr(
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
      idxTypes.push_back(
          IntegerType::get(context, min(1, helper::clog2(shape[i]))));
    }
  }

  elementTy = memTy.getElementType();
  return success();
}

void printMemrefAndElementType(OpAsmPrinter &printer, Operation *,
                               Type memrefTy, TypeRange, Type) {
  printer << memrefTy;
}

ParseResult parseTypeAndDelayList(mlir::OpAsmParser &parser,
                                  SmallVectorImpl<Type> &typeList,
                                  ArrayAttr &delayList) {
  SmallVector<Attribute> delayAttrArray;
  do {
    Type ty;
    int delay;
    if (parser.parseType(ty))
      return failure();
    typeList.push_back(ty);
    if (succeeded(parser.parseOptionalKeyword("delay"))) {
      if (parser.parseInteger(delay))
        return failure();
      delayAttrArray.push_back(parser.getBuilder().getI64IntegerAttr(delay));
    } else if (helper::isBuiltinSizedType(ty)) {
      delayAttrArray.push_back(parser.getBuilder().getI64IntegerAttr(0));
    } else {
      delayAttrArray.push_back(Attribute());
    }
  } while (succeeded(parser.parseOptionalComma()));
  delayList = parser.getBuilder().getArrayAttr(delayAttrArray);
  return success();
}

void printTypeAndDelayList(mlir::OpAsmPrinter &printer, TypeRange typeList,
                           ArrayAttr delayList) {
  for (uint64_t i = 0; i < typeList.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << typeList[i];
    if (!delayList[i])
      continue;
    int delay = delayList[i].dyn_cast<IntegerAttr>().getInt();
    if (delay != 0)
      printer << " delay " << delay;
  }
}

ParseResult parseBinOpOperandsAndResultType(mlir::OpAsmParser &parser,
                                            Type &resultTy, Type &op1Ty,
                                            Type &op2Ty) {
  if (parser.parseType(resultTy))
    return failure();
  op1Ty = resultTy;
  op2Ty = resultTy;
  return success();
}

void printBinOpOperandsAndResultType(mlir::OpAsmPrinter &printer, Operation *,
                                     Type resultTy, Type, Type) {
  printer << resultTy;
}

ParseResult parseCopyType(mlir::OpAsmParser &parser, Type &destTy, Type srcTy) {
  destTy = srcTy;
  return success();
}
void printCopyType(mlir::OpAsmPrinter &, Operation *, Type, Type) {}

ParseResult parseWithSSANames(mlir::OpAsmParser &parser,
                              mlir::NamedAttrList &attrDict) {

  if (parser.parseOptionalAttrDict(attrDict))
    return failure();

  // If the attribute dictionary contains no 'names' attribute, infer it from
  // the SSA name (if specified).
  bool hadNames = llvm::any_of(
      attrDict, [](NamedAttribute attr) { return attr.first == "names"; });

  // If there was no name specified, check to see if there was a useful name
  // specified in the asm file.
  if (hadNames || parser.getNumResults() == 0)
    return success();

  SmallVector<StringRef, 4> names;
  auto *context = parser.getBuilder().getContext();

  for (size_t i = 0, e = parser.getNumResults(); i != e; ++i) {
    auto resultName = parser.getResultName(i);
    StringRef nameStr;
    if (!resultName.first.empty() && !isdigit(resultName.first[0]))
      nameStr = resultName.first;

    names.push_back(nameStr);
  }

  auto namesAttr = parser.getBuilder().getStrArrayAttr(names);
  attrDict.push_back({Identifier::get("names", context), namesAttr});
  return success();
}

void printWithSSANames(mlir::OpAsmPrinter &printer, Operation *op,
                       mlir::DictionaryAttr attrDict) {

  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  auto names = op->getAttrOfType<ArrayAttr>("names");
  bool namesDisagree;
  if (names)
    namesDisagree = names.size() != op->getNumResults();
  else
    namesDisagree = true;

  SmallString<32> resultNameStr;
  for (size_t i = 0, e = op->getNumResults(); i != e && !namesDisagree; ++i) {
    resultNameStr.clear();
    llvm::raw_svector_ostream tmpStream(resultNameStr);
    printer.printOperand(op->getResult(i), tmpStream);

    auto expectedName = names[i].dyn_cast<StringAttr>();
    if (!expectedName ||
        tmpStream.str().drop_front() != expectedName.getValue()) {
      if (!expectedName.getValue().empty())
        namesDisagree = true;
    }
  }
  SmallVector<StringRef, 10> elidedAttrs = {
      "offset",         "delay",        "ports",
      "port",           "result_attrs", "callee",
      "funcTy",         "portNums",     "operand_segment_sizes",
      "index",          "mem_type",     "argNames",
      "instance_name",  "value",        "mem_kind",
      "iter_arg_delays"};
  if (!namesDisagree)
    elidedAttrs.push_back("names");
  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}
