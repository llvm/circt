#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace hir;
using namespace llvm;

// Helper Methods.

static IntegerAttr getIntegerAttr(OpAsmParser &parser, int width, int value) {
  IntegerAttr::get(IntegerType::get(32, parser.getBuilder().getContext()),
                   APInt(32, 0));
}
static Type getIntegerType(OpAsmParser &parser, int bitwidth) {
  return IntegerType::get(bitwidth, parser.getBuilder().getContext());
}
static ConstType getConstIntType(OpAsmParser &parser,int bitwidth) {
  return ConstType::get(parser.getBuilder().getContext(), getIntegerType(parser,bitwidth));
}

static Type getTimeType(OpAsmParser &parser) {
  return TimeType::get(parser.getBuilder().getContext());
}

static ParseResult parseIntegerAttr(IntegerAttr &value, int bitwidth,
                                    StringRef attrName, OpAsmParser &parser,
                                    OperationState &result) {

  return parser.parseAttribute(value, getIntegerType(parser, bitwidth),
                               attrName, result.attributes);
}

static ParseResult
parseOperands(OpAsmParser &parser,
              SmallVectorImpl<OpAsmParser::OperandType> &operands) {
  /// operands-list ::= firstOperand (, nextOperand)*
  OpAsmParser::OperandType firstOperand;
  if (parser.parseOperand(firstOperand))
    return failure();
  operands.push_back(firstOperand);
  while (!parser.parseComma()) {
    OpAsmParser::OperandType nextOperand;
    if (parser.parseOperand(nextOperand))
      return failure();
    operands.push_back(nextOperand);
  }
  return success();
}

static ParseResult
parseOptionalOperands(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::OperandType> &operands) {
  /// ::= `(` `)` or `(` operands-list `)`.
  if (parser.parseLParen())
    return failure();
  if (parser.parseOptionalRParen())
    if (parseOperands(parser, operands))
      return failure();
  return success();
}

static ParseResult parseTypes(OpAsmParser &parser,
                              SmallVectorImpl<Type> &types) {
  Type firstType;
  if (parser.parseType(firstType))
    return failure();
  types.push_back(firstType);
  while (!parser.parseComma()) {
    Type nextType;
    if (parser.parseType(nextType))
      return failure();
    types.push_back(nextType);
  }
  return success();
}

static ParseResult parseOptionalTypes(OpAsmParser &parser,
                                      SmallVectorImpl<Type> &types) {
  /// ::= `(` `)` or `(` operands-list `)`.
  if (parser.parseLParen())
    return failure();
  if (parser.parseOptionalRParen())
    if (parseTypes(parser, types))
      return failure();
  return success();
}

static ParseResult parseAndResolveTimeOperand(OpAsmParser &parser,
                                              OperationState &result) {

  OpAsmParser::OperandType tstart;
  OpAsmParser::OperandType offsetVar;
  if (parser.parseOperand(tstart))
    return failure();
  if (parser.resolveOperand(tstart, getTimeType(parser), result.operands))
    return failure();
  if (parser.parseOptionalKeyword("offset"))
    return success();

  auto offsetIsIdx = parser.parseOptionalOperand(offsetVar);
  if (offsetIsIdx.hasValue()) {
    if (parser.resolveOperand(offsetVar, getConstIntType(parser,32),
                              result.operands)) {
      return failure();
    } else
      return success();
  }
  IntegerAttr offset;
  if (parser.parseAttribute(offset, getIntegerType(parser, 32), "offset",
                            result.attributes))
    return failure();
}
static ParseResult parseTimeOperand(OpAsmParser &parser,
                                    OpAsmParser::OperandType &timeVar,
                                    StringRef attrName, NamedAttrList &attrs) {
  if (parser.parseOperand(timeVar))
    return failure();
  if (parser.parseOptionalKeyword("offset"))
    return success();
  IntegerAttr offset;
  if (parser.parseAttribute(offset, getIntegerType(parser, 32), attrName,
                            attrs))
    return failure();
  return success();
}

static ParseResult parseFunctionType(OpAsmParser &parser, int num_operands,
                                     SmallVectorImpl<Type> &operandTypes,
                                     SmallVectorImpl<Type> &resultTypes) {
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parseOptionalTypes(parser, operandTypes))
    return failure();
  if (operandTypes.size() != num_operands)
    return parser.emitError(typeLoc, "Wrong number of type parameters.");
  if (parser.parseArrow())
    return failure();
  if (parseOptionalTypes(parser, resultTypes))
    return failure();
  return success();
}

/* ForOp.
 * Example:
 * hir.for %i = %l to %u step %s iter_time(%ti = %t){...}
 * hir.for %i = %l to %u step %s iter_time(%ti = %t tstep %n){...}
 */

static void printForOp(OpAsmPrinter &printer, ForOp op) {
  printer << "hir.for"
          << " " << op.getInductionVar() << " : " <<op.getInductionVar().getType() 
          << " = " << op.lb() << " : " << op.lb().getType() << " to "
          << op.ub() << " : " << op.ub().getType() << " step " << op.step() 
          << " : " <<op.step().getType() << " iter_time( "
          << op.getIterTimeVar() << " = " << op.tstart();

  // print optional tstep.

  if (op.getNumOperands()==5)
    printer << " tstep " << op.tstep() << " : " << op.tstep().getType();
  printer << " ) ";

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  Type timeTy = getTimeType(parser);
  Type lbRawType;
  Type ubRawType;
  Type stepRawType;
  Type tstartRawType = timeTy;
  Type tstepRawType;
  Type regionRawOperandTypes[2];
  ArrayRef<Type> regionOperandTypes(regionRawOperandTypes);
  regionRawOperandTypes[1] = timeTy;

  OpAsmParser::OperandType lbRawOperand;
  OpAsmParser::OperandType ubRawOperand;
  OpAsmParser::OperandType stepRawOperand;
  OpAsmParser::OperandType tstartRawOperand;
  OpAsmParser::OperandType tstepRawOperand;
  OpAsmParser::OperandType regionRawOperands[2];

  ArrayRef<OpAsmParser::OperandType> regionOperands(regionRawOperands);
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(regionRawOperands[0]) || parser.parseColonType(regionRawOperandTypes[0])|| parser.parseEqual())
    return failure();

  // Parse loop bounds.
  if (parser.parseOperand(lbRawOperand) ||parser.parseColonType(lbRawType)|| parser.parseKeyword("to") ||
      parser.parseOperand(ubRawOperand) ||parser.parseColonType(ubRawType)|| parser.parseKeyword("step") ||
      parser.parseOperand(stepRawOperand)||parser.parseColonType(stepRawType))
    return failure();

  // Parse iter time.
  if (parser.parseKeyword("iter_time") || parser.parseLParen() ||
      parser.parseRegionArgument(regionRawOperands[1]) || parser.parseEqual() ||
      parser.parseOperand(tstartRawOperand))
    return failure();

  // Parse optional tstep.
  bool hasTstep = false;
  if (!parser.parseOptionalKeyword("tstep")) {
    if (parser.parseOperand(tstepRawOperand) || parser.parseColonType(tstepRawType))
      return failure();
    hasTstep = true;
  }
  if(parser.parseRParen())
    return failure();

  if (parser.resolveOperand(lbRawOperand, lbRawType, result.operands) ||
      parser.resolveOperand(ubRawOperand, ubRawType, result.operands) ||
      parser.resolveOperand(stepRawOperand, stepRawType, result.operands) ||
      parser.resolveOperand(tstartRawOperand, tstartRawType, result.operands))
    return failure();

  if (hasTstep)
    if (parser.resolveOperand(tstepRawOperand, tstepRawType, result.operands))
      return failure();

  // Parse the body region.

  Region *body = result.addRegion();

  if (parser.parseRegion(*body, regionOperands, regionOperandTypes))
    return failure();
  result.addTypes(getTimeType(parser));
  ForOp::ensureTerminator(*body, builder, result.location);
  return success();
}

Region &ForOp::getLoopBody() { return region(); }

bool ForOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult ForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
  return success();
}
/* UnrollForOp.
 * Example:
 * hir.unroll_for %i = 0 to 100 step 3 iter_time(%ti = %t){...}
 * hir.unroll_for %i = %l to %u step %s iter_time(%ti = %t tstep 3){...}
 */

static void printUnrollForOp(OpAsmPrinter &printer, UnrollForOp op) {
  printer << "hir.unroll_for"
          << " " << op.getInductionVar() << " = " << op.lb() << " to "
          << op.ub() << " step " << op.step() << " iter_time( "
          << op.getIterTimeVar() << " = " << op.tstart();

  // print optional tstep.
  
  if(op.tstep().hasValue())
    printer << " tstep " << op.tstep().getValue();
  printer << " ) ";

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

static ParseResult parseUnrollForOp(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();
  Type timeTypeVar = getTimeType(parser);
  Type tstartRawType = timeTypeVar;
  Type tstepRawType = getConstIntType(parser,32);
  Type regionRawOperandTypes[2];
  ArrayRef<Type> regionOperandTypes(regionRawOperandTypes);
  regionRawOperandTypes[0] = getConstIntType(parser,32);
  regionRawOperandTypes[1] = timeTypeVar;

  IntegerAttr lbAttr;
  IntegerAttr ubAttr;
  IntegerAttr stepAttr;
  OpAsmParser::OperandType tstartRawOperand;
  OpAsmParser::OperandType tstepRawOperand;
  OpAsmParser::OperandType regionRawOperands[2];

  ArrayRef<OpAsmParser::OperandType> regionOperands(regionRawOperands);
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(regionRawOperands[0]) || parser.parseEqual())
    return failure();

  // Parse loop bounds.

  if (parseIntegerAttr(lbAttr, 32, "lb", parser, result) ||
      parser.parseKeyword("to") ||
      parseIntegerAttr(ubAttr, 32, "ub", parser, result) ||
      parser.parseKeyword("step") ||
      parseIntegerAttr(stepAttr, 32, "step", parser, result))
    return failure();

  // Parse iter time.
  if (parser.parseKeyword("iter_time") || parser.parseLParen() ||
      parser.parseRegionArgument(regionRawOperands[1]) || parser.parseEqual() ||
      parser.parseOperand(tstartRawOperand))
    return failure();

  // Parse optional tstep.
  if (!parser.parseOptionalKeyword("tstep")) {
    if (parseIntegerAttr(stepAttr, 32, "tstep", parser, result))
      return failure();
  }

  // Parse the type of induction variable.
  if (parser.parseRParen())
    return failure();

  // resolve operand
  if (parser.resolveOperand(tstartRawOperand, tstartRawType, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();

  if (parser.parseRegion(*body, regionOperands, regionOperandTypes))
    return failure();
  ForOp::ensureTerminator(*body, builder, result.location);
  result.addTypes(getTimeType(parser));
  return success();
}

Region &UnrollForOp::getLoopBody() { return region(); }

bool UnrollForOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult UnrollForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
  return success();
}

/* DefOp
 * Example:
 * hir.def @foo at %t (%x :!hir.int, %y:!hir.int) ->(!hir.int){}
 */

// CallableOpInterface
Region *DefOp::getCallableRegion() {
  return isExternal() ? nullptr : &body();
}

// CallableOpInterface
ArrayRef<Type> DefOp::getCallableResults() {
  return getType().getResults();
}


static ParseResult parseDefOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  OpAsmParser::OperandType tstart;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse tstart
  if(parser.parseKeyword("at") || parser.parseRegionArgument(tstart))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (impl::parseFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                                   argTypes, argAttrs, isVariadic, resultTypes,
                                   resultAttrs))
    return failure();

  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(impl::getTypeAttrName(), TypeAttr::get(fnType));

  // If additional attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Add the attributes to the function arguments.
  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());
  impl::addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  entryArgs.push_back(tstart);
  argTypes.push_back(getTimeType(parser));
  return parser.parseOptionalRegion(
      *body, entryArgs, argTypes);
}

static void printDefOp(OpAsmPrinter &printer, DefOp op) {
  // Print function name, signature, and control.
  printer << "hir.def ";
  printer.printSymbolName(op.sym_name());
  Region &body = op.getOperation()->getRegion(0);
  printer << " at " << body.front().getArgument(body.front().getNumArguments()-1)<< " ";

  auto fnType = op.getType();
  impl::printFunctionSignature(printer, op, fnType.getInputs(),
                               /*isVariadic=*/false, fnType.getResults());
  impl::printFunctionAttributes(
      printer, op, fnType.getNumInputs(), fnType.getNumResults());

  // Print the body if this is not an external function.
  if (!body.empty())
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
}

LogicalResult DefOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

LogicalResult DefOp::verifyBody() {
  //TODO
  return success();
}
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.cpp.inc"
