//=========- HIR.cpp - Parser & Printer for Ops ---------------------------===//
//
// This file implements parsers and printers for ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace hir;
using namespace llvm;

// Helper Methods.

static IntegerAttr getIntegerAttr(OpAsmParser &parser, int width, int value) {
  return IntegerAttr::get(
      IntegerType::get(width, parser.getBuilder().getContext()),
      APInt(width, value));
}
static Type getIntegerType(OpAsmParser &parser, int bitwidth) {
  return IntegerType::get(bitwidth, parser.getBuilder().getContext());
}
static ConstType getConstIntType(OpAsmParser &parser) {
  return ConstType::get(parser.getBuilder().getContext());
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

/// parse a comma separated list of operands.
static ParseResult
parseOperands(OpAsmParser &parser,
              SmallVectorImpl<OpAsmParser::OperandType> &operands) {
  // operands-list ::= firstOperand (, nextOperand)* .
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

/// parse an optional comma separated list of operands with paren as delimiter.
static ParseResult
parseOptionalOperands(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::OperandType> &operands) {
  // ::= `(` `)` or `(` operands-list `)`.
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
  if (parser.parseLParen())
    return failure();
  if (parser.parseOptionalRParen())
    if (parseTypes(parser, types))
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
/// CallOp
/// Syntax:
/// $callee `(` $operands `)` `at` $tstart (`offset` $offset^ )?
///   `:` ($operand (delay $delay^)?) `->` ($results)

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  OpAsmParser::OperandType tstart;
  OpAsmParser::OperandType offset;
  FlatSymbolRefAttr calleeAttr;
  if (parser.parseAttribute(calleeAttr,
                            parser.getBuilder().getType<::mlir::NoneType>(),
                            "callee", result.attributes))
    return failure();

  if (parser.parseLParen())
    return failure();

  if (parser.parseOperandList(operands))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("at"))
    return failure();

  if (parser.parseOperand(tstart))
    return failure();

  bool offset_present = false;
  if (succeeded(parser.parseOptionalKeyword("offset"))) {
    offset_present = true;
    if (parser.parseOperand(offset))
      return failure();
  }
  if (parser.parseColon())
    return failure();

  // parse arg types and delays.
  SmallVector<Attribute, 4> input_delays;
  SmallVector<Attribute, 4> output_delays;

  if (parser.parseLParen())
    return failure();
  int count = 0;
  auto argLoc = parser.getCurrentLocation();
  while (1) {
    Type type;
    IntegerAttr delayAttr;
    if (parser.parseType(type))
      return failure();
    argTypes.push_back(type);
    NamedAttrList tempAttrs;
    if (type.isa<IntegerType>() &&
        succeeded(parser.parseOptionalKeyword("delay"))) {
      if (parser.parseAttribute(delayAttr, getIntegerType(parser, 64), "delay",
                                tempAttrs))
        return failure();
      input_delays.push_back(delayAttr);
    } else {
      // Default delay is 0.
      input_delays.push_back(getIntegerAttr(parser, 64, 0));
    }
    count++;
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseRParen())
    return failure();

  ArrayAttr argDelayAttrs = parser.getBuilder().getArrayAttr(input_delays);
  result.attributes.push_back(
      parser.getBuilder().getNamedAttr("input_delays", argDelayAttrs));

  // resolve operands.
  parser.resolveOperands(operands, argTypes, argLoc, result.operands);
  parser.resolveOperand(tstart, getTimeType(parser), result.operands);
  if (offset_present)
    parser.resolveOperand(offset, getConstIntType(parser), result.operands);

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(operands.size()), 1,
                           static_cast<int32_t>(offset_present ? 1 : 0)}));

  // Return if no output args.
  if (parser.parseOptionalArrow()) {
    // blank output_delays attr
    ArrayAttr resultDelayAttrs =
        parser.getBuilder().getArrayAttr(output_delays);
    result.attributes.push_back(
        parser.getBuilder().getNamedAttr("output_delays", resultDelayAttrs));

    auto fnType = parser.getBuilder().getFunctionType(argTypes, resultTypes);
    result.addAttribute(impl::getTypeAttrName(), TypeAttr::get(fnType));
    return success();
  }

  if (parser.parseLParen())
    return failure();

  while (1) {
    Type type;
    IntegerAttr delayAttr;
    if (parser.parseType(type))
      return failure();
    resultTypes.push_back(type);
    NamedAttrList tempAttrs;
    if (type.isa<IntegerType>() &&
        succeeded(parser.parseOptionalKeyword("delay"))) {
      if (parser.parseAttribute(delayAttr, getIntegerType(parser, 64), "delay",
                                tempAttrs))
        return failure();
      output_delays.push_back(delayAttr);
    } else {
      // Default delay is 0.
      output_delays.push_back(getIntegerAttr(parser, 64, 0));
    }
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseRParen())
    return failure();

  ArrayAttr resultDelayAttrs = parser.getBuilder().getArrayAttr(output_delays);
  result.attributes.push_back(
      parser.getBuilder().getNamedAttr("output_delays", resultDelayAttrs));

  result.addTypes(resultTypes);
  return success();
}

static void printCallOp(OpAsmPrinter &printer, CallOp op) {
  auto input_delays = op.input_delays();
  auto output_delays = op.output_delays();
  auto operands = op.operands();
  auto resultTypes = op.getResultTypes();
  assert(input_delays.size() == operands.size());
  assert(output_delays.size() == resultTypes.size());
  printer << "hir.call @" << op.callee() << "(" << op.operands() << ") at "
          << op.tstart();
  if (op.offset())
    printer << " offset " << op.offset();
  printer << " : (";
  for (int i = 0; i < operands.size(); i++) {
    if (i > 0)
      printer << ", ";
    Type type = operands[i].getType();
    int delay = input_delays[i].dyn_cast<IntegerAttr>().getInt();
    assert(delay >= 0);
    printer << type;
    if (delay > 0)
      printer << " delay " << delay;
  }
  printer << ")";
  if (resultTypes.size() == 0)
    return;
  printer << " -> (";
  for (int i = 0; i < resultTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    Type type = resultTypes[i];
    int delay = output_delays[i].dyn_cast<IntegerAttr>().getInt();
    assert(delay >= 0);
    printer << type;
    if (delay > 0)
      printer << " delay " << delay;
  }
  printer << ")";
}

/// IfOp
static void printIfOp(OpAsmPrinter &printer, IfOp op) {

  printer << "hir.if (" << op.cond() << ") at " << op.tstart();
  printer.printRegion(op.if_region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}
static ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType cond;
  OpAsmParser::OperandType tstart;
  if (parser.parseLParen() || parser.parseOperand(cond) || parser.parseRParen())
    return failure();
  if (parser.parseKeyword("at") || parser.parseOperand(tstart))
    return failure();

  if (parser.resolveOperand(cond, getIntegerType(parser, 1), result.operands) ||
      parser.resolveOperand(tstart, getTimeType(parser), result.operands))
    return failure();

  Region *if_body = result.addRegion();
  if (parser.parseRegion(*if_body, {}, {}))
    return failure();
  auto &builder = parser.getBuilder();
  IfOp::ensureTerminator(*if_body, builder, result.location);
  return success();
}

/// ForOp.
/// Example:
/// hir.for %i = %l to %u step %s iter_time(%ti = %t){...}
static void printForOp(OpAsmPrinter &printer, ForOp op) {
  printer << "hir.for"
          << " " << op.getInductionVar() << " : "
          << op.getInductionVar().getType() << " = " << op.lb() << " : "
          << op.lb().getType() << " to " << op.ub() << " : "
          << op.ub().getType() << " step " << op.step() << " : "
          << op.step().getType() << " iter_time( " << op.getIterTimeVar()
          << " = " << op.tstart() << " offset" << op.offset() << ")";

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
  Type offsetType = getConstIntType(parser);
  Type regionRawOperandTypes[2];
  ArrayRef<Type> regionOperandTypes(regionRawOperandTypes);
  regionRawOperandTypes[1] = timeTy;

  OpAsmParser::OperandType lbRawOperand;
  OpAsmParser::OperandType ubRawOperand;
  OpAsmParser::OperandType offsetRawOperand;
  OpAsmParser::OperandType stepRawOperand;
  OpAsmParser::OperandType tstartRawOperand;
  OpAsmParser::OperandType regionRawOperands[2];

  ArrayRef<OpAsmParser::OperandType> regionOperands(regionRawOperands);
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(regionRawOperands[0]) ||
      parser.parseColonType(regionRawOperandTypes[0]) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  if (parser.parseOperand(lbRawOperand) || parser.parseColonType(lbRawType) ||
      parser.parseKeyword("to") || parser.parseOperand(ubRawOperand) ||
      parser.parseColonType(ubRawType) || parser.parseKeyword("step") ||
      parser.parseOperand(stepRawOperand) || parser.parseColonType(stepRawType))
    return failure();

  // Parse iter time.
  if (parser.parseKeyword("iter_time") || parser.parseLParen() ||
      parser.parseRegionArgument(regionRawOperands[1]) || parser.parseEqual() ||
      parser.parseOperand(tstartRawOperand) || parser.parseKeyword("offset") ||
      parser.parseOperand(offsetRawOperand) || parser.parseRParen())
    return failure();

  if (parser.resolveOperand(lbRawOperand, lbRawType, result.operands) ||
      parser.resolveOperand(ubRawOperand, ubRawType, result.operands) ||
      parser.resolveOperand(stepRawOperand, stepRawType, result.operands) ||
      parser.resolveOperand(tstartRawOperand, tstartRawType, result.operands) ||
      parser.resolveOperand(offsetRawOperand, offsetType, result.operands))
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

/// UnrollForOp.
/// Example:
/// hir.unroll_for %i = 0 to 100 step 3 iter_time(%ti = %t){...}
static void printUnrollForOp(OpAsmPrinter &printer, UnrollForOp op) {
  printer << "hir.unroll_for"
          << " " << op.getInductionVar() << " = " << op.lb() << " to "
          << op.ub() << " step " << op.step() << " iter_time( "
          << op.getIterTimeVar() << " = " << op.tstart() << " ) ";

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

static ParseResult parseUnrollForOp(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();
  Type timeTypeVar = getTimeType(parser);
  Type tstartRawType = timeTypeVar;
  Type regionRawOperandTypes[2];
  ArrayRef<Type> regionOperandTypes(regionRawOperandTypes);
  regionRawOperandTypes[0] = getConstIntType(parser);
  regionRawOperandTypes[1] = timeTypeVar;

  IntegerAttr lbAttr;
  IntegerAttr ubAttr;
  IntegerAttr stepAttr;
  OpAsmParser::OperandType tstartRawOperand;
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

  // Parse the type of induction variable.
  if (parser.parseRParen())
    return failure();

  // resolve operand.
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

/// DefOp
/// Example:
/// hir.def @foo at %t (%x :!hir.int, %y:!hir.int) ->(!hir.int){}

static ParseResult parseDefSignature(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &entryArgs,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<NamedAttrList> &argAttrs,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<NamedAttrList> &resultAttrs, OperationState &result) {

  SmallVector<Attribute, 4> input_delays;
  SmallVector<Attribute, 4> output_delays;
  // parse operand args
  if (parser.parseLParen())
    return failure();

  while (1) {
    // Parse operand and type
    OpAsmParser::OperandType operand;
    Type operandType;
    if (parser.parseOperand(operand) || parser.parseColon() ||
        parser.parseType(operandType))
      return failure();
    entryArgs.push_back(operand);
    argTypes.push_back(operandType);

    // Parse argAttr
    if (operandType.isa<IntegerType>() &&
        !parser.parseOptionalKeyword("delay")) {
      NamedAttrList tempAttrs;
      IntegerAttr delayAttr;
      if (parser.parseAttribute(delayAttr, getIntegerType(parser, 64), "delay",
                                tempAttrs))
        return failure();
      input_delays.push_back(delayAttr);
    } else {
      // Default delay is 0.
      input_delays.push_back(getIntegerAttr(parser, 64, 0));
    }

    NamedAttrList blankAttrs;
    argAttrs.push_back(blankAttrs);

    if (parser.parseOptionalComma())
      break;
  }
  if (parser.parseRParen())
    return failure();

  ArrayAttr argDelayAttrs = parser.getBuilder().getArrayAttr(input_delays);
  result.attributes.push_back(
      parser.getBuilder().getNamedAttr("input_delays", argDelayAttrs));

  // Return if no output args.
  if (parser.parseOptionalArrow()) {
    // blank output_delays attr
    ArrayAttr resultDelayAttrs =
        parser.getBuilder().getArrayAttr(output_delays);
    result.attributes.push_back(
        parser.getBuilder().getNamedAttr("output_delays", resultDelayAttrs));
    return success();
  }

  // parse result args
  if (parser.parseLParen())
    return failure();

  while (1) {
    // Parse result type
    Type resultType;
    if (parser.parseType(resultType))
      return failure();
    resultTypes.push_back(resultType);

    // Parse delayAttr
    if (resultType.isa<IntegerType>() &&
        !parser.parseOptionalKeyword("delay")) {
      IntegerAttr delayAttr;
      NamedAttrList tempAttrs;
      if (parser.parseAttribute(delayAttr, getIntegerType(parser, 64), "delay",
                                tempAttrs))
        return failure();
      output_delays.push_back(delayAttr);
    } else {
      // Default delay is 0.
      output_delays.push_back(getIntegerAttr(parser, 64, 0));
    }
    NamedAttrList blankAttrs;
    resultAttrs.push_back(blankAttrs);
    if (parser.parseOptionalComma())
      break;
  }
  if (parser.parseRParen())
    return failure();

  ArrayAttr resultDelayAttrs = parser.getBuilder().getArrayAttr(output_delays);
  result.attributes.push_back(
      parser.getBuilder().getNamedAttr("output_delays", resultDelayAttrs));
  return success();
}

// Parse method.
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
  // Parse tstart.
  if (parser.parseKeyword("at") || parser.parseRegionArgument(tstart))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (parseDefSignature(parser, entryArgs, argTypes, argAttrs, resultTypes,
                        resultAttrs, result))
    return failure();

  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(impl::getTypeAttrName(), TypeAttr::get(fnType));

  // If additional attributes are present, parse them.
  // if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
  //  return failure();

  // Add the attributes to the function arguments.
  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());
  impl::addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  entryArgs.push_back(tstart);
  argTypes.push_back(getTimeType(parser));
  return parser.parseOptionalRegion(*body, entryArgs, argTypes);
}

static void printDefSignature(OpAsmPrinter &printer, DefOp op) {
  auto fnType = op.getType();
  Region &body = op.getOperation()->getRegion(0);
  auto argTypes = fnType.getInputs();
  auto resTypes = fnType.getResults();
  ArrayAttr input_delays = op.input_delays();
  ArrayAttr output_delays = op.output_delays();
  printer << "(";
  bool firstArg = true;
  for (int i = 0; i < argTypes.size(); i++) {
    if (!firstArg)
      printer << ", ";
    firstArg = false;
    Type type = argTypes[i];
    auto arg = body.front().getArgument(i);
    int delay = input_delays[i].dyn_cast<IntegerAttr>().getInt();
    assert(delay >= 0);
    printer << arg << " : " << type;
    if (delay > 0)
      printer << " delay " << delay;
  }
  printer << ")";
  if (resTypes.size() == 0)
    return;

  printer << " -> (";
  bool firstRes = true;
  for (int i = 0; i < resTypes.size(); i++) {
    if (!firstRes)
      printer << ",";
    firstRes = false;
    Type type = resTypes[i];
    int delay = output_delays[i].dyn_cast<IntegerAttr>().getInt();
    assert(delay >= 0);
    printer << type;
    if (delay > 0)
      printer << " delay " << delay;
  }
  printer << ")";
}

static void printDefOp(OpAsmPrinter &printer, DefOp op) {
  // Print function name, signature, and control.
  printer << op.getOperationName() << " ";
  printer.printSymbolName(op.sym_name());
  Region &body = op.getOperation()->getRegion(0);
  printer << " at "
          << body.front().getArgument(body.front().getNumArguments() - 1)
          << " ";

  printDefSignature(printer, op);

  // Print the body if this is not an external function.
  if (!body.empty())
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
}

// CallableOpInterface.
Region *DefOp::getCallableRegion() { return isExternal() ? nullptr : &body(); }

// CallableOpInterface.
ArrayRef<Type> DefOp::getCallableResults() { return getType().getResults(); }

LogicalResult DefOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

/// required for functionlike trait
LogicalResult DefOp::verifyBody() { return success(); }
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.cpp.inc"
