//=========- HIR.cpp - Parser & Printer for Ops ---------------------------===//
//
// This file implements parsers and printers for ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

#include "circt/Dialect/HIR/helper.h"
using namespace mlir;
using namespace hir;
using namespace llvm;

/// CallOp
/// Syntax:
/// $callee `(` $operands `)` `at` $tstart (`offset` $offset^ )?
///   `:` ($operand (delay $delay^)?) `->` ($results)

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {

  OpAsmParser::OperandType calleeVar;
  Type calleeTy;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  OpAsmParser::OperandType tstart;
  OpAsmParser::OperandType offset;
  FlatSymbolRefAttr calleeAttr;
  bool calleeIsSymbol = false;

  OptionalParseResult res = parser.parseOptionalOperand(calleeVar);
  if ((!res.hasValue()) || res.getValue()) {
    if (parser.parseAttribute(calleeAttr,
                              parser.getBuilder().getType<::mlir::NoneType>(),
                              "callee", result.attributes))
      return failure();
    calleeIsSymbol = true;
  }

  if (parser.parseLParen())
    return failure();

  llvm::SMLoc argLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operands))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("at"))
    return failure();

  if (parser.parseOperand(tstart))
    return failure();

  bool isOffsetPresent = false;
  if (succeeded(parser.parseOptionalPlus())) {
    isOffsetPresent = true;
    if (parser.parseOperand(offset))
      return failure();
  }

  // parse arg types and delays.
  if (parser.parseColon())
    return failure();

  auto locCalleeTy = parser.getCurrentLocation();
  if (parser.parseType(calleeTy))
    return failure();

  // resolve operands.
  hir::FuncType funcTy = calleeTy.dyn_cast<hir::FuncType>();
  if (!funcTy)
    return parser.emitError(locCalleeTy, "expected !hir.func type!");

  if (!calleeIsSymbol)
    parser.resolveOperand(calleeVar, calleeTy, result.operands);
  parser.resolveOperands(operands, funcTy.getFunctionType().getInputs(), argLoc,
                         result.operands);
  parser.resolveOperand(tstart,
                        helper::getTimeType(parser.getBuilder().getContext()),
                        result.operands);
  if (isOffsetPresent)
    parser.resolveOperand(
        offset, helper::getConstIntType(parser.getBuilder().getContext()),
        result.operands);

  // add attributes.
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(calleeIsSymbol ? 0 : 1),
                           static_cast<int32_t>(operands.size()), 1,
                           static_cast<int32_t>(isOffsetPresent ? 1 : 0)}));

  result.attributes.push_back(
      parser.getBuilder().getNamedAttr("inputDelays", funcTy.getInputDelays()));

  result.attributes.push_back(parser.getBuilder().getNamedAttr(
      "outputDelays", funcTy.getOutputDelays()));

  result.addAttribute("funcTy", TypeAttr::get(calleeTy));
  result.addTypes(funcTy.getFunctionType().getResults());

  return success();
}

static void printCallOp(OpAsmPrinter &printer, CallOp op) {
  if (op.callee().hasValue())
    printer << "hir.call @" << op.callee();
  else
    printer << "hir.call " << op.callee_var();
  printer << "(" << op.operands() << ") at " << op.tstart();
  if (op.offset())
    printer << " + " << op.offset();
  printer << " : ";
  printer << op.funcTy();
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

  if (parser.resolveOperand(
          cond, helper::getIntegerType(parser.getBuilder().getContext(), 1),
          result.operands) ||
      parser.resolveOperand(
          tstart, helper::getTimeType(parser.getBuilder().getContext()),
          result.operands))
    return failure();

  Region *ifBody = result.addRegion();
  if (parser.parseRegion(*ifBody, {}, {}))
    return failure();
  auto &builder = parser.getBuilder();
  IfOp::ensureTerminator(*ifBody, builder, result.location);
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
          << " = " << op.tstart() << " + " << op.offset() << ")";

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}
static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  Type timeTy = helper::getTimeType(parser.getBuilder().getContext());
  Type lbRawType;
  Type ubRawType;
  Type stepRawType;
  Type tstartRawType = timeTy;
  Type offsetType = helper::getConstIntType(parser.getBuilder().getContext());
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
      parser.parseOperand(tstartRawOperand) || parser.parsePlus() ||
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
  result.addTypes(helper::getTimeType(parser.getBuilder().getContext()));
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
  Type timeTypeVar = helper::getTimeType(parser.getBuilder().getContext());
  Type tstartRawType = timeTypeVar;
  Type regionRawOperandTypes[2];
  ArrayRef<Type> regionOperandTypes(regionRawOperandTypes);
  regionRawOperandTypes[0] =
      helper::getConstIntType(parser.getBuilder().getContext());
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

  if (helper::parseIntegerAttr(lbAttr, 32, "lb", parser, result) ||
      parser.parseKeyword("to") ||
      helper::parseIntegerAttr(ubAttr, 32, "ub", parser, result) ||
      parser.parseKeyword("step") ||
      helper::parseIntegerAttr(stepAttr, 32, "step", parser, result))
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
  result.addTypes(helper::getTimeType(parser.getBuilder().getContext()));
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

/// FuncOp
/// Example:
/// hir.def @foo at %t (%x :!hir.int, %y:!hir.int) ->(!hir.int){}

static ParseResult parseFuncSignature(
    OpAsmParser &parser, hir::FuncType &funcTy,
    SmallVectorImpl<OpAsmParser::OperandType> &entryArgs,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<NamedAttrList> &argAttrs,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<NamedAttrList> &resultAttrs, OperationState &result) {

  SmallVector<Attribute, 4> inputDelays;
  SmallVector<Attribute, 4> outputDelays;
  // parse operand args
  if (parser.parseLParen())
    return failure();
  if (parser.parseOptionalRParen()) {
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
        if (parser.parseAttribute(
                delayAttr,
                helper::getIntegerType(parser.getBuilder().getContext(), 64),
                "delay", tempAttrs))
          return failure();
        inputDelays.push_back(delayAttr);
      } else {
        // Default delay is 0.
        inputDelays.push_back(
            helper::getIntegerAttr(parser.getBuilder().getContext(), 64, 0));
      }
      NamedAttrList blankAttrs;
      argAttrs.push_back(blankAttrs);

      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseRParen())
      return failure();
  }

  ArrayAttr argDelayAttrs = parser.getBuilder().getArrayAttr(inputDelays);

  // Return if no output args.
  if (!parser.parseOptionalArrow()) {

    // parse result args
    if (parser.parseLParen())
      return failure();

    if (parser.parseOptionalRParen()) {
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
          if (parser.parseAttribute(
                  delayAttr,
                  helper::getIntegerType(parser.getBuilder().getContext(), 64),
                  "delay", tempAttrs))
            return failure();
          outputDelays.push_back(delayAttr);
        } else {
          // Default delay is 0.
          outputDelays.push_back(
              helper::getIntegerAttr(parser.getBuilder().getContext(), 64, 0));
        }
        NamedAttrList blankAttrs;
        resultAttrs.push_back(blankAttrs);
        if (parser.parseOptionalComma())
          break;
      }
      if (parser.parseRParen())
        return failure();
    }
  }

  ArrayAttr resultDelayAttrs = parser.getBuilder().getArrayAttr(outputDelays);
  auto functionTy = parser.getBuilder().getFunctionType(argTypes, resultTypes);
  funcTy = hir::FuncType::get(parser.getBuilder().getContext(), functionTy,
                              argDelayAttrs, resultDelayAttrs);
  return success();
}

// Parse method.
static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
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
  hir::FuncType funcTy;
  if (parseFuncSignature(parser, funcTy, entryArgs, argTypes, argAttrs,
                         resultTypes, resultAttrs, result))
    return failure();

  auto functionTy = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(impl::getTypeAttrName(), TypeAttr::get(functionTy));
  result.addAttribute("funcTy", TypeAttr::get(funcTy));

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
  argTypes.push_back(helper::getTimeType(parser.getBuilder().getContext()));
  auto r = parser.parseOptionalRegion(*body, entryArgs, argTypes);
  if (r.hasValue())
    return r.getValue();
  return success();
}

static void printFuncSignature(OpAsmPrinter &printer, hir::FuncOp op) {
  auto fnType = op.getType();
  Region &body = op.getOperation()->getRegion(0);
  auto argTypes = fnType.getInputs();
  auto resTypes = fnType.getResults();
  ArrayAttr inputDelays = op.funcTy().dyn_cast<FuncType>().getInputDelays();
  ArrayAttr outputDelays = op.funcTy().dyn_cast<FuncType>().getOutputDelays();

  printer << "(";
  bool firstArg = true;
  for (unsigned i = 0; i < argTypes.size(); i++) {
    if (!firstArg)
      printer << ", ";
    firstArg = false;
    Type type = argTypes[i];
    auto arg = body.front().getArgument(i);
    int delay = inputDelays[i].dyn_cast<IntegerAttr>().getInt();
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
  for (unsigned i = 0; i < resTypes.size(); i++) {
    if (!firstRes)
      printer << ",";
    firstRes = false;
    Type type = resTypes[i];
    int delay = outputDelays[i].dyn_cast<IntegerAttr>().getInt();
    assert(delay >= 0);
    printer << type;
    if (delay > 0)
      printer << " delay " << delay;
  }
  printer << ")";
}

static void printFuncOp(OpAsmPrinter &printer, hir::FuncOp op) {
  // Print function name, signature, and control.
  printer << op.getOperationName() << " ";
  printer.printSymbolName(op.sym_name());
  Region &body = op.getOperation()->getRegion(0);
  printer << " at "
          << body.front().getArgument(body.front().getNumArguments() - 1)
          << " ";

  printFuncSignature(printer, op);

  // Print the body if this is not an external function.
  if (!body.empty())
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
}

// CallableOpInterface.
Region *hir::FuncOp::getCallableRegion() {
  return isExternal() ? nullptr : &body();
}

// CallableOpInterface.
ArrayRef<Type> hir::FuncOp::getCallableResults() {
  return getType().getResults();
}

LogicalResult hir::FuncOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}
/// required for functionlike trait
LogicalResult hir::FuncOp::verifyBody() { return success(); }

#include "HIROpSyntax.h"
#include "HIROpVerifier.h"

namespace mlir {
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.cpp.inc"
} // namespace mlir
