//=========- HIR.cpp - Parser & Printer for Ops ---------------------------===//
//
// This file implements parsers and printers for ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
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

using namespace circt;
using namespace hir;
using namespace llvm;

//------------------------------------------------------------------------------
// Helper functions.
//------------------------------------------------------------------------------
ParseResult
parseOperandColonType(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::OperandType> &entryArgs,
                      SmallVectorImpl<Type> &argTypes) {

  OpAsmParser::OperandType operand;
  Type operandTy;
  if (parser.parseOperand(operand) || parser.parseColonType(operandTy))
    return failure();
  entryArgs.push_back(operand);
  argTypes.push_back(operandTy);
  return success();
}

ParseResult parseDelayAttr(OpAsmParser &parser,
                           SmallVectorImpl<DictionaryAttr> &attrsList) {
  NamedAttrList argAttrs;
  IntegerAttr delayAttr;
  auto *context = parser.getBuilder().getContext();
  if (succeeded(parser.parseOptionalKeyword("delay"))) {
    if (parser.parseAttribute(delayAttr, IntegerType::get(context, 64),
                              "hir.delay", argAttrs))
      return failure();
    attrsList.push_back(DictionaryAttr::get(context, argAttrs));
  } else {
    attrsList.push_back(helper::getDictionaryAttr(
        parser.getBuilder(), "hir.delay", helper::getIntegerAttr(context, 0)));
  }
  return success();
}

ParseResult parseMemrefPortsAttr(OpAsmParser &parser,
                                 SmallVectorImpl<DictionaryAttr> &attrsList) {

  Attribute memrefPortsAttr;
  if (parser.parseKeyword("ports") || parser.parseAttribute(memrefPortsAttr))
    return failure();

  attrsList.push_back(helper::getDictionaryAttr(
      parser.getBuilder(), "hir.memref.ports", memrefPortsAttr));

  return success();
}

ParseResult parseBusPortsAttr(OpAsmParser &parser,
                              SmallVectorImpl<DictionaryAttr> &attrsList) {

  llvm::SmallString<5> busPort;
  auto *context = parser.getBuilder().getContext();

  if (parser.parseKeyword("ports") || parser.parseLSquare())
    return failure();
  if (succeeded(parser.parseOptionalKeyword("send")))
    busPort = "send";
  else {
    if (failed(parser.parseOptionalKeyword("recv")))
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected 'send' or 'recv' port.");
    busPort = "recv";
  }
  if (parser.parseRSquare())
    return failure();

  attrsList.push_back(helper::getDictionaryAttr(
      parser.getBuilder(), "hir.bus.ports",
      ArrayAttr::get(context,
                     SmallVector<Attribute>({StringAttr::get(
                         parser.getBuilder().getContext(), busPort)}))));
  return success();
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Type members.
//------------------------------------------------------------------------------
static LogicalResult
verifyDelayAttribute(mlir::function_ref<InFlightDiagnostic()> emitError,
                     DictionaryAttr attrDict) {
  auto delayNameAndAttr = attrDict.getNamed("hir.delay");
  if (!delayNameAndAttr.hasValue())
    return failure();
  if (!delayNameAndAttr->second.dyn_cast<IntegerAttr>())
    return failure();
  return success();
}

static LogicalResult
verifyMemrefPortsAttribute(mlir::function_ref<InFlightDiagnostic()> emitError,
                           DictionaryAttr attrDict) {
  auto memrefPortsNameAndAttr = attrDict.getNamed("hir.memref.ports");
  if (!memrefPortsNameAndAttr.hasValue())
    return failure();
  if (!memrefPortsNameAndAttr->second.dyn_cast<ArrayAttr>())
    return failure();
  return success();
}

static LogicalResult
verifyBusPortsAttribute(mlir::function_ref<InFlightDiagnostic()> emitError,
                        DictionaryAttr attrDict) {
  auto memrefPortsNameAndAttr = attrDict.getNamed("hir.bus.ports");
  if (!memrefPortsNameAndAttr.hasValue())
    return failure();
  if (!memrefPortsNameAndAttr->second.dyn_cast<ArrayAttr>())
    return failure();
  return success();
}

LogicalResult FuncType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<Type> inputTypes,
                               ArrayRef<DictionaryAttr> inputAttrs,
                               ArrayRef<Type> resultTypes,
                               ArrayRef<DictionaryAttr> resultAttrs) {
  if (inputAttrs.size() != inputTypes.size())
    return emitError() << "Number of input attributes is not same as number of "
                          "input types.";

  if (resultAttrs.size() != resultTypes.size())
    return emitError()
           << "Number of result attributes is not same as number of "
              "result types.";

  // Verify inputs.
  for (size_t i = 0; i < inputTypes.size(); i++) {
    if (helper::isBuiltinType(inputTypes[i])) {
      if (failed(verifyDelayAttribute(emitError, inputAttrs[i])))
        return emitError() << "Expected hir.delay attribute to be an "
                              "IntegerAttr for input arg"
                           << std::to_string(i) << ".";
    } else if (inputTypes[i].dyn_cast<hir::MemrefType>()) {
      if (failed(verifyMemrefPortsAttribute(emitError, inputAttrs[i])))
        return emitError() << "Expected hir.memref.ports attribute to be an "
                              "ArrayAttr for input arg"
                           << std::to_string(i) << ".";
    } else if (inputTypes[i].dyn_cast<hir::BusType>()) {
      if (failed(verifyBusPortsAttribute(emitError, inputAttrs[i])))
        return emitError() << "Expected hir.bus.ports attribute to be an "
                              "ArrayAttr for input arg"
                           << std::to_string(i) << ".";
    } else if (inputTypes[i].dyn_cast<hir::TimeType>()) {
      return success();
    } else {
      return emitError()
             << "Expected MLIR-builtin-type or hir::MemrefType or "
                "hir::BusType or hir::TimeType in inputTypes, got :\n\t"
             << inputTypes[i];
    }
  }

  // Verify results.
  for (size_t i = 0; i < resultTypes.size(); i++) {
    if (helper::isBuiltinType(resultTypes[i])) {
      if (failed(verifyDelayAttribute(emitError, resultAttrs[i])))
        return emitError() << "Expected hir.delay attribute to be an "
                              "IntegerAttr for result arg"
                           << std::to_string(i) << ".";
    } else if (resultTypes[i].dyn_cast<hir::TimeType>()) {
      return success();
    } else {
      return emitError() << "Expected MLIR-builtin-type or hir::TimeType in "
                            "resultTypes, got :\n\t"
                         << resultTypes[i];
    }
  }
  return success();
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Operation parsers.
//------------------------------------------------------------------------------
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

  mlir::OptionalParseResult res = parser.parseOptionalOperand(calleeVar);
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

  auto *context = parser.getBuilder().getContext();
  // resolve operands.
  hir::FuncType funcTy = calleeTy.dyn_cast<hir::FuncType>();
  if (!funcTy)
    return parser.emitError(locCalleeTy, "expected !hir.func type!");

  if (!calleeIsSymbol)
    parser.resolveOperand(calleeVar, calleeTy, result.operands);
  parser.resolveOperands(operands, funcTy.getFunctionType().getInputs(), argLoc,
                         result.operands);
  parser.resolveOperand(tstart, helper::getTimeType(context), result.operands);
  if (isOffsetPresent)
    parser.resolveOperand(offset, IndexType::get(context), result.operands);

  // add attributes.
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(calleeIsSymbol ? 0 : 1),
                           static_cast<int32_t>(operands.size()), 1,
                           static_cast<int32_t>(isOffsetPresent ? 1 : 0)}));

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

  printer << "hir.if (" << op.cond() << ") at ";
  if (op.tstart()) {
    printer << op.tstart();
    if (op.offset())
      printer << " + " << op.offset();
  } else
    printer << "?";

  printer.printRegion(op.if_region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

static ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType cond;
  OpAsmParser::OperandType tstart;
  OpAsmParser::OperandType offset;
  if (parser.parseLParen() || parser.parseOperand(cond) || parser.parseRParen())
    return failure();
  if (parser.parseKeyword("at"))
    return failure();

  bool tstartPresent = false;
  bool offsetPresent = false;
  if (failed(parser.parseOptionalQuestion())) {
    if (parser.parseOperand(tstart))
      return failure();
    tstartPresent = true;
    if (succeeded(parser.parseOptionalPlus()))
      if (parser.parseOperand(offset))
        offsetPresent = true;
  }
  auto *context = parser.getBuilder().getContext();
  if (parser.resolveOperand(cond, IntegerType::get(context, 1),
                            result.operands))
    return failure();

  if (tstartPresent)
    if (parser.resolveOperand(tstart, TimeType::get(context), result.operands))
      return failure();
  if (offsetPresent)
    if (parser.resolveOperand(offset, IndexType::get(context), result.operands))
      return failure();

  result.addAttribute(
      "operand_segment_sizes",
      parser.getBuilder().getI32VectorAttr({1, // cond
                                            (int32_t)(tstartPresent ? 1 : 0),
                                            (int32_t)(offsetPresent ? 1 : 0)}));
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
  printer << "hir.for " << op.getInductionVar() << " : "
          << op.getInductionVar().getType() << " = " << op.lb() << " to "
          << op.ub() << " step " << op.step() << " iter_time( "
          << op.getIterTimeVar() << " = ";
  if (op.tstart()) {
    printer << op.tstart();
    if (op.offset())
      printer << " + " << op.offset();
  } else
    printer << "?";
  printer << ")";

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto *context = parser.getBuilder().getContext();
  Type timeTy = helper::getTimeType(context);
  Type lbRawType;
  Type ubRawType;
  Type stepRawType;
  Type tstartRawType = timeTy;
  Type offsetType = IndexType::get(context);
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

  lbRawType = regionRawOperandTypes[0];
  ubRawType = regionRawOperandTypes[0];
  stepRawType = regionRawOperandTypes[0];

  // Parse loop bounds.
  if (parser.parseOperand(lbRawOperand) || parser.parseKeyword("to") ||
      parser.parseOperand(ubRawOperand) || parser.parseKeyword("step") ||
      parser.parseOperand(stepRawOperand))
    return failure();

  // Parse iter time.
  if (parser.parseKeyword("iter_time") || parser.parseLParen() ||
      parser.parseRegionArgument(regionRawOperands[1]) || parser.parseEqual())
    return failure();

  bool tstartPresent = false;
  bool offsetPresent = false;
  if (failed(parser.parseOptionalQuestion())) {
    if (parser.parseOperand(tstartRawOperand))
      return failure();
    tstartPresent = true;
    if (succeeded(parser.parseOptionalPlus())) {
      if (parser.parseOperand(offsetRawOperand))
        return failure();
      offsetPresent = true;
    }
  }
  if (parser.parseRParen())
    return failure();

  // parse the loop bounds.
  if (parser.resolveOperand(lbRawOperand, lbRawType, result.operands) ||
      parser.resolveOperand(ubRawOperand, ubRawType, result.operands) ||
      parser.resolveOperand(stepRawOperand, stepRawType, result.operands))
    return failure();

  // parse optional tstart and offset.
  if (tstartPresent)
    if (parser.resolveOperand(tstartRawOperand, tstartRawType, result.operands))
      return failure();
  if (offsetPresent)
    if (parser.resolveOperand(offsetRawOperand, offsetType, result.operands))
      return failure();

  result.addAttribute(
      "operand_segment_sizes",
      parser.getBuilder().getI32VectorAttr({1, // lb
                                            1, // ub
                                            1, // step
                                            (int32_t)(tstartPresent ? 1 : 0),
                                            (int32_t)(offsetPresent ? 1 : 0)}));

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionOperands, regionOperandTypes))
    return failure();

  // First result is the time at which last iteration yields.
  result.addTypes(TimeType::get(context));
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
          << op.getIterTimeVar() << " = " << op.tstart();
  if (op.offset())
    printer << " + " << op.offset();
  printer << ")";

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

static ParseResult parseUnrollForOp(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();
  auto *context = parser.getBuilder().getContext();
  Type timeTypeVar = helper::getTimeType(context);
  Type tstartRawType = timeTypeVar;
  Type offsetType = IndexType::get(context);

  Type regionRawOperandTypes[2];
  ArrayRef<Type> regionOperandTypes(regionRawOperandTypes);
  regionRawOperandTypes[0] = IndexType::get(context);
  regionRawOperandTypes[1] = timeTypeVar;

  IntegerAttr lbAttr;
  IntegerAttr ubAttr;
  IntegerAttr stepAttr;
  OpAsmParser::OperandType tstartRawOperand;
  OpAsmParser::OperandType offsetRawOperand;
  OpAsmParser::OperandType regionRawOperands[2];

  ArrayRef<OpAsmParser::OperandType> regionOperands(regionRawOperands);
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(regionRawOperands[0]) || parser.parseEqual())
    return failure();

  // Parse loop bounds.

  if (helper::parseIntegerAttr(lbAttr, "lb", parser, result) ||
      parser.parseKeyword("to") ||
      helper::parseIntegerAttr(ubAttr, "ub", parser, result) ||
      parser.parseKeyword("step") ||
      helper::parseIntegerAttr(stepAttr, "step", parser, result))
    return failure();

  // Parse iter time.
  if (parser.parseKeyword("iter_time") || parser.parseLParen() ||
      parser.parseRegionArgument(regionRawOperands[1]) || parser.parseEqual() ||
      parser.parseOperand(tstartRawOperand))
    return failure();
  bool offsetIsPresent = false;
  if (!parser.parseOptionalPlus()) {
    if (parser.parseOperand(offsetRawOperand))
      return failure();
    offsetIsPresent = true;
  }

  if (parser.parseRParen())
    return failure();

  // resolve operands.
  if (parser.resolveOperand(tstartRawOperand, tstartRawType, result.operands))
    return failure();

  if (offsetIsPresent &&
      parser.resolveOperand(offsetRawOperand, offsetType, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();

  if (parser.parseRegion(*body, regionOperands, regionOperandTypes))
    return failure();
  ForOp::ensureTerminator(*body, builder, result.location);
  result.addTypes(helper::getTimeType(context));
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
/// hir.def @foo at %t (%x :i32 delay 1, %y: f32) ->(i1 delay 4){}
static ParseResult
parseFuncSignature(OpAsmParser &parser, hir::FuncType &funcTy,
                   SmallVectorImpl<OpAsmParser::OperandType> &entryArgs) {

  SmallVector<DictionaryAttr> inputAttrs;
  SmallVector<Type, 4> inputTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type, 4> resultTypes;
  auto *context = parser.getBuilder().getContext();
  // parse operand args
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    while (1) {
      // Parse operand and type
      if (parseOperandColonType(parser, entryArgs, inputTypes))
        return failure();

      // Parse argAttr
      if (helper::isBuiltinType(inputTypes.back())) {
        if (parseDelayAttr(parser, inputAttrs))
          return failure();
      } else if (inputTypes.back().dyn_cast<hir::TimeType>()) {
        inputAttrs.push_back(
            DictionaryAttr::get(context, SmallVector<NamedAttribute>({})));
      } else if (inputTypes.back().dyn_cast<hir::MemrefType>()) {
        if (parseMemrefPortsAttr(parser, inputAttrs))
          return failure();
      } else if (inputTypes.back().dyn_cast<hir::BusType>()) {
        if (parseBusPortsAttr(parser, inputAttrs))
          return failure();
      } else {
        return parser.emitError(parser.getCurrentLocation(),
                                "Expected builtin type or hir dialect type.");
      }

      if (failed(parser.parseOptionalComma()))
        break;
    }
    if (parser.parseRParen())
      return failure();
  }

  // If result types present then parse them.
  if (succeeded(parser.parseOptionalArrow())) {
    // parse result args
    if (parser.parseLParen())
      return failure();

    if (parser.parseOptionalRParen()) {
      while (1) {
        // Parse result type
        Type resultTy;
        if (parser.parseType(resultTy))
          return failure();
        resultTypes.push_back(resultTy);

        // Parse delayAttr
        if (parseDelayAttr(parser, resultAttrs))
          if (parser.parseOptionalComma())
            break;
        if (failed(parser.parseOptionalComma()))
          break;
      }
      if (parser.parseRParen())
        return failure();
    }
  }

  funcTy = hir::FuncType::get(parser.getBuilder().getContext(), inputTypes,
                              inputAttrs, resultTypes, resultAttrs);
  return success();
}

// Parse method.
static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  OpAsmParser::OperandType tstart;
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
  if (parseFuncSignature(parser, funcTy, entryArgs))
    return failure();

  auto functionTy = funcTy.getFunctionType();
  result.addAttribute(mlir::function_like_impl::getTypeAttrName(),
                      TypeAttr::get(functionTy));
  result.addAttribute("funcTy", TypeAttr::get(funcTy));

  // Add the attributes to the function arguments.
  mlir::function_like_impl::addArgAndResultAttrs(
      builder, result, funcTy.getInputAttrs(), funcTy.getResultAttrs());
  // Parse the optional function body.
  auto *body = result.addRegion();
  entryArgs.push_back(tstart);
  SmallVector<Type> entryArgTypes;
  for (auto ty : funcTy.getFunctionType().getInputs()) {
    entryArgTypes.push_back(ty);
  }
  entryArgTypes.push_back(
      helper::getTimeType(parser.getBuilder().getContext()));
  auto r = parser.parseOptionalRegion(*body, entryArgs, entryArgTypes);
  if (r.hasValue())
    return r.getValue();
  return success();
}

static void printFuncSignature(OpAsmPrinter &printer, hir::FuncOp op) {
  auto fnType = op.getType();
  Region &body = op.getOperation()->getRegion(0);
  auto inputTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();
  ArrayRef<DictionaryAttr> inputAttrs =
      op.funcTy().dyn_cast<FuncType>().getInputAttrs();
  ArrayRef<DictionaryAttr> resultAttrs =
      op.funcTy().dyn_cast<FuncType>().getResultAttrs();

  printer << "(";
  for (unsigned i = 0; i < inputTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << body.front().getArgument(i) << " : " << inputTypes[i];

    if (helper::isBuiltinType(inputTypes[i])) {
      auto delay = helper::extractDelayFromDict(inputAttrs[i]);
      if (delay != 0)
        printer << " delay " << delay;
    } else if (inputTypes[i].dyn_cast<hir::MemrefType>()) {
      auto ports = helper::extractMemrefPortsFromDict(inputAttrs[i]);
      printer << " ports " << ports;
    } else if (inputTypes[i].dyn_cast<hir::BusType>()) {
      auto busPortStr = helper::extractBusPortFromDict(inputAttrs[i]);
      printer << " ports [" << busPortStr << "]";
    }
  }
  printer << ")";
  if (resultTypes.size() == 0)
    return;

  printer << " -> (";
  for (unsigned i = 0; i < resultTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << resultTypes[i];
    auto delay = helper::extractDelayFromDict(resultAttrs[i]);
    if (delay != 0)
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

namespace circt {
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/IR/HIR.cpp.inc"
} // namespace circt
