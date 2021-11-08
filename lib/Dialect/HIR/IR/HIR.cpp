//=========- HIR.cpp - Parser & Printer for Ops ---------------------------===//
//
// This file implements parsers and printers for ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/HIROpSyntax.h"
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
static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (const auto &argAttr : attrs)
    if (argAttr.first == name)
      return true;
  return false;
}

static StringAttr getNameAttr(MLIRContext *context, StringRef name) {
  if (!name.empty()) {
    // Ignore numeric names like %42
    assert(name.size() > 1 && name[0] == '%' && "Unknown MLIR name");
    if (isdigit(name[1]))
      name = StringRef();
    else
      name = name.drop_front();
  }
  return StringAttr::get(context, name);
}

LogicalResult addNamesAttribute(MLIRContext *context, StringRef attrName,
                                ArrayRef<OpAsmParser::OperandType> args,
                                OperationState &result) {

  if (args.empty())
    return success();

  // Use SSA names only if names are not previously defined.
  if (!hasAttribute(attrName, result.attributes)) {
    SmallVector<Attribute> argNames;
    for (const auto &arg : args)
      argNames.push_back(getNameAttr(context, arg.name));

    result.addAttribute(attrName, ArrayAttr::get(context, argNames));
  }
  return success();
}

ParseResult
parseNamedOperandColonType(OpAsmParser &parser,
                           SmallVectorImpl<OpAsmParser::OperandType> &entryArgs,
                           SmallVectorImpl<Type> &argTypes) {

  OpAsmParser::OperandType operand;
  Type operandTy;
  auto operandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operand) || parser.parseColonType(operandTy))
    return failure();
  if (operand.name.empty())
    return parser.emitError(operandLoc) << "SSA value must have a valid name.";
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
    attrsList.push_back(
        helper::getDictionaryAttr(parser.getBuilder(), "hir.delay",
                                  helper::getI64IntegerAttr(context, 0)));
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
  if (!attrDict)
    return failure();
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

LogicalResult MemrefType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType,
                                 ArrayRef<DimKind> dimKinds) {
  for (size_t i = 0; i < shape.size(); i++) {
    if (dimKinds[i] == ADDR) {
      if ((pow(2, helper::clog2(shape[i]))) != shape[i]) {
        return emitError()
               << "hir.memref dimension sizes must be a power of two, dim " << i
               << " has size " << shape[i];
      }
      if (shape[i] <= 0) {
        return emitError() << "hir.memref dimension size must be >0. dim " << i
                           << " has size " << shape[i];
      }
    }
  }
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
    if (helper::isBuiltinSizedType(inputTypes[i])) {
      if (failed(verifyDelayAttribute(emitError, inputAttrs[i])))
        return emitError() << "Expected hir.delay IntegerAttr for input arg"
                           << std::to_string(i) << ".";
    } else if (inputTypes[i].dyn_cast<hir::MemrefType>()) {
      if (failed(verifyMemrefPortsAttribute(emitError, inputAttrs[i])))
        return emitError()
               << "Expected hir.memref.ports ArrayAttr for input arg"
               << std::to_string(i) << ".";
    } else if (helper::isBusLikeType(inputTypes[i])) {
      if (failed(verifyBusPortsAttribute(emitError, inputAttrs[i])))
        return emitError() << "Expected hir.bus.ports ArrayAttr for input arg"
                           << std::to_string(i) << ".";
    } else if (inputTypes[i].dyn_cast<hir::TimeType>()) {
      continue;
    } else {
      return emitError() << "Expected MLIR-builtin-type or hir::MemrefType or "
                            "hir::BusType or hir::BusTensorType or "
                            "hir::TimeType in inputTypes, got :\n\t"
                         << inputTypes[i];
    }
  }

  // Verify results.
  for (size_t i = 0; i < resultTypes.size(); i++) {
    if (helper::isBuiltinSizedType(resultTypes[i])) {
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
  mlir::FlatSymbolRefAttr calleeAttr;
  Type calleeTy;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  llvm::Optional<OpAsmParser::OperandType> tstart;
  IntegerAttr offset;
  StringAttr instanceNameAttr;
  auto instanceNameParseResult =
      parser.parseOptionalAttribute(instanceNameAttr);
  if (instanceNameParseResult.hasValue() && instanceNameParseResult.getValue())
    return failure();

  if (parser.parseAttribute(calleeAttr))
    return failure();

  if (parser.parseLParen())
    return failure();

  if (parser.parseOperandList(operands))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("at"))
    return failure();
  parseTimeAndOffset(parser, tstart, offset);

  if (parseWithSSANames(parser, result.attributes))
    return failure();

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

  if (parser.resolveOperands(operands, funcTy.getFunctionType().getInputs(),
                             parser.getNameLoc(), result.operands))
    return failure();
  if (tstart.hasValue())
    if (parser.resolveOperand(tstart.getValue(), helper::getTimeType(context),
                              result.operands))
      return failure();

  // add attributes.
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(operands.size()),
                           static_cast<int32_t>(tstart.hasValue() ? 1 : 0)}));

  if (instanceNameAttr)
    result.addAttribute("instance_name", instanceNameAttr);
  result.addAttribute("callee", calleeAttr);
  result.addAttribute("funcTy", TypeAttr::get(calleeTy));
  if (offset)
    result.addAttribute("offset", offset);
  result.addTypes(funcTy.getFunctionType().getResults());

  return success();
}

static void printCallOp(OpAsmPrinter &printer, CallOp op) {
  printer << " ";
  if (op.instance_name().hasValue())
    printer << op.instance_nameAttr();
  printer << " @" << op.callee();
  printer << "(" << op.operands() << ") at ";
  printTimeAndOffset(printer, op, op.tstart(), op.offsetAttr());
  printWithSSANames(printer, op, op->getAttrDictionary());
  printer << " : ";
  printer << op.funcTy();
}

/// CallInstanceOp
/// Syntax:
/// $callee `(` $operands `)` `at` $tstart (`offset` $offset^ )?
///   `:` ($operand (delay $delay^)?) `->` ($results)

static ParseResult parseCallInstanceOp(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::OperandType callee;
  Type calleeTy;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  llvm::Optional<OpAsmParser::OperandType> tstart;
  IntegerAttr offsetAttr;

  if (parser.parseOperand(callee))
    return failure();

  if (parser.parseLParen())
    return failure();

  llvm::SMLoc argLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operands))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("at"))
    return failure();
  parseTimeAndOffset(parser, tstart, offsetAttr);

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

  if (parser.resolveOperand(callee, funcTy, result.operands) ||
      parser.resolveOperands(operands, funcTy.getFunctionType().getInputs(),
                             argLoc, result.operands))
    return failure();
  if (tstart.hasValue())
    if (parser.resolveOperand(tstart.getValue(), helper::getTimeType(context),
                              result.operands))
      return failure();

  // add attributes.
  result.addAttribute(
      "operand_segment_sizes",
      parser.getBuilder().getI32VectorAttr({
          1,                                              // callee
          static_cast<int32_t>(operands.size()),          // operands
          static_cast<int32_t>(tstart.hasValue() ? 1 : 0) // tstart
      }));

  result.addAttribute("offset", offsetAttr);
  result.addTypes(funcTy.getFunctionType().getResults());

  return success();
}

static void printCallInstanceOp(OpAsmPrinter &printer, CallInstanceOp op) {
  printer << "hir.call.instance " << op.callee();
  printer << "(" << op.operands() << ") at ";
  printTimeAndOffset(printer, op, op.tstart(), op.offsetAttr());
  printer << " : ";
  printer << op.callee().getType();
}

/// IfOp
static void printIfOp(OpAsmPrinter &printer, IfOp op) {

  printer << " " << op.condition();

  printer << " at time(" << op.if_region().getArgument(0) << " = ";
  printTimeAndOffset(printer, op, op.tstart(), op.offsetAttr());
  printer << ")";

  if (op.results().size() > 0) {
    printer << " -> (";
    printTypeAndDelayList(printer, op->getResultTypes(),
                          op.result_attrs().getValue());
    printer << ")";
  }

  printer.printRegion(op.if_region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printer << "else";
  printer.printRegion(op.else_region(), false, true);
  printWithSSANames(printer, op, op->getAttrDictionary());
}

static ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType cond;
  llvm::Optional<OpAsmParser::OperandType> tstart;
  OpAsmParser::OperandType timevar;
  IntegerAttr offsetAttr;
  SmallVector<Type> resultTypes;
  ArrayAttr resultAttrs;

  // parse the boolean condition
  if (parser.parseOperand(cond))
    return failure();

  // parse time.
  if (parser.parseKeyword("at") || parser.parseKeyword("time") ||
      parser.parseLParen() || parser.parseRegionArgument(timevar) ||
      parser.parseEqual())
    return failure();

  parseTimeAndOffset(parser, tstart, offsetAttr);
  if (parser.parseRParen())
    return failure();

  if (succeeded(parser.parseOptionalArrow()))
    if (parser.parseLParen() ||
        parseTypeAndDelayList(parser, resultTypes, resultAttrs) ||
        parser.parseRParen())
      return failure();
  auto *context = parser.getBuilder().getContext();
  if (parser.resolveOperand(cond, IntegerType::get(context, 1),
                            result.operands))
    return failure();

  if (tstart.hasValue())
    if (parser.resolveOperand(tstart.getValue(), TimeType::get(context),
                              result.operands))
      return failure();

  if (offsetAttr)
    result.addAttribute("offset", offsetAttr);
  if (resultTypes.size() > 0)
    result.addAttribute("result_attrs", resultAttrs);
  // Add outputs.
  if (resultTypes.size() > 0)
    result.addTypes(resultTypes);

  Region *ifBody = result.addRegion();
  Region *elseBody = result.addRegion();

  if (parser.parseRegion(*ifBody, {timevar}, {hir::TimeType::get(context)}))
    return failure();
  if (parser.parseKeyword("else"))
    return failure();
  if (parser.parseRegion(*elseBody, {timevar}, {hir::TimeType::get(context)}))
    return failure();

  parseWithSSANames(parser, result.attributes);

  // IfOp::ensureTerminator(*ifBody, builder, result.location);
  return success();
}

// WhileOp
// Example:
// hir.while(%b) at iter_time(%tw = %t + 1){}
static ParseResult parseWhileOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType iterTimeVar;
  OpAsmParser::OperandType conditionVar;
  llvm::Optional<OpAsmParser::OperandType> tstart;
  IntegerAttr offset;
  if (parser.parseOperand(conditionVar))
    return failure();

  if (parser.parseKeyword("at") || parser.parseKeyword("iter_time") ||
      parser.parseLParen() || parser.parseRegionArgument(iterTimeVar) ||
      parser.parseEqual())
    return failure();

  if (parseTimeAndOffset(parser, tstart, offset) || parser.parseRParen())
    return failure();

  if (parser.resolveOperand(conditionVar, parser.getBuilder().getI1Type(),
                            result.operands))
    return failure();

  if (tstart.hasValue())
    if (parser.resolveOperand(
            tstart.getValue(),
            hir::TimeType::get(parser.getBuilder().getContext()),
            result.operands))
      return failure();

  if (offset)
    result.addAttribute("offset", offset);

  Region *body = result.addRegion();
  if (parser.parseRegion(
          *body, {iterTimeVar},
          {hir::TimeType::get(parser.getBuilder().getContext())}))
    return failure();
  // Parse the attr-dict
  if (parseWithSSANames(parser, result.attributes))
    return failure();
  result.addTypes(TimeType::get(parser.getBuilder().getContext()));
  return success();
}

static void printWhileOp(OpAsmPrinter &printer, WhileOp op) {
  printer << " " << op.condition();
  printer << " at iter_time(" << op.getIterTimeVar() << " = ";
  printTimeAndOffset(printer, op, op.tstart(), op.offsetAttr());
  printer << ")";
  printer.printRegion(op->getRegion(0),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printWithSSANames(printer, op, op->getAttrDictionary());
}

// ForOp.
// Example:
// hir.for %i = %l to %u step %s iter_time(%ti = %t + 1){...}
static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
  auto *context = parser.getBuilder().getContext();
  Type inductionVarTy;

  OpAsmParser::OperandType iterTimeVar;
  OpAsmParser::OperandType inductionVar;
  OpAsmParser::OperandType lb;
  OpAsmParser::OperandType ub;
  OpAsmParser::OperandType step;
  llvm::Optional<OpAsmParser::OperandType> tstart;
  IntegerAttr offset;

  // Parse the induction variable followed by '='.
  auto inductionVarLoc = parser.getCurrentLocation();
  if (parser.parseRegionArgument(inductionVar) ||
      parser.parseColonType(inductionVarTy) || parser.parseEqual())
    return failure();
  if (inductionVar.name.empty())
    return parser.emitError(inductionVarLoc) << "Expected valid name.";

  // Parse loop bounds.
  if (parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  // Parse iter-time.
  if (parser.parseKeyword("iter_time") || parser.parseLParen())
    return failure();
  auto iterTimeVarLoc = parser.getCurrentLocation();
  if (parser.parseRegionArgument(iterTimeVar) || parser.parseEqual())
    return failure();

  if (iterTimeVar.name.empty())
    return parser.emitError(iterTimeVarLoc) << "Expected valid name.";

  if (parseTimeAndOffset(parser, tstart, offset) || parser.parseRParen())
    return failure();

  // resolve the loop bounds.
  if (parser.resolveOperand(lb, inductionVarTy, result.operands) ||
      parser.resolveOperand(ub, inductionVarTy, result.operands) ||
      parser.resolveOperand(step, inductionVarTy, result.operands))
    return failure();

  // resolve optional tstart and offset.
  if (tstart.hasValue())
    if (parser.resolveOperand(tstart.getValue(), TimeType::get(context),
                              result.operands))
      return failure();
  if (offset)
    result.addAttribute("offset", offset);

  if (failed(addNamesAttribute(context, "argNames", {inductionVar, iterTimeVar},
                               result)))
    return parser.emitError(inductionVarLoc)
           << "Failed to add names for induction var and time var";

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {inductionVar, iterTimeVar},
                         {inductionVarTy, TimeType::get(context)}))
    return failure();

  // Parse the attr-dict
  if (parseWithSSANames(parser, result.attributes))
    return failure();

  // ForOp result is the time at which last iteration calls next_iter.
  result.addTypes(TimeType::get(context));

  // ForOp::ensureTerminator(*body, builder, result.location);
  return success();
}

static void printForOp(OpAsmPrinter &printer, ForOp op) {
  printer << " " << op.getInductionVar() << " : "
          << op.getInductionVar().getType() << " = " << op.lb() << " to "
          << op.ub() << " step " << op.step();

  printer << " iter_time( " << op.getIterTimeVar() << " = ";
  printTimeAndOffset(printer, op, op.tstart(), op.offsetAttr());
  printer << ")";

  printer.printRegion(op->getRegion(0),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printWithSSANames(printer, op, op->getAttrDictionary());
}

Region &ForOp::getLoopBody() { return body(); }

bool ForOp::isDefinedOutsideOfLoop(Value value) {
  return !getLoopBody().isAncestor(value.getParentRegion());
}

LogicalResult ForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
  return success();
}

/// FuncOp
/// Example:
/// hir.def @foo at %t (%x :i32 delay 1, %y: f32) ->(%out: i1 delay 4){}
static ParseResult parseArgList(OpAsmParser &parser,
                                SmallVectorImpl<OpAsmParser::OperandType> &args,
                                SmallVectorImpl<Type> &argTypes,
                                SmallVectorImpl<DictionaryAttr> &argAttrs) {

  auto *context = parser.getBuilder().getContext();
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    while (1) {
      // Parse operand and type
      auto operandLoc = parser.getCurrentLocation();
      if (parseNamedOperandColonType(parser, args, argTypes))
        return failure();
      auto argTy = argTypes.back();
      // Parse argAttr
      if (helper::isBuiltinSizedType(argTy)) {
        if (parseDelayAttr(parser, argAttrs))
          return failure();
      } else if (argTy.isa<hir::TimeType>()) {
        argAttrs.push_back(
            DictionaryAttr::get(context, SmallVector<NamedAttribute>({})));
      } else if (argTy.isa<hir::MemrefType>()) {
        if (parseMemrefPortsAttr(parser, argAttrs))
          return failure();
      } else if (helper::isBusLikeType(argTy)) {
        if (parseBusPortsAttr(parser, argAttrs))
          return failure();
      } else
        return parser.emitError(operandLoc, "Unsupported type.");

      if (failed(parser.parseOptionalComma()))
        break;
    }
    if (parser.parseRParen())
      return failure();
  }
  return success();
}

static ParseResult
parseFuncSignature(OpAsmParser &parser, hir::FuncType &funcTy,
                   SmallVectorImpl<OpAsmParser::OperandType> &args,
                   SmallVectorImpl<OpAsmParser::OperandType> &results) {
  SmallVector<Type, 4> argTypes;
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<Type, 4> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  // parse args
  if (parseArgList(parser, args, argTypes, argAttrs))
    return failure();

  // If result types present then parse them.
  if (succeeded(parser.parseOptionalArrow()))
    if (parseArgList(parser, results, resultTypes, resultAttrs))
      return failure();

  funcTy = hir::FuncType::get(parser.getBuilder().getContext(), argTypes,
                              argAttrs, resultTypes, resultAttrs);
  return success();
}
static ParseResult
parseFuncDecl(OpAsmParser &parser, OperationState &result,
              SmallVectorImpl<OpAsmParser::OperandType> &inputVars,
              hir::FuncType &funcTy) {

  SmallVector<OpAsmParser::OperandType, 4> resultArgs;
  OpAsmParser::OperandType tstart;
  auto &builder = parser.getBuilder();
  auto *context = builder.getContext();
  // Parse the name as a symbol.
  StringAttr functionName;
  if (parser.parseSymbolName(functionName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();
  // Parse tstart.
  if (parser.parseKeyword("at") || parser.parseRegionArgument(tstart))
    return failure();
  if (tstart.name.empty())
    return parser.emitError(parser.getCurrentLocation())
           << "Expected valid name for start time.";
  auto loc = parser.getCurrentLocation();
  // Parse the function signature.
  if (parseFuncSignature(parser, funcTy, inputVars, resultArgs))
    return failure();
  inputVars.push_back(tstart);

  result.addAttribute("funcTy", TypeAttr::get(funcTy));

  // Use the argument and result names if not already specified.
  if (failed(addNamesAttribute(context, "argNames", inputVars, result)))
    return parser.emitError(loc)
           << "Function input arguments must have a valid name.";
  if (failed(addNamesAttribute(context, "resultNames", resultArgs, result)))
    return parser.emitError(loc)
           << "Function return arguments must have a valid name.";

  // Add the attributes for FunctionLike interface.
  auto functionTy = funcTy.getFunctionType();
  result.addAttribute(mlir::function_like_impl::getTypeAttrName(),
                      TypeAttr::get(functionTy));

  mlir::function_like_impl::addArgAndResultAttrs(
      builder, result, funcTy.getInputAttrs(), funcTy.getResultAttrs());

  return success();
}

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  hir::FuncType funcTy;

  auto &builder = parser.getBuilder();

  if (parseFuncDecl(parser, result, entryArgs, funcTy))
    return failure();

  // Parse the function body.
  auto *body = result.addRegion();
  SmallVector<Type> entryArgTypes;
  for (auto ty : funcTy.getFunctionType().getInputs()) {
    entryArgTypes.push_back(ty);
  }
  entryArgTypes.push_back(TimeType::get(builder.getContext()));

  if (parser.parseRegion(*body, entryArgs, entryArgTypes))
    return failure();

  parser.parseOptionalAttrDict(result.attributes);
  FuncOp::ensureTerminator(*body, builder, result.location);
  return success();
}

static ParseResult parseFuncExternOp(OpAsmParser &parser,
                                     OperationState &result) {

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  hir::FuncType funcTy;
  if (parseFuncDecl(parser, result, entryArgs, funcTy))
    return failure();
  // Parse the function body.
  auto *body = result.addRegion();
  SmallVector<Type> entryArgTypes;
  for (auto ty : funcTy.getFunctionType().getInputs()) {
    entryArgTypes.push_back(ty);
  }
  auto &builder = parser.getBuilder();
  entryArgTypes.push_back(TimeType::get(builder.getContext()));
  body->push_back(new Block);
  body->front().addArguments(entryArgTypes);
  parser.parseOptionalAttrDict(result.attributes);
  FuncExternOp::ensureTerminator(*body, builder, result.location);
  return success();
}

static ParseResult printArgList(OpAsmPrinter &printer, ArrayAttr argNames,
                                ArrayRef<Type> argTypes,
                                ArrayRef<DictionaryAttr> argAttrs) {
  printer << "(";
  for (unsigned i = 0; i < argTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << "%" << argNames[i].dyn_cast<StringAttr>().getValue() << " : "
            << argTypes[i];

    if (helper::isBuiltinSizedType(argTypes[i])) {
      auto delay = helper::extractDelayFromDict(argAttrs[i]);
      if (delay)
        printer << " delay " << delay;
    } else if (argTypes[i].isa<hir::MemrefType>()) {
      printer << " ports " << helper::extractMemrefPortsFromDict(argAttrs[i]);
    } else if (helper::isBusLikeType(argTypes[i])) {
      printer << " ports [" << helper::extractBusPortFromDict(argAttrs[i])
              << "]";
    }
  }
  printer << ")";

  return success();
}

static void printFuncSignature(OpAsmPrinter &printer, hir::FuncType funcTy,
                               ArrayAttr inputNames, ArrayAttr resultNames) {
  auto inputTypes = funcTy.getInputTypes();
  auto resultTypes = funcTy.getResultTypes();
  ArrayRef<DictionaryAttr> inputAttrs = funcTy.getInputAttrs();
  ArrayRef<DictionaryAttr> resultAttrs = funcTy.getResultAttrs();

  printArgList(printer, inputNames, inputTypes, inputAttrs);

  if (resultTypes.size() == 0)
    return;

  printer << " -> ";

  printArgList(printer, resultNames, resultTypes, resultAttrs);
}

static void printFuncExternOp(OpAsmPrinter &printer, hir::FuncExternOp op) {
  // Print function name, signature, and control.
  printer << " ";
  printer.printSymbolName(op.sym_name());
  auto inputNames = op.getInputNames();
  printer << " at "
          << "%"
          << inputNames[inputNames.size() - 1].dyn_cast<StringAttr>().getValue()
          << " ";
  printFuncSignature(printer, op.getFuncType(), op.getInputNames(),
                     op.getResultNames());
  printer.printOptionalAttrDict(op->getAttrs(),
                                {"funcTy", "type", "arg_attrs", "res_attrs",
                                 "sym_name", "argNames", "resultNames"});
}

static void printFuncOp(OpAsmPrinter &printer, hir::FuncOp op) {
  // Print function name, signature, and control.
  printer << " ";
  printer.printSymbolName(op.sym_name());
  Region &body = op.getOperation()->getRegion(0);
  printer << " at "
          << body.front().getArgument(body.front().getNumArguments() - 1)
          << " ";

  printFuncSignature(printer, op.getFuncType(), op.getInputNames(),
                     op.getResultNames());

  printer.printRegion(body, /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDict(op->getAttrs(),
                                {"funcTy", "type", "arg_attrs", "res_attrs",
                                 "sym_name", "argNames", "resultNames"});
}

/// TensorInsertOp parser and printer.
/// Syntax: hir.tensor.insert %element into %tensor[%c0, %c1]
/// custom<WithSSANames>(attr-dict): type($res)
static ParseResult parseBusTensorInsertElementOp(OpAsmParser &parser,
                                                 OperationState &result) {
  OpAsmParser::OperandType element;
  OpAsmParser::OperandType inputTensor;
  SmallVector<OpAsmParser::OperandType> indices;
  hir::BusTensorType resTy;
  auto builder = parser.getBuilder();
  if (parser.parseOperand(element) || parser.parseKeyword("into") ||
      parser.parseOperand(inputTensor))
    return failure();

  if (parser.parseOperandList(indices, -1,
                              mlir::OpAsmParser::Delimiter::Square))
    return failure();

  if (parseWithSSANames(parser, result.attributes))
    return failure();
  if (parser.parseColonType(resTy))
    return failure();

  if (parser.resolveOperand(
          element,
          hir::BusType::get(parser.getContext(), resTy.getElementType()),
          result.operands))
    return failure();
  if (parser.resolveOperand(inputTensor, resTy, result.operands))
    return failure();
  if (parser.resolveOperands(indices, builder.getIndexType(), result.operands))
    return failure();

  result.addTypes(resTy);
  return success();
}

static void printBusTensorInsertElementOp(OpAsmPrinter &printer,
                                          hir::BusTensorInsertElementOp op) {
  printer << " " << op.element() << " into " << op.tensor();
  printer << "[";
  printer.printOperands(op.indices());
  printer << "]";
  printWithSSANames(printer, op, op->getAttrDictionary());
  printer << " : ";
  printer << op.res().getType();
}

/// BusMapOp parser and printer
static ParseResult parseBusMapOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType> operands;
  SmallVector<Type> regionArgTypes;
  SmallVector<OpAsmParser::OperandType> regionArgs;
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::OperandType regionArg;
    OpAsmParser::OperandType operand;
    if (parser.parseRegionArgument(regionArg) || parser.parseEqual() ||
        parser.parseOperand(operand))
      return failure();
    operands.push_back(operand);
    regionArgs.push_back(regionArg);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen() || parser.parseColon())
    return failure();

  mlir::FunctionType funcTy;
  if (parser.parseType(funcTy))
    return failure();

  for (auto ty : funcTy.getInputs()) {
    auto busTy = ty.dyn_cast<hir::BusType>();
    if (!busTy)
      return parser.emitError(parser.getNameLoc())
             << "Inputs must be hir.bus type.";
    regionArgTypes.push_back(busTy.getElementType());
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs, regionArgTypes))
    return failure();

  parser.resolveOperands(operands, funcTy.getInputs(), parser.getNameLoc(),
                         result.operands);
  result.addTypes(funcTy.getResults());
  return success();
}

static void printBusMapOp(OpAsmPrinter &printer, hir::BusMapOp op) {
  printer << " (";
  for (size_t i = 0; i < op.getNumOperands(); i++) {
    if (i > 0)
      printer << ",";
    printer << op.body().front().getArgument(i) << " = " << op.operands()[i];
  }
  printer << ") : (";
  for (size_t i = 0; i < op.getNumOperands(); i++) {
    if (i > 0)
      printer << ",";
    printer << op.operands()[i].getType();
  }
  printer << ") -> (";
  for (size_t i = 0; i < op.getNumResults(); i++) {
    auto result = op.getResult(i);
    if (i > 0)
      printer << ",";
    printer << result.getType();
  }
  printer << ")";
  printer.printRegion(op.body(), false, true);
}

/// BusTensorMapOp parser and printer
static ParseResult parseBusTensorMapOp(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::OperandType> operands;
  SmallVector<OpAsmParser::OperandType> regionArgs;
  SmallVector<Type> regionArgTypes;
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::OperandType regionArg;
    OpAsmParser::OperandType operand;
    if (parser.parseRegionArgument(regionArg) || parser.parseEqual() ||
        parser.parseOperand(operand))
      return failure();
    operands.push_back(operand);
    regionArgs.push_back(regionArg);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen() || parser.parseColon())
    return failure();

  mlir::FunctionType funcTy;
  if (parser.parseType(funcTy))
    return failure();

  for (auto ty : funcTy.getInputs()) {
    auto busTensorTy = ty.dyn_cast<hir::BusTensorType>();
    if (!busTensorTy)
      return parser.emitError(parser.getNameLoc())
             << "Inputs must be hir.bus_tensor type.";
    regionArgTypes.push_back(busTensorTy.getElementType());
  }
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs, regionArgTypes))
    return failure();

  parser.resolveOperands(operands, funcTy.getInputs(), parser.getNameLoc(),
                         result.operands);
  result.addTypes(funcTy.getResults());
  return success();
}

static void printBusTensorMapOp(OpAsmPrinter &printer, hir::BusTensorMapOp op) {
  printer << " (";
  for (size_t i = 0; i < op.getNumOperands(); i++) {
    if (i > 0)
      printer << ",";
    printer << op.body().front().getArgument(i) << " = " << op.operands()[i];
  }
  printer << ") : (";
  for (size_t i = 0; i < op.getNumOperands(); i++) {
    if (i > 0)
      printer << ",";
    printer << op.operands()[i].getType();
  }
  printer << ") -> (";
  for (size_t i = 0; i < op.getNumResults(); i++) {
    auto result = op.getResult(i);
    if (i > 0)
      printer << ",";
    printer << result.getType();
  }
  printer << ")";
  printer.printRegion(op.body(), false, true);
}

LogicalResult hir::FuncExternOp::verifyType() { return success(); }

LogicalResult hir::FuncOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}
/// required for functionlike trait
LogicalResult hir::FuncOp::verifyBody() { return success(); }

LogicalResult
BusTensorType::verify(mlir::function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<int64_t> shape, Type elementTy) {
  if (!helper::isBuiltinSizedType(elementTy))
    emitError() << "Bus inner type can only be an integer/float or a "
                   "tuple/tensor of these types.";
  for (auto dim : shape)
    if (dim <= 0)
      emitError() << "Dimension size must be greater than zero.";
  return success();
}

#include "HIROpVerifier.h"

#define GET_OP_CLASSES
#include "circt/Dialect/HIR/IR/HIR.cpp.inc"
