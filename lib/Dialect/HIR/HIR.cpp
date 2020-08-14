#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
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

// TODO: replace all integer type with hir.int type

// Helper Methods.

static Type getIntegerType(OpAsmParser &parser, int bitwidth) {
  return parser.getBuilder().getIntegerType(bitwidth);
}

static Type getTimeType(OpAsmParser &parser) {
  return TimeType::get(parser.getBuilder().getContext());
}

/* MemReadOp.
 * Example:
 * hir.mem_read %mem[%add] at %t : !hir.mem_interface -> i32
 */

/*
static void printMemReadOp(OpAsmPrinter &printer, MemReadOp op) {
  printer << "hir.mem_read"
          << " " << op.mem() << "[" << op.addr() << "]"
          << " at " << op.tstart() << " : " << op.mem().getType() << " -> "
          << op.res().getType();
  if (op.mem() == op.addr()) {
  }
}

static ParseResult parseMemReadOp(OpAsmParser &parser, OperationState &result) {
  Type memType;
  Type resRawType;
  OpAsmParser::OperandType memRawOperand;
  OpAsmParser::OperandType addrRawOperand;
  OpAsmParser::OperandType tstartRawOperand;
  if (parser.parseOperand(memRawOperand) || parser.parseLSquare() ||
      parser.parseOperand(addrRawOperand) || parser.parseRSquare() ||
      parser.parseKeyword("at") || parser.parseOperand(tstartRawOperand) ||
      parser.parseColon() || parser.parseType(memType) || parser.parseArrow() ||
      parser.parseType(resRawType))
    return failure();

  if (parser.resolveOperand(memRawOperand, memType, result.operands))
    return failure();
  Type odsBuildableTimeType = getTimeType(parser);
  Type intTypeVar = getIntegerType(parser, 32);
  if (parser.resolveOperand(addrRawOperand, intTypeVar, result.operands))
    return failure();
  if (parser.resolveOperand(tstartRawOperand, odsBuildableTimeType,
                            result.operands))
    return failure();

  // Right now the mem interface takes only one address.
  // TODO: Add support for arbitrary number of address parameters.
  auto numAddrOperandsAttr = parser.getBuilder().getIntegerAttr(intTypeVar, 1);
  result.attributes.set("num_addr_operands", numAddrOperandsAttr);
  result.addTypes(resRawType);
  return success();
}
*/

/* MemWriteOp
 * Example:
 * hir.mem_write %v to %mem[%add] at %t : (!hir.int, !hir.mem_interface)
 */

static void printMemWriteOp(OpAsmPrinter &printer, MemWriteOp op) {
  printer << "hir.mem_write"
          << " " << op.value() << " to " << op.mem() << "[ " << op.addr()
          << " ]"
          << " at " << op.tstart() << " : "
          << "( " << op.value().getType() << ", " << op.mem().getType() << " )";
}

static ParseResult parseMemWriteOp(OpAsmParser &parser,
                                   OperationState &result) {
  Type valueType;
  Type memType;
  OpAsmParser::OperandType valueRawOperand;
  OpAsmParser::OperandType memRawOperand;
  OpAsmParser::OperandType addrRawOperand;
  OpAsmParser::OperandType tstartRawOperand;
  if (parser.parseOperand(valueRawOperand) || parser.parseKeyword("to") ||
      parser.parseOperand(memRawOperand) || parser.parseLSquare() ||
      parser.parseOperand(addrRawOperand) || parser.parseRSquare() ||
      parser.parseKeyword("at") || parser.parseOperand(tstartRawOperand) ||
      parser.parseColon() || parser.parseLParen() ||
      parser.parseType(valueType) || parser.parseType(memType) ||
      parser.parseRParen())
    return failure();

  if (parser.resolveOperand(valueRawOperand, valueType, result.operands))
    return failure();
  if (parser.resolveOperand(memRawOperand, memType, result.operands))
    return failure();
  Type timeTypeVar = getTimeType(parser);
  Type intTypeVar = getIntegerType(parser, 32);
  if (parser.resolveOperand(addrRawOperand, intTypeVar, result.operands))
    return failure();
  if (parser.resolveOperand(tstartRawOperand, timeTypeVar, result.operands))
    return failure();

  // Right now the mem interface takes only one address. Thus there are two
  // parameters, the memory interface and the address.
  // TODO: Add support for arbitrary number of address parameters.
  auto numAddrOperandsAttr = parser.getBuilder().getIntegerAttr(intTypeVar, 1);
  result.attributes.set("num_addr_operands", numAddrOperandsAttr);
  return success();
}

/* ForOp.
 * Example:
 * hir.for %i = %l to %u step %s iter_time(%ti = %t){...}
 * hir.for %i = %l to %u step %s iter_time(%ti = %t tstep %n){...}
 */

static void printForOp(OpAsmPrinter &printer, ForOp op) {
  printer << "hir.for"
          << " " << op.getInductionVar() << " = " << op.lb() << " to "
          << op.ub() << " step " << op.step() << " iter_time( "
          << op.getIterTimeVar() << " = " << op.tstart();

  // print optional tstep.
  if (op.getAttrOfType<BoolAttr>("has_tstep").getValue())
    printer << " tstep " << op.tstep();
  printer << " ) ";

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  Type intTypeVar = getIntegerType(parser, 32);
  Type timeTypeVar = getTimeType(parser);
  Type lbRawType = intTypeVar;
  Type ubRawType = intTypeVar;
  Type stepRawType = intTypeVar;
  Type tstartRawType = timeTypeVar;
  Type tstepRawType = intTypeVar;
  Type regionRawOperandTypes[2];
  ArrayRef<Type> regionOperandTypes(regionRawOperandTypes);
  regionRawOperandTypes[1] = timeTypeVar;

  OpAsmParser::OperandType lbRawOperand;
  OpAsmParser::OperandType ubRawOperand;
  OpAsmParser::OperandType stepRawOperand;
  OpAsmParser::OperandType tstartRawOperand;
  OpAsmParser::OperandType tstepRawOperand;
  OpAsmParser::OperandType regionRawOperands[2];

  ArrayRef<OpAsmParser::OperandType> regionOperands(regionRawOperands);
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(regionRawOperands[0]) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  if (parser.parseOperand(lbRawOperand) || parser.parseKeyword("to") ||
      parser.parseOperand(ubRawOperand) || parser.parseKeyword("step") ||
      parser.parseOperand(stepRawOperand))
    return failure();

  // Parse iter time.
  if (parser.parseKeyword("iter_time") || parser.parseLParen() ||
      parser.parseRegionArgument(regionRawOperands[1]) || parser.parseEqual() ||
      parser.parseOperand(tstartRawOperand))
    return failure();

  // Parse optional tstep.
  bool hasTstep = false;
  if (!parser.parseOptionalKeyword("tstep")) {
    if (parser.parseOperand(tstepRawOperand))
      return failure();
    hasTstep = true;
  }
  auto hasTstepAttr = parser.getBuilder().getBoolAttr(hasTstep);
  result.attributes.set("has_tstep", hasTstepAttr);

  // Parse the type of induction variable.
  if (parser.parseRParen() || parser.parseColon() ||
      parser.parseType(regionRawOperandTypes[0]))
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

/* CallOp.
 * Example:
 * %z = hir.call @Foo (%x,%y) at %t
 *                : (!hir.int,!hir.mem_interface) -> (!hir.int)
 */

static LogicalResult verifyCallOp(CallOp op) {
  // TODO
}

/* DefOp
 * Example:
 * hir.def @foo(%x at %t, %y) at %t{}
 */

static ParseResult parseDefOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> args;
  OpAsmParser::OperandType tstartRawOperand;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parser.parseOperandList(args, OpAsmParser::Delimiter::Paren))
    return failure();
  if (parser.parseKeyword("at") || parser.parseOperand(tstartRawOperand))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon() || parser.parseLParen() ||
      parser.parseTypeList(argTypes) || parser.parseRParen() ||
      parser.parseArrow() || parser.parseLParen() ||
      parser.parseTypeList(resTypes) || parser.parseRParen())
    return failure();

  // Parse the optional function body.
  args.push_back(tstartRawOperand);
  printf("args.size = %d\n",args.size());
  argTypes.push_back(getTimeType(parser));
  printf("argTypes.size = %d\n",argTypes.size());
  auto type = builder.getFunctionType(argTypes, resTypes);
  result.addAttribute(impl::getTypeAttrName(), TypeAttr::get(type));
  auto *body = result.addRegion();
  if(parser.parseOptionalRegion(*body, args, argTypes))
    return failure();
  DefOp::ensureTerminator(*body, builder, result.location);
  return success();
}
static void printDefOp(OpAsmPrinter &printer, DefOp op) {}
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.cpp.inc"
