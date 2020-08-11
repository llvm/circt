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

// Helper Methods

static Type getIntegerType(OpAsmParser &parser, int bitwidth) {
  return parser.getBuilder().getIntegerType(bitwidth);
}

static Type getTimeType(OpAsmParser &parser) {
  return TimeType::get(parser.getBuilder().getContext());
}
// MemReadOp
static void printMemReadOp(OpAsmPrinter &printer, MemReadOp op) {
  printer << "hir.mem_read"
          << " " << op.mem() << "[" << op.addr() << "]"
          << " at " << op.at() << " : " << op.mem().getType() << " -> "
          << op.res().getType();
}

static ParseResult parseMemReadOp(OpAsmParser &parser, OperationState &result) {
  Type memType;
  Type resRawType;
  OpAsmParser::OperandType memRawOperand;
  OpAsmParser::OperandType addrRawOperand;
  OpAsmParser::OperandType atRawOperand;
  if (parser.parseOperand(memRawOperand) || parser.parseLSquare() ||
      parser.parseOperand(addrRawOperand) || parser.parseRSquare() ||
      parser.parseKeyword("at") || parser.parseOperand(atRawOperand) ||
      parser.parseColon() || parser.parseType(memType) || parser.parseArrow() ||
      parser.parseType(resRawType))
    return failure();

  if (parser.resolveOperand(memRawOperand, memType, result.operands))
    return failure();
  Type odsBuildableTimeType = getTimeType(parser);
  Type odsBuildableI32Type = getIntegerType(parser, 32);
  if (parser.resolveOperand(addrRawOperand, odsBuildableI32Type,
                            result.operands))
    return failure();
  if (parser.resolveOperand(atRawOperand, odsBuildableTimeType,
                            result.operands))
    return failure();

  auto num_input_operands_attr =
      parser.getBuilder().getIntegerAttr(odsBuildableI32Type, 2);
  result.attributes.set("num_input_operands", num_input_operands_attr);
  result.addTypes(resRawType);
}

// ForOp
static void printForOp(OpAsmPrinter &printer, ForOp op) {
  printer << "hir.for"
          << " " << op.getInductionVar() << " = " << op.lb() << " to "
          << op.ub() << " step " << op.step() << " iter_time( "
          << op.getIterTimeVar() << " = " << op.at();

  // print optional tstep
  if (op.getAttrOfType<BoolAttr>("hasTstepAttr").getValue())
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
  Type atRawType = timeTypeVar;
  Type tstepRawType = intTypeVar;
  Type regionRawOperandTypes[2];
  ArrayRef<Type> regionOperandTypes(regionRawOperandTypes);
  regionRawOperandTypes[1] = timeTypeVar;

  OpAsmParser::OperandType lbRawOperand;
  OpAsmParser::OperandType ubRawOperand;
  OpAsmParser::OperandType stepRawOperand;
  OpAsmParser::OperandType atRawOperand;
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

  if (parser.parseKeyword("iter_time") || parser.parseLParen() ||
      parser.parseRegionArgument(regionRawOperands[1]) || parser.parseEqual() ||
      parser.parseOperand(atRawOperand))
    return failure();

  bool hasTstep = false;
  if (!parser.parseOptionalKeyword("tstep")) {
    if (parser.parseOperand(tstepRawOperand))
      return failure();
    hasTstep = true;
  }
  auto hasTstepAttr = parser.getBuilder().getBoolAttr(hasTstep);
  result.attributes.set("hasTstepAttr", hasTstepAttr);

  if (parser.parseRParen() || parser.parseColon() ||
      parser.parseType(regionRawOperandTypes[0]))
    return failure();

  if (parser.resolveOperand(lbRawOperand, lbRawType, result.operands) ||
      parser.resolveOperand(ubRawOperand, ubRawType, result.operands) ||
      parser.resolveOperand(stepRawOperand, stepRawType, result.operands) ||
      parser.resolveOperand(atRawOperand, atRawType, result.operands))
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

#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.cpp.inc"
