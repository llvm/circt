#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace hir;
// Operations
static void print(OpAsmPrinter &printer, MemReadOp op) {
  printer << "hir.mem_read"
          << " " << op.mem() << "[" << op.addr() << "]"
          << " at " << op.at() << " : " << op.mem().getType() << " -> "
          << op.res().getType();
}

static ParseResult parseMemReadOp(OpAsmParser &parser,
                                    OperationState &result) {
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
  Type odsBuildableTimeType = TimeType::get(parser.getBuilder().getContext());
  Type odsBuildableI32Type = parser.getBuilder().getIntegerType(32);
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

static ParseResult parse_For_Op(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType inductionVariable;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return failure();

  // Parse loop bounds.

  Type idxRawType;
  OpAsmParser::OperandType lbRawOperand;
  OpAsmParser::OperandType ubRawOperand;
  OpAsmParser::OperandType stepRawOperand;

  if (parser.parseOperand(lbRawOperand) || parser.parseKeyword("to") ||
      parser.parseOperand(ubRawOperand) || parser.parseKeyword("step") ||
      parser.parseOperand(stepRawOperand) || parser.parseColon() ||
      parser.parseType(idxRawType))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVariable, idxRawType))
    return failure();
  For_Op::ensureTerminator(*body, builder, result.location);
  return success();
}
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.cpp.inc"
