#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"


namespace mlir {
namespace hir {
// Operations
static void print(mlir::OpAsmPrinter &printer, Mem_Read_Op op) {
  printer << "hir.mem_read"
          << " " << op.mem() << "[" << op.addr() << "]"
          << " at " <<op.at()
          << " : " << op.mem().getType() <<"->" <<op.res().getType();
}

static mlir::ParseResult parseMem_Read_Op(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  mlir::Type memType;
  mlir::Type resRawTypes[1];
  llvm::ArrayRef<mlir::Type> resTypes(resRawTypes);
  mlir::OpAsmParser::OperandType memRawOperand;
  mlir::OpAsmParser::OperandType addrRawOperand;
  mlir::OpAsmParser::OperandType atRawOperand;
  if (parser.parseOperand(memRawOperand) || parser.parseLSquare() ||
      parser.parseOperand(addrRawOperand) || parser.parseRSquare() ||
      parser.parseKeyword("at") ||
      parser.parseOperand(atRawOperand) ||
      parser.parseColon() || parser.parseType(memType) || parser.parseArrow() ||
      parser.parseType(resRawTypes[0]))
    return mlir::failure();

  if (parser.resolveOperand(memRawOperand, memType, result.operands))
    return mlir::failure();
  mlir::Type odsBuildableTimeType = TimeType::get(parser.getBuilder().getContext());
  mlir::Type odsBuildableI32Type = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperand(addrRawOperand,odsBuildableI32Type, result.operands))
    return mlir::failure();
  if (parser.resolveOperand(atRawOperand, odsBuildableTimeType, result.operands))
    return mlir::failure();

  result.addTypes(resTypes);
}

#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.cpp.inc"
} // namespace hir
} // namespace mlir
