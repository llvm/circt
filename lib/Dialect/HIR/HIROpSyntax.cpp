#include "HIROpSyntax.h"
#include "helper.h"

namespace helper {} // namespace helper.

// parser and printer for Time and offset.
ParseResult
parseTimeAndOffset(mlir::OpAsmParser &parser, OpAsmParser::OperandType &tstart,
                   llvm::Optional<OpAsmParser::OperandType> &varOffset,
                   IntegerAttr &constOffset) {

  constOffset = getIntegerAttr(parser, 32, 0);
  if (parser.parseOperand(tstart))
    return failure();

  // early exit if no offsets.
  if (parser.parseOptionalPlus())
    return success();

  OpAsmParser::OperandType tempOffset;
  if (succeeded(parser.parseOptionalOperand(tempOffset)))
    return success();
}

void printTimeAndOffset(OpAsmPrinter &printer, Operation *op, Value tstart,
                        Value offset) {}

// parser and printer for array address types.
ParseResult parseArrayAccessTypes(mlir::OpAsmParser &parser,
                                  SmallVectorImpl<Type> &addrTypes) {

  return success();
}

void printArrayAccessTypes(OpAsmPrinter &printer, Operation *op,
                           TypeRange addrTypes) {}
