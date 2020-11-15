//===- ESIOps.cpp - ESI op code defs ----------------------------*- C++ -*-===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace circt::esi;

ParseResult parseChannelBuffer(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  OpAsmParser::OperandType inputOperand;
  if (parser.parseOperand(inputOperand))
    return failure();

  ChannelBufferOptions optionsAttr;
  if (parser.parseAttribute(optionsAttr,
                            parser.getBuilder().getType<NoneType>(), "options",
                            result.attributes))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  Type innerOutputType;
  if (parser.parseType(innerOutputType))
    return failure();
  auto outputType =
      ChannelPort::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({outputType});

  if (parser.resolveOperands({inputOperand}, {outputType}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void print(OpAsmPrinter &p, ChannelBuffer &op) {
  p << "esi.buffer " << op.input() << " ";
  p.printAttributeWithoutType(op.options());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"options"});
  p << " : " << op.output().getType().cast<ChannelPort>().getInner();
}

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.cpp.inc"
