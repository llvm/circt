//===- ESIOps.cpp - ESI op code defs ----------------------------*- C++ -*-===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/RTL/Ops.h"

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

static LogicalResult verifyInstantiatedOp(InstantiatedOp op) {
  auto moduleIR = op.getParentOfType<SnippetOp>();
  if (moduleIR == nullptr) {
    op.emitError("Must be contained within a snippet region");
    return failure();
  }
  auto referencedModule =
      mlir::SymbolTable::lookupSymbolIn(moduleIR, op.moduleName());
  if (referencedModule == nullptr) {
    op.emitError("Cannot find module definition '") << op.moduleName() << "'.";
    return failure();
  }
  if (!isa<circt::rtl::RTLExternModuleOp>(referencedModule)) {
    op.emitError("Symbol resolved to '")
        << referencedModule->getName() << "' which is not a RTLExternModuleOp.";
    return failure();
  }
  return success();
}
#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.cpp.inc"
