#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace hir;
using namespace llvm;

// Helper Methods.

static IntegerAttr getIntegerAttr(OpAsmParser &parser, int width, int value) {
  return IntegerAttr::get(
      IntegerType::get(parser.getBuilder().getContext(), width),
      APInt(width, value));
}

static Type getIntegerType(OpAsmParser &parser, int bitwidth) {
  return IntegerType::get(parser.getBuilder().getContext(), bitwidth);
}

static ConstType getConstIntType(OpAsmParser &parser) {
  return ConstType::get(parser.getBuilder().getContext());
}

static Type getTimeType(OpAsmParser &parser) {
  return TimeType::get(parser.getBuilder().getContext());
}

static ParseResult parseIntegerAttr(IntegerAttr &value, int bitwidth,
                                    StringRef attrName, OpAsmParser &parser,
                                    OperationState &result) {

  return parser.parseAttribute(value, getIntegerType(parser, bitwidth),
                               attrName, result.attributes);
}
