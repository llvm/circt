#ifndef PARSERHELPER
#define PARSERHELPER

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

static IntegerAttr getIntegerAttr(MLIRContext *context, int width, int value) {
  return IntegerAttr::get(IntegerType::get(context, width),
                          APInt(width, value));
}

static Type getIntegerType(MLIRContext *context, int bitwidth) {
  return IntegerType::get(context, bitwidth);
}

static ConstType getConstIntType(MLIRContext *context) {
  return ConstType::get(context);
}

static Type getTimeType(MLIRContext *context) { return TimeType::get(context); }

static ParseResult parseIntegerAttr(IntegerAttr &value, int bitwidth,
                                    StringRef attrName, OpAsmParser &parser,
                                    OperationState &result) {

  return parser.parseAttribute(
      value, getIntegerType(parser.getBuilder().getContext(), bitwidth),
      attrName, result.attributes);
}
#endif
