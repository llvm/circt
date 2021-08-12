//=========- HIR.cpp - Registration, Parser & Printer----------------------===//
//
// This file implements parsers and printers for Types and registers the types
// and operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;
using namespace hir;

void HIRDialect::initialize() {
  addTypes<TimeType, BusType, FuncType, MemrefType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/IR/HIR.cpp.inc"
      >();
}

Operation *HIRDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  return builder.create<mlir::ConstantOp>(loc, type, value);
}

#include "circt/Dialect/HIR/IR/HIRDialect.cpp.inc"
