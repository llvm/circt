//===- HandshakeOps.cpp - Handshake MLIR Operations -----------------------===//
//
// Copyright 2019 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// =============================================================================

#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;

#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace handshake {
#include "circt/Dialect/Handshake/HandshakeOps.inc"
}
} // namespace circt

//===----------------------------------------------------------------------===//
// HandshakeOpsDialect
//===----------------------------------------------------------------------===//

HandshakeOpsDialect::HandshakeOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              ::mlir::TypeID::get<HandshakeOpsDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Handshake/HandshakeOps.cpp.inc"
      >();
}

void ForkOp::build(OpBuilder &builder, OperationState &result, Value operand,
                   int outputs) {

  auto type = operand.getType();

  // Fork has results as many as there are successor ops
  result.types.append(outputs, type);

  // Single operand
  result.addOperands(operand);

  // Fork is control-only if it is the no-data output of a ControlMerge or a
  // StartOp
  auto *op = operand.getDefiningOp();
  bool isControl = ((dyn_cast<ControlMergeOp>(op) || dyn_cast<StartOp>(op)) &&
                    operand == op->getResult(0))
                       ? true
                       : false;
  result.addAttribute("control", builder.getBoolAttr(isControl));
}
void handshake::ForkOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleForksPattern>(context);
}

void LazyForkOp::build(OpBuilder &builder, OperationState &result,
                       Value operand, int outputs) {

  auto type = operand.getType();

  // Fork has results as many as there are successor ops
  result.types.append(outputs, type);

  // Single operand
  result.addOperands(operand);

  // Fork is control-only if it is the no-data output of a ControlMerge or a
  // StartOp
  auto *op = operand.getDefiningOp();
  bool isControl = ((dyn_cast<ControlMergeOp>(op) || dyn_cast<StartOp>(op)) &&
                    operand == op->getResult(0))
                       ? true
                       : false;
  result.addAttribute("control", builder.getBoolAttr(isControl));
}

void MergeOp::build(OpBuilder &builder, OperationState &result, Value operand,
                    int inputs) {

  auto type = operand.getType();
  result.types.push_back(type);

  // Operand to keep defining value (used when connecting merges)
  // Removed afterwards
  result.addOperands(operand);

  // Operands from predecessor blocks
  for (int i = 0, e = inputs; i < e; ++i)
    result.addOperands(operand);
}

void MergeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleMergesPattern>(context);
}

void MuxOp::build(OpBuilder &builder, OperationState &result, Value operand,
                  int inputs) {

  auto type = operand.getType();
  result.types.push_back(type);

  // Operand connected to ControlMerge from same block
  result.addOperands(operand);

  // Operands from predecessor blocks
  for (int i = 0, e = inputs; i < e; ++i)
    result.addOperands(operand);
}

static LogicalResult verify(MuxOp op) {
  unsigned numDataOperands = static_cast<int>(op.dataOperands().size());
  if (numDataOperands < 2)
    return op.emitError("need at least two inputs to mux");

  auto selectType = op.selectOperand().getType();

  unsigned selectBits;
  if (auto integerType = selectType.dyn_cast<IntegerType>())
    selectBits = integerType.getWidth();
  else if (selectType.isIndex())
    selectBits = IndexType::kInternalStorageBitWidth;
  else
    return op.emitError("unsupported type for select operand: ") << selectType;

  double maxDataOperands = std::pow(2, selectBits);
  if (numDataOperands > maxDataOperands)
    return op.emitError("select bitwidth was ")
           << selectBits << ", which can mux "
           << static_cast<int64_t>(maxDataOperands) << " operands, but found "
           << numDataOperands << " operands";

  return success();
}

void ControlMergeOp::build(OpBuilder &builder, OperationState &result,
                           Value operand, int inputs) {

  auto type = operand.getType();
  result.types.push_back(type);
  // Second result gives the input index to the muxes
  // Number of bits depends on encoding (log2/1-hot)
  result.types.push_back(builder.getIndexType());

  // Operand to keep defining value (used when connecting merges)
  // Removed afterwards
  result.addOperands(operand);

  // Operands from predecessor blocks
  for (int i = 0, e = inputs; i < e; ++i)
    result.addOperands(operand);

  result.addAttribute("control", builder.getBoolAttr(true));
}
// void ControlMergeOp::getCanonicalizationPatterns(OwningRewritePatternList
// &results,
//                                           MLIRContext *context) {
//   results.insert<circt::handshake::EliminateSimpleControlMergesPattern>(context);
// }

void handshake::BranchOp::build(OpBuilder &builder, OperationState &result,
                                Value dataOperand) {

  auto type = dataOperand.getType();
  result.types.push_back(type);
  result.addOperands(dataOperand);

  // Branch is control-only if it is the no-data output of a ControlMerge or a
  // StartOp This holds because Branches are inserted before Forks
  auto *op = dataOperand.getDefiningOp();
  bool isControl = ((dyn_cast<ControlMergeOp>(op) || dyn_cast<StartOp>(op)) &&
                    dataOperand == op->getResult(0))
                       ? true
                       : false;
  result.addAttribute("control", builder.getBoolAttr(isControl));
}
void handshake::BranchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleBranchesPattern>(context);
}

void handshake::ConditionalBranchOp::build(OpBuilder &builder,
                                           OperationState &result,
                                           Value condOperand,
                                           Value dataOperand) {

  auto type = dataOperand.getType();
  result.types.append(2, type);
  result.addOperands(condOperand);
  result.addOperands(dataOperand);

  // Branch is control-only if it is the no-data output of a ControlMerge or a
  // StartOp This holds because Branches are inserted before Forks
  auto *op = dataOperand.getDefiningOp();
  bool isControl = ((dyn_cast<ControlMergeOp>(op) || dyn_cast<StartOp>(op)) &&
                    dataOperand == op->getResult(0))
                       ? true
                       : false;
  result.addAttribute("control", builder.getBoolAttr(isControl));
}

void StartOp::build(OpBuilder &builder, OperationState &result) {
  // Control-only output, has no type
  auto type = builder.getNoneType();
  result.types.push_back(type);
  result.addAttribute("control", builder.getBoolAttr(true));
}

void EndOp::build(OpBuilder &builder, OperationState &result, Value operand) {

  result.addOperands(operand);
}

void handshake::ReturnOp::build(OpBuilder &builder, OperationState &result,
                                ArrayRef<Value> operands) {

  result.addOperands(operands);
}

void SinkOp::build(OpBuilder &builder, OperationState &result, Value operand) {

  result.addOperands(operand);
}

void handshake::ConstantOp::build(OpBuilder &builder, OperationState &result,
                                  Attribute value, Value operand) {

  result.addOperands(operand);

  auto type = value.getType();
  result.types.push_back(type);

  result.addAttribute("value", value);
}

void handshake::TerminatorOp::build(OpBuilder &builder, OperationState &result,
                                    ArrayRef<Block *> successors) {
  // Add all the successor blocks of the block which contains this terminator
  result.addSuccessors(successors);
  // for (auto &succ : successors)
  //   result.addSuccessor(succ, {});
}

void MemoryOp::build(OpBuilder &builder, OperationState &result,
                     ArrayRef<Value> operands, int outputs, int control_outputs,
                     bool lsq, int id, Value memref) {

  result.addOperands(operands);

  auto memrefType = memref.getType().cast<MemRefType>();

  // Data outputs (get their type from memref)
  result.types.append(outputs, memrefType.getElementType());

  // Control outputs
  result.types.append(control_outputs, builder.getNoneType());

  // Indicates whether a memory is an LSQ
  result.addAttribute("lsq", builder.getBoolAttr(lsq));

  // Memref info
  result.addAttribute("type", TypeAttr::get(memrefType));

  // Memory ID (individual ID for each MemoryOp)
  Type i32Type = builder.getIntegerType(32);
  result.addAttribute("id", builder.getIntegerAttr(i32Type, id));

  if (!lsq) {

    result.addAttribute("ld_count", builder.getIntegerAttr(i32Type, outputs));
    result.addAttribute(
        "st_count", builder.getIntegerAttr(i32Type, control_outputs - outputs));
  }
}

void handshake::LoadOp::build(OpBuilder &builder, OperationState &result,
                              Value memref, ArrayRef<Value> indices) {

  // Address indices
  // result.addOperands(memref);
  result.addOperands(indices);

  // Data type
  auto memrefType = memref.getType().cast<MemRefType>();

  // Data output (from load to successor ops)
  result.types.push_back(memrefType.getElementType());

  // Address outputs (to lsq)
  result.types.append(indices.size(), builder.getIndexType());
}

void handshake::StoreOp::build(OpBuilder &builder, OperationState &result,
                               Value valueToStore, ArrayRef<Value> indices) {

  // Data
  result.addOperands(valueToStore);

  // Address indices
  result.addOperands(indices);

  // Data output (from store to LSQ)
  result.types.push_back(valueToStore.getType());

  // Address outputs (from store to lsq)
  result.types.append(indices.size(), builder.getIndexType());
}

void JoinOp::build(OpBuilder &builder, OperationState &result,
                   ArrayRef<Value> operands) {

  auto type = builder.getNoneType();
  result.types.push_back(type);

  result.addOperands(operands);

  result.addAttribute("control", builder.getBoolAttr(true));
}

// for let printer/parser/verifier in Handshake_Op class
/*static LogicalResult verify(ForkOp op) {
  return success();
}
void print(OpAsmPrinter &p, ForkOp op) {
  p << "handshake.fork ";
  p.printOperands(op.getOperands());
 // p << " : " << op.getType();
}
ParseResult parseForkOp(OpAsmParser &parser, OperationState &result) {
  return success();
}*/

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

// Code below is largely duplicated from Standard/Ops.cpp
static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

static void print(OpAsmPrinter &p, handshake::ReturnOp op) {
  p << "handshake.return";
  if (op.getNumOperands() != 0) {
    p << ' ';
    p.printOperands(op.getOperands());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

static LogicalResult verify(handshake::ReturnOp op) {
  auto *parent = op.getParentOp();
  auto function = dyn_cast<handshake::FuncOp>(parent);
  if (!function)
    return op.emitOpError("must have a handshake.func parent");

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ")
           << op.getNumOperands()
           << " operands, but enclosing function returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i).getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i).getType()
             << ") doesn't match function result type (" << results[i] << ")";

  return success();
}

namespace circt {
namespace handshake {

#include "circt/Dialect/Handshake/HandshakeInterfaces.cpp.inc"

} // namespace handshake
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/Handshake/HandshakeOps.cpp.inc"
