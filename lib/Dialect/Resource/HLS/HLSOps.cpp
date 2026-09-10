//===- HLSOps.cpp - HLS op implementations --------------------------------===//

#include "circt/Dialect/Resource/HLS/HLSOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"   // <- OpAsmParser / OpAsmPrinter full defs
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/IntegerSet.h"                                // IntegerSet complete type
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"   // the interface
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Resource/HLS/HLS.cpp.inc"

using namespace circt;
using namespace circt::hls_analysis;
using namespace mlir;

//===----------------------------------------------------------------------===//
// StoreEnableOp
//===----------------------------------------------------------------------===//

// Unconditional Write on the memref operand. The predicate is an operand, not
// an effect -- this is what lets the BRAM estimator count one write / zero
// reads, and what keeps DCE/CSE from deleting the op.
void StoreEnableOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getMemrefMutable(),
                       SideEffects::DefaultResource::get());
}

LogicalResult StoreEnableOp::verify() {
  auto memrefTy = llvm::cast<MemRefType>(getMemref().getType());

  if (static_cast<int64_t>(getIndices().size()) != memrefTy.getRank())
    return emitOpError("expected ")
           << memrefTy.getRank() << " indices for memref of rank "
           << memrefTy.getRank() << ", got " << getIndices().size();

  if (getValue().getType() != memrefTy.getElementType())
    return emitOpError("value type ")
           << getValue().getType()
           << " does not match memref element type "
           << memrefTy.getElementType();

  return success();
}

//===----------------------------------------------------------------------===//
// AffineStoreEnableOp
//===----------------------------------------------------------------------===//

void AffineStoreEnableOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getMemrefMutable(),
                       SideEffects::DefaultResource::get());
}

LogicalResult AffineStoreEnableOp::verify() {
  auto memrefTy = llvm::cast<MemRefType>(getMemref().getType());

  // map result count must equal memref rank
  if (getMap().getNumResults() != static_cast<unsigned>(memrefTy.getRank()))
    return emitOpError("affine map result count (")
           << getMap().getNumResults() << ") must equal memref rank ("
           << memrefTy.getRank() << ")";

  // map operand count must match the map's input count
  if (getMap().getNumInputs() != getMapOperands().size())
    return emitOpError("affine map expects ")
           << getMap().getNumInputs() << " operands, got "
           << getMapOperands().size();

  // set operand count must match the integer set's input count
  if (getCondition().getNumInputs() != getSetOperands().size())
    return emitOpError("integer set expects ")
           << getCondition().getNumInputs() << " operands, got "
           << getSetOperands().size();

  // stored value type must match element type
  if (getValue().getType() != memrefTy.getElementType())
    return emitOpError("value type ")
           << getValue().getType() << " does not match element type "
           << memrefTy.getElementType();

  return success();
}


ParseResult AffineStoreEnableOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  // --- value ',' ---
  OpAsmParser::UnresolvedOperand valueOperand;
  if (parser.parseOperand(valueOperand) || parser.parseComma())
    return failure();

  // --- 'if' #set ( setOperands ) ',' ---
  IntegerSetAttr setAttr;
  SmallVector<OpAsmParser::UnresolvedOperand> setOperands;
  if (parser.parseKeyword("if") ||
      parser.parseAttribute(
          setAttr, AffineStoreEnableOp::getConditionAttrName(result.name),
          result.attributes))
    return failure();
  IntegerSet set = setAttr.getValue();
  if (parser.parseOperandList(setOperands, set.getNumInputs(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.parseComma())
    return failure();

  // --- memref [ #map ( mapOperands ) ] ---
  OpAsmParser::UnresolvedOperand memrefOperand;
  if (parser.parseOperand(memrefOperand))
    return failure();

  AffineMapAttr mapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand> mapOperands;
  if (parser.parseAffineMapOfSSAIds(
          mapOperands, mapAttr,
          AffineStoreEnableOp::getMapAttrName(result.name).getValue(),
          result.attributes, OpAsmParser::Delimiter::Square))
    return failure();

  // --- ':' memref-type ---
  Type memrefType;
  if (parser.parseColonType(memrefType))
    return failure();
  auto mt = dyn_cast<MemRefType>(memrefType);
  if (!mt)
    return parser.emitError(parser.getNameLoc(), "expected memref type");

  // --- resolve operands in declared order: value, memref, map, set ---
  if (parser.resolveOperand(valueOperand, mt.getElementType(),
                            result.operands) ||
      parser.resolveOperand(memrefOperand, mt, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands) ||
      parser.resolveOperands(setOperands, indexTy, result.operands))
    return failure();

  // --- operandSegmentSizes: {value=1, memref=1, #map, #set} ---
  result.addAttribute(
      AffineStoreEnableOp::getOperandSegmentSizesAttrName(result.name),
      builder.getDenseI32ArrayAttr(
          {1, 1, static_cast<int32_t>(mapOperands.size()),
           static_cast<int32_t>(setOperands.size())}));

  return success();
}

void AffineStoreEnableOp::print(OpAsmPrinter &p) {
  p << ' ' << getValue() << ", if " << getConditionAttr() << '(';
  p.printOperands(getSetOperands());
  p << "), " << getMemref() << '[';
  p.printAffineMapOfSSAIds(getMapAttr(), getMapOperands());
  p << "] : " << getMemref().getType();
}