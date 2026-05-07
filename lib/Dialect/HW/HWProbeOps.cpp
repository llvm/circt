//===- HWProbeOps.cpp - Implement the HW probe operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HW probe operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace circt::hw;

//===----------------------------------------------------------------------===//
// UninitializedProbeConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult UninitializedProbeConstantOp::fold(FoldAdaptor adaptor) {
  // Always fold to itself - this is a constant operation
  return getResult();
}

//===----------------------------------------------------------------------===//
// ProbeSendOp
//===----------------------------------------------------------------------===//

ParseResult ProbeSendOp::parse(OpAsmParser &parser, OperationState &result) {
  // Check for optional 'forceable' keyword
  bool forceable = succeeded(parser.parseOptionalKeyword("forceable"));
  if (forceable)
    result.addAttribute("forceable", parser.getBuilder().getUnitAttr());

  OpAsmParser::UnresolvedOperand input;
  Type inputType;

  if (parser.parseOperand(input))
    return failure();

  // Parse optional 'sym' keyword for inner symbol
  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    InnerSymAttr innerSym;
    if (parser.parseCustomAttributeWithFallback(innerSym))
      return failure();
    result.addAttribute(InnerSymbolTable::getInnerSymbolAttrName(), innerSym);
  }

  if (parser.parseColon() || parser.parseType(inputType))
    return failure();

  // Resolve the operand
  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();

  // The result type depends on whether forceable is present
  // The inner type is the same as the input type (no special conversion here)
  Type resultType = forceable ? (Type)RWProbeType::get(inputType)
                              : (Type)ProbeType::get(inputType);
  result.addTypes(resultType);

  return success();
}

void ProbeSendOp::print(OpAsmPrinter &p) {
  // Print 'forceable' keyword if the attribute is present
  if (getForceable())
    p << " forceable";

  p << " " << getInput();

  // Print 'sym' keyword and inner symbol if present
  if (auto innerSym = getInnerSymAttr()) {
    p << " sym ";
    innerSym.print(p);
  }

  // Print attributes excluding 'forceable', 'inner_sym' (already printed)
  SmallVector<StringRef> elidedAttrs;
  elidedAttrs.push_back("forceable");
  elidedAttrs.push_back(InnerSymbolTable::getInnerSymbolAttrName());
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Print the input operand type
  p << " : " << getInput().getType();
}

OpFoldResult ProbeSendOp::fold(FoldAdaptor adaptor) {
  // No constant folding for now
  return {};
}

std::optional<size_t> ProbeSendOp::getTargetResultIndex() {
  // Inner symbol targets the result (the probe)
  return 0;
}

// LogicalResult ProbeSendOp::verify() {
//   Type inputType = getInput().getType();
//     return success();

//   // Allow HW value types
//   if (isHWValueType(inputType) || isa<seq::ClockType>(inputType))
//     return success();

//   return emitOpError("input must be a valid HW value type, got ") <<
//   inputType;
// }

//===----------------------------------------------------------------------===//
// ProbeResolveOp
//===----------------------------------------------------------------------===//

ParseResult ProbeResolveOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand ref;
  Type refType;

  if (parser.parseOperand(ref) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(refType))
    return failure();

  // Resolve the operand
  if (parser.resolveOperand(ref, refType, result.operands))
    return failure();

  // Extract the inner type from the probe type
  auto probeType = dyn_cast<ProbeType>(refType);
  if (!probeType) {
    auto rwProbeType = dyn_cast<RWProbeType>(refType);
    if (!rwProbeType)
      return parser.emitError(parser.getNameLoc(), "expected probe type");
    result.addTypes(rwProbeType.getInnerType());
  } else {
    result.addTypes(probeType.getInnerType());
  }

  return success();
}

void ProbeResolveOp::print(OpAsmPrinter &p) {
  p << " " << getRef();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getRef().getType();
}

//===----------------------------------------------------------------------===//
// ProbeSubOp
//===----------------------------------------------------------------------===//

ParseResult ProbeSubOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType;
  IntegerAttr indexAttr;

  if (parser.parseOperand(input) || parser.parseLSquare() ||
      parser.parseAttribute(indexAttr) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return failure();

  // Add the index attribute
  result.addAttribute("index", indexAttr);

  // Resolve the operand
  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();

  // Compute the result type by indexing into the probe type
  Type innerType;
  if (auto probeType = dyn_cast<ProbeType>(inputType)) {
    innerType = probeType.getInnerType();
  } else if (auto rwProbeType = dyn_cast<RWProbeType>(inputType)) {
    innerType = rwProbeType.getInnerType();
  } else {
    return parser.emitError(parser.getNameLoc(), "expected probe type");
  }

  // Index into the inner type
  Type elemType;
  if (auto arrayType = dyn_cast<hw::ArrayType>(innerType)) {
    elemType = arrayType.getElementType();
  } else if (auto structType = dyn_cast<hw::StructType>(innerType)) {
    auto index = indexAttr.getInt();
    if (index < 0 ||
        static_cast<size_t>(index) >= structType.getElements().size())
      return parser.emitError(parser.getNameLoc(),
                              "struct index out of bounds");
    elemType = structType.getElements()[index].type;
  } else {
    return parser.emitError(parser.getNameLoc(),
                            "can only index into array or struct");
  }

  // Result is a probe of the element type
  if (isa<ProbeType>(inputType))
    result.addTypes(ProbeType::get(elemType));
  else
    result.addTypes(RWProbeType::get(elemType));

  return success();
}

void ProbeSubOp::print(OpAsmPrinter &p) {
  p << " " << getInput() << "[";
  p.printAttribute(getIndexAttr());
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
  p << " : " << getInput().getType();
}

//===----------------------------------------------------------------------===//
// ProbeCastOp
//===----------------------------------------------------------------------===//

ParseResult ProbeCastOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType, resultType;

  if (parser.parseOperand(input) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  // Resolve the operand
  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();

  result.addTypes(resultType);
  return success();
}

void ProbeCastOp::print(OpAsmPrinter &p) {
  p << " " << getInput();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getInput().getType() << " -> " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// ProbeCastOp Folding
//===----------------------------------------------------------------------===//

OpFoldResult ProbeCastOp::fold(FoldAdaptor adaptor) {
  // If input and result types are the same, fold away the cast
  if (getInput().getType() == getResult().getType())
    return getInput();
  return {};
}

void ProbeCastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // Remove redundant casts
  struct RemoveIdentityCast : public OpRewritePattern<ProbeCastOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(ProbeCastOp op,
                                  PatternRewriter &rewriter) const override {
      if (op.getInput().getType() == op.getResult().getType()) {
        rewriter.replaceOp(op, op.getInput());
        return success();
      }
      return failure();
    }
  };

  results.add<RemoveIdentityCast>(context);
}

//===----------------------------------------------------------------------===//
// XMRRemoteOp
//===----------------------------------------------------------------------===//

void XMRRemoteOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Auto-generate SSA name from the referenced path
  setNameFn(getResult(), "xmr");
}

hw::HierPathOp XMRRemoteOp::getReferencedPath(const hw::HWSymbolCache *cache) {
  if (cache)
    if (auto *result = cache->getDefinition(getRefAttr().getAttr()))
      return cast<hw::HierPathOp>(result);

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol<hw::HierPathOp>(getRefAttr().getValue());
}

LogicalResult XMRRemoteOp::verify() {
  // Basic verification - the symbol verification is done in verifySymbolUses
  return success();
}

LogicalResult
XMRRemoteOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *table = SymbolTable::getNearestSymbolTable(*this);
  auto path = dyn_cast_or_null<hw::HierPathOp>(
      symbolTable.lookupSymbolIn(table, getRefAttr()));
  if (!path)
    return emitError("Referenced path doesn't exist: ") << getRefAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// ProbeRWProbeOp
//===----------------------------------------------------------------------===//

ParseResult ProbeRWProbeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse optional 'sym' keyword for inner symbol
  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    InnerSymAttr innerSym;
    if (parser.parseCustomAttributeWithFallback(innerSym))
      return failure();
    result.addAttribute(InnerSymbolTable::getInnerSymbolAttrName(), innerSym);
  }

  // Parse the symbol reference  (@Module::@symbol)
  mlir::SymbolRefAttr symRef;
  Type resultType;

  if (parser.parseAttribute(symRef) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(resultType))
    return failure();

  // Convert SymbolRefAttr to InnerRefAttr
  if (symRef.getNestedReferences().size() != 1)
    return parser.emitError(parser.getNameLoc(),
                            "expected @module::@symbol format");

  auto innerRef =
      InnerRefAttr::get(symRef.getRootReference(), symRef.getLeafReference());
  result.addAttribute("target", innerRef);
  result.addTypes(resultType);

  return success();
}

void ProbeRWProbeOp::print(OpAsmPrinter &p) {
  // Print optional inner symbol
  if (auto innerSym = (*this)->getAttrOfType<InnerSymAttr>(
          InnerSymbolTable::getInnerSymbolAttrName())) {
    p << " sym ";
    innerSym.print(p);
  }

  // Print as @Module::@symbol
  auto target = getTargetAttr();
  p << " @" << target.getModule().getValue() << "::@"
    << target.getName().getValue();

  // Print attributes excluding 'target', 'inner_sym' (already printed), and
  // 'dummy' (internal parsing artifact)
  SmallVector<StringRef> elidedAttrs;
  elidedAttrs.push_back("target");
  elidedAttrs.push_back(InnerSymbolTable::getInnerSymbolAttrName());
  elidedAttrs.push_back("dummy");
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  p << " : " << getResult().getType();
}

LogicalResult ProbeRWProbeOp::verifyInnerRefs(InnerRefNamespace &ns) {
  // Verify that the target inner reference is valid
  auto target = getTargetAttr();
  if (!target)
    return emitOpError("requires a target inner reference");

  // The actual verification of the inner ref is done by the namespace
  return success();
}

std::optional<size_t> ProbeRWProbeOp::getTargetResultIndex() {
  // Inner symbol targets the result (the rwprobe)
  return 0;
}

//===----------------------------------------------------------------------===//
// ProbeDefineOp
//===----------------------------------------------------------------------===//

LogicalResult ProbeDefineOp::verify() {
  // Check that destination is not a result of certain operations
  // that shouldn't be on the left-hand side of a define
  if (auto *op = getDest().getDefiningOp()) {
    // Cannot define to a sub-element of a probe
    if (isa<ProbeSubOp>(op))
      return emitError(
          "destination reference cannot be a sub-element of a reference");

    // Cannot define to a cast
    if (isa<ProbeCastOp>(op))
      return emitError(
          "destination reference cannot be a cast of another reference");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ProbeToInOutOp
//===----------------------------------------------------------------------===//

ParseResult ProbeToInOutOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand ref;
  Type refType;

  if (parser.parseOperand(ref) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(refType))
    return failure();

  // Resolve the operand
  if (parser.resolveOperand(ref, refType, result.operands))
    return failure();

  // Extract the inner type from the probe type and create InOutType
  auto probeType = dyn_cast<ProbeType>(refType);
  if (!probeType) {
    auto rwProbeType = dyn_cast<RWProbeType>(refType);
    if (!rwProbeType)
      return parser.emitError(parser.getNameLoc(),
                              "expected probe or rwprobe type");
    result.addTypes(InOutType::get(rwProbeType.getInnerType()));
  } else {
    result.addTypes(InOutType::get(probeType.getInnerType()));
  }

  return success();
}

void ProbeToInOutOp::print(OpAsmPrinter &p) {
  p << " " << getRef();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getRef().getType();
}

LogicalResult ProbeToInOutOp::verify() {
  // Get the probe type from the input
  auto probeType = dyn_cast<ProbeType>(getRef().getType());
  auto rwProbeType = dyn_cast<RWProbeType>(getRef().getType());

  if (!probeType && !rwProbeType)
    return emitOpError("input must be a probe or rwprobe type");

  // Extract inner type from the probe
  Type innerType =
      probeType ? probeType.getInnerType() : rwProbeType.getInnerType();

  // Verify that the result type is an inout of the inner type
  auto inoutType = dyn_cast<InOutType>(getResult().getType());
  if (!inoutType)
    return emitOpError("result must be an inout type");

  if (inoutType.getElementType() != innerType)
    return emitOpError("result inout element type must match probe inner type");

  return success();
}
