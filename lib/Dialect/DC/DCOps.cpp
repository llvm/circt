//===- DCOps.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/DC/DCOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace circt;
using namespace dc;
using namespace mlir;

bool circt::dc::isI1ValueType(Type t) {
  auto vt = t.dyn_cast<ValueType>();
  if (!vt || vt.getInnerTypes().size() != 1)
    return false;

  auto innerWidth = vt.getInnerTypes()[0].getIntOrFloatBitWidth();
  return innerWidth == 1;
}

namespace circt {
namespace dc {
#include "circt/Dialect/DC/DCCanonicalization.h.inc"

// =============================================================================
// JoinOp
// =============================================================================

void JoinOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {}

// =============================================================================
// ForkOp
// =============================================================================

template <typename TInt>
static ParseResult parseIntInSquareBrackets(OpAsmParser &parser, TInt &v) {
  if (parser.parseLSquare() || parser.parseInteger(v) || parser.parseRSquare())
    return failure();
  return success();
}

ParseResult ForkOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  size_t size = 0;
  if (parseIntInSquareBrackets(parser, size))
    return failure();

  if (size == 0)
    return parser.emitError(parser.getNameLoc(),
                            "fork size must be greater than 0");

  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto tt = dc::TokenType::get(parser.getContext());
  llvm::SmallVector<Type> operandTypes{tt};
  SmallVector<Type> resultTypes{size, tt};
  result.addTypes(resultTypes);
  if (parser.resolveOperand(operand, tt, result.operands))
    return failure();
  return success();
}

void ForkOp::print(OpAsmPrinter &p) {
  p << "[" << getNumResults() << "] ";
  p << getOperand() << " ";
  auto attrs = (*this)->getAttrs();
  if (!attrs.empty()) {
    p << " ";
    p.printOptionalAttrDict(attrs);
  }
}

void ForkOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {}

// =============================================================================
// UnpackOp
// =============================================================================

void UnpackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {}

LogicalResult UnpackOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<OpFoldResult> &results) {
  // Unpack of a pack is a no-op.
  if (auto pack = getInput().getDefiningOp<PackOp>()) {
    results.push_back(pack.getToken());
    results.append(pack.getInputs().begin(), pack.getInputs().end());
    return success();
  }

  return failure();
}

LogicalResult UnpackOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  auto inputType = operands.front().getType().cast<ValueType>();
  results.push_back(TokenType::get(context));
  results.append(inputType.getInnerTypes().begin(),
                 inputType.getInnerTypes().end());
  return success();
}

// =============================================================================
// PackOp
// =============================================================================

void PackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {}

OpFoldResult PackOp::fold(FoldAdaptor adaptor) {
  auto token = getToken();
  auto inputs = getInputs();

  // Pack of an unpack is a no-op.
  if (auto unpack = token.getDefiningOp<UnpackOp>()) {
    llvm::SmallVector<Value> unpackResults = unpack.getResults();
    if (unpackResults.size() == inputs.size() &&
        llvm::all_of(llvm::zip(getInputs(), unpackResults), [&](auto it) {
          return std::get<0>(it) == std::get<1>(it);
        })) {
      return unpack.getInput();
    }
  }

  return {};
}

LogicalResult PackOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  llvm::SmallVector<Type> inputTypes;
  for (auto t : operands.drop_front().getTypes())
    inputTypes.push_back(t);
  auto valueType = dc::ValueType::get(context, inputTypes);
  results.push_back(valueType);
  return success();
}

// =============================================================================
// SelectOp
// =============================================================================

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {}

// =============================================================================
// BufferOp
// =============================================================================

FailureOr<SmallVector<int64_t>> BufferOp::getInitValueArray() {
  assert(getInitValues() && "initValues attribute not set");
  SmallVector<int64_t> values;
  for (auto value : getInitValuesAttr()) {
    if (auto iValue = value.dyn_cast<IntegerAttr>()) {
      values.push_back(iValue.getValue().getSExtValue());
    } else {
      return emitError() << "initValues attribute must be an array of integers";
    }
  }
  return values;
}

LogicalResult BufferOp::verify() {
  // Verify that exactly 'size' number of initial values have been provided, if
  // an initializer list have been provided.
  if (auto initVals = getInitValuesAttr()) {
    auto nInits = initVals.size();
    if (nInits != getSize())
      return emitOpError() << "expected " << getSize()
                           << " init values but got " << nInits << ".";
  }

  return success();
}

} // namespace dc
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/DC/DC.cpp.inc"
