//===- LLHDOps.cpp - Implement the LLHD operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the LLHD ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace mlir;

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = function_ref<ElementValueT(ElementValueT)>>
static Attribute constFoldUnaryOp(ArrayRef<Attribute> operands,
                                  const CalculationT &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (!operands[0])
    return {};

  if (auto val = dyn_cast<AttrElementT>(operands[0])) {
    return AttrElementT::get(val.getType(), calculate(val.getValue()));
  } else if (auto val = dyn_cast<SplatElementsAttr>(operands[0])) {
    // Operand is a splat so we can avoid expanding the value out and
    // just fold based on the splat value.
    auto elementResult = calculate(val.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(val.getType(), elementResult);
  }
  if (auto val = dyn_cast<ElementsAttr>(operands[0])) {
    // Operand is ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto valIt = val.getValues<ElementValueT>().begin();
    SmallVector<ElementValueT, 4> elementResults;
    elementResults.reserve(val.getNumElements());
    for (size_t i = 0, e = val.getNumElements(); i < e; ++i, ++valIt)
      elementResults.push_back(calculate(*valIt));
    return DenseElementsAttr::get(val.getType(), elementResults);
  }
  return {};
}

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = function_ref<
              ElementValueT(ElementValueT, ElementValueT, ElementValueT)>>
static Attribute constFoldTernaryOp(ArrayRef<Attribute> operands,
                                    const CalculationT &calculate) {
  assert(operands.size() == 3 && "ternary op takes three operands");
  if (!operands[0] || !operands[1] || !operands[2])
    return {};

  if (isa<AttrElementT>(operands[0]) && isa<AttrElementT>(operands[1]) &&
      isa<AttrElementT>(operands[2])) {
    auto fst = cast<AttrElementT>(operands[0]);
    auto snd = cast<AttrElementT>(operands[1]);
    auto trd = cast<AttrElementT>(operands[2]);

    return AttrElementT::get(
        fst.getType(),
        calculate(fst.getValue(), snd.getValue(), trd.getValue()));
  }
  if (isa<SplatElementsAttr>(operands[0]) &&
      isa<SplatElementsAttr>(operands[1]) &&
      isa<SplatElementsAttr>(operands[2])) {
    // Operands are splats so we can avoid expanding the values out and
    // just fold based on the splat value.
    auto fst = cast<SplatElementsAttr>(operands[0]);
    auto snd = cast<SplatElementsAttr>(operands[1]);
    auto trd = cast<SplatElementsAttr>(operands[2]);

    auto elementResult = calculate(fst.getSplatValue<ElementValueT>(),
                                   snd.getSplatValue<ElementValueT>(),
                                   trd.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(fst.getType(), elementResult);
  }
  if (isa<ElementsAttr>(operands[0]) && isa<ElementsAttr>(operands[1]) &&
      isa<ElementsAttr>(operands[2])) {
    // Operands are ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto fst = cast<ElementsAttr>(operands[0]);
    auto snd = cast<ElementsAttr>(operands[1]);
    auto trd = cast<ElementsAttr>(operands[2]);

    auto fstIt = fst.getValues<ElementValueT>().begin();
    auto sndIt = snd.getValues<ElementValueT>().begin();
    auto trdIt = trd.getValues<ElementValueT>().begin();
    SmallVector<ElementValueT, 4> elementResults;
    elementResults.reserve(fst.getNumElements());
    for (size_t i = 0, e = fst.getNumElements(); i < e;
         ++i, ++fstIt, ++sndIt, ++trdIt)
      elementResults.push_back(calculate(*fstIt, *sndIt, *trdIt));
    return DenseElementsAttr::get(fst.getType(), elementResults);
  }
  return {};
}

namespace {

struct constant_int_all_ones_matcher {
  bool match(Operation *op) {
    APInt value;
    return mlir::detail::constant_int_value_binder(&value).match(op) &&
           value.isAllOnes();
  }
};

} // anonymous namespace

unsigned circt::llhd::getLLHDTypeWidth(Type type) {
  if (auto sig = dyn_cast<llhd::SigType>(type))
    type = sig.getUnderlyingType();
  else if (auto ptr = dyn_cast<llhd::PtrType>(type))
    type = ptr.getUnderlyingType();
  if (auto array = dyn_cast<hw::ArrayType>(type))
    return array.getNumElements();
  if (auto tup = dyn_cast<hw::StructType>(type))
    return tup.getElements().size();
  return type.getIntOrFloatBitWidth();
}

Type circt::llhd::getLLHDElementType(Type type) {
  if (auto sig = dyn_cast<llhd::SigType>(type))
    type = sig.getUnderlyingType();
  else if (auto ptr = dyn_cast<llhd::PtrType>(type))
    type = ptr.getUnderlyingType();
  if (auto array = dyn_cast<hw::ArrayType>(type))
    return array.getElementType();
  return type;
}

//===---------------------------------------------------------------------===//
// LLHD Operations
//===---------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantTimeOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::ConstantTimeOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "const has no operands");
  return getValueAttr();
}

void llhd::ConstantTimeOp::build(OpBuilder &builder, OperationState &result,
                                 unsigned time, const StringRef &timeUnit,
                                 unsigned delta, unsigned epsilon) {
  auto *ctx = builder.getContext();
  auto attr = TimeAttr::get(ctx, time, timeUnit, delta, epsilon);
  return build(builder, result, TimeType::get(ctx), attr);
}

//===----------------------------------------------------------------------===//
// SigExtractOp and PtrExtractOp
//===----------------------------------------------------------------------===//

template <class Op>
static OpFoldResult foldSigPtrExtractOp(Op op, ArrayRef<Attribute> operands) {

  if (!operands[1])
    return nullptr;

  // llhd.sig.extract(input, 0) with inputWidth == resultWidth => input
  if (op.getResultWidth() == op.getInputWidth() &&
      cast<IntegerAttr>(operands[1]).getValue().isZero())
    return op.getInput();

  return nullptr;
}

OpFoldResult llhd::SigExtractOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrExtractOp(*this, adaptor.getOperands());
}

OpFoldResult llhd::PtrExtractOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrExtractOp(*this, adaptor.getOperands());
}

//===----------------------------------------------------------------------===//
// SigArraySliceOp and PtrArraySliceOp
//===----------------------------------------------------------------------===//

template <class Op>
static OpFoldResult foldSigPtrArraySliceOp(Op op,
                                           ArrayRef<Attribute> operands) {
  if (!operands[1])
    return nullptr;

  // llhd.sig.array_slice(input, 0) with inputWidth == resultWidth => input
  if (op.getResultWidth() == op.getInputWidth() &&
      cast<IntegerAttr>(operands[1]).getValue().isZero())
    return op.getInput();

  return nullptr;
}

OpFoldResult llhd::SigArraySliceOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrArraySliceOp(*this, adaptor.getOperands());
}

OpFoldResult llhd::PtrArraySliceOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrArraySliceOp(*this, adaptor.getOperands());
}

template <class Op>
static LogicalResult canonicalizeSigPtrArraySliceOp(Op op,
                                                    PatternRewriter &rewriter) {
  IntegerAttr indexAttr;
  if (!matchPattern(op.getLowIndex(), m_Constant(&indexAttr)))
    return failure();

  // llhd.sig.array_slice(llhd.sig.array_slice(target, a), b)
  //   => llhd.sig.array_slice(target, a+b)
  IntegerAttr a;
  if (matchPattern(op.getInput(),
                   m_Op<Op>(matchers::m_Any(), m_Constant(&a)))) {
    auto sliceOp = op.getInput().template getDefiningOp<Op>();
    op.getInputMutable().assign(sliceOp.getInput());
    Value newIndex = rewriter.create<hw::ConstantOp>(
        op->getLoc(), a.getValue() + indexAttr.getValue());
    op.getLowIndexMutable().assign(newIndex);

    return success();
  }

  return failure();
}

LogicalResult llhd::SigArraySliceOp::canonicalize(llhd::SigArraySliceOp op,
                                                  PatternRewriter &rewriter) {
  return canonicalizeSigPtrArraySliceOp(op, rewriter);
}

LogicalResult llhd::PtrArraySliceOp::canonicalize(llhd::PtrArraySliceOp op,
                                                  PatternRewriter &rewriter) {
  return canonicalizeSigPtrArraySliceOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// SigStructExtractOp and PtrStructExtractOp
//===----------------------------------------------------------------------===//

template <class SigPtrType>
static LogicalResult inferReturnTypesOfStructExtractOp(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Type type =
      cast<hw::StructType>(
          cast<SigPtrType>(operands[0].getType()).getUnderlyingType())
          .getFieldType(
              cast<StringAttr>(attrs.getNamed("field")->getValue()).getValue());
  if (!type) {
    context->getDiagEngine().emit(loc.value_or(UnknownLoc()),
                                  DiagnosticSeverity::Error)
        << "invalid field name specified";
    return failure();
  }
  results.push_back(SigPtrType::get(type));
  return success();
}

LogicalResult llhd::SigStructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<llhd::SigType>(
      context, loc, operands, attrs, properties, regions, results);
}

LogicalResult llhd::PtrStructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<llhd::PtrType>(
      context, loc, operands, attrs, properties, regions, results);
}

//===----------------------------------------------------------------------===//
// DrvOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::DrvOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &result) {
  if (!getEnable())
    return failure();

  if (matchPattern(getEnable(), m_One())) {
    getEnableMutable().clear();
    return success();
  }

  return failure();
}

LogicalResult llhd::DrvOp::canonicalize(llhd::DrvOp op,
                                        PatternRewriter &rewriter) {
  if (!op.getEnable())
    return failure();

  if (matchPattern(op.getEnable(), m_Zero())) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

// Implement this operation for the BranchOpInterface
SuccessorOperands llhd::WaitOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOpsMutable());
}

//===----------------------------------------------------------------------===//
// EntityOp
//===----------------------------------------------------------------------===//

/// Parse an argument list of an entity operation.
/// The argument list and argument types are returned in args and argTypes
/// respectively.
static ParseResult
parseArgumentList(OpAsmParser &parser,
                  SmallVectorImpl<OpAsmParser::Argument> &args,
                  SmallVectorImpl<Type> &argTypes) {
  auto parseElt = [&]() -> ParseResult {
    OpAsmParser::Argument argument;
    Type argType;
    auto optArg = parser.parseOptionalArgument(argument);
    if (optArg.has_value()) {
      if (succeeded(optArg.value())) {
        if (!argument.ssaName.name.empty() &&
            succeeded(parser.parseColonType(argType))) {
          args.push_back(argument);
          argTypes.push_back(argType);
          args.back().type = argType;
        }
      }
    }
    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseElt);
}

/// parse an entity signature with syntax:
/// (%arg0 : T0, %arg1 : T1, <...>) -> (%out0 : T0, %out1 : T1, <...>)
static ParseResult
parseEntitySignature(OpAsmParser &parser, OperationState &result,
                     SmallVectorImpl<OpAsmParser::Argument> &args,
                     SmallVectorImpl<Type> &argTypes) {
  if (parseArgumentList(parser, args, argTypes))
    return failure();
  // create the integer attribute with the number of inputs.
  IntegerAttr insAttr = parser.getBuilder().getI64IntegerAttr(args.size());
  result.addAttribute("ins", insAttr);
  if (succeeded(parser.parseOptionalArrow()))
    if (parseArgumentList(parser, args, argTypes))
      return failure();

  return success();
}

ParseResult llhd::EntityOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr entityName;
  SmallVector<OpAsmParser::Argument, 4> args;
  SmallVector<Type, 4> argTypes;

  if (parser.parseSymbolName(entityName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parseEntitySignature(parser, result, args, argTypes))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto type = parser.getBuilder().getFunctionType(argTypes, std::nullopt);
  result.addAttribute(
      circt::llhd::EntityOp::getFunctionTypeAttrName(result.name),
      TypeAttr::get(type));

  auto &body = *result.addRegion();
  if (parser.parseRegion(body, args))
    return failure();
  if (body.empty())
    body.push_back(std::make_unique<Block>().release());

  return success();
}

static void printArgumentList(OpAsmPrinter &printer,
                              std::vector<BlockArgument> args) {
  printer << "(";
  llvm::interleaveComma(args, printer, [&](BlockArgument arg) {
    printer << arg << " : " << arg.getType();
  });
  printer << ")";
}

void llhd::EntityOp::print(OpAsmPrinter &printer) {
  std::vector<BlockArgument> ins, outs;
  uint64_t nIns = getInsAttr().getInt();
  for (uint64_t i = 0; i < getBody().front().getArguments().size(); ++i) {
    // no furter verification for the attribute type is required, already
    // handled by verify.
    if (i < nIns) {
      ins.push_back(getBody().front().getArguments()[i]);
    } else {
      outs.push_back(getBody().front().getArguments()[i]);
    }
  }
  auto entityName =
      (*this)
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  printer << " ";
  printer.printSymbolName(entityName);
  printer << " ";
  printArgumentList(printer, ins);
  printer << " -> ";
  printArgumentList(printer, outs);
  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs =*/{SymbolTable::getSymbolAttrName(),
                        getFunctionTypeAttrName(), "ins"});
  printer << " ";
  printer.printRegion(getBody(), false, false);
}

LogicalResult llhd::EntityOp::verify() {
  uint64_t numArgs = getNumArguments();
  uint64_t nIns = getInsAttr().getInt();
  // check that there is at most one flag for each argument
  if (numArgs < nIns) {
    return emitError(
               "Cannot have more inputs than arguments, expected at most ")
           << numArgs << " but got: " << nIns;
  }

  // Check that all block arguments are of signal type
  for (size_t i = 0; i < numArgs; ++i)
    if (!isa<llhd::SigType>(getArgument(i).getType()))
      return emitError("usage of invalid argument type. Got ")
             << getArgument(i).getType() << ", expected LLHD signal type";

  return success();
}

LogicalResult circt::llhd::EntityOp::verifyType() {
  FunctionType type = getFunctionType();

  // Fail if function returns any values. An entity's outputs are specially
  // marked arguments.
  if (type.getNumResults() > 0)
    return emitOpError("an entity cannot have return types.");

  // Check that all operands are of signal type
  for (Type inputType : type.getInputs())
    if (!isa<llhd::SigType>(inputType))
      return emitOpError("usage of invalid argument type. Got ")
             << inputType << ", expected LLHD signal type";

  return success();
}

LogicalResult circt::llhd::EntityOp::verifyBody() {
  // check signal names are unique
  llvm::StringSet<> sigSet;
  llvm::StringSet<> instSet;
  auto walkResult = walk([&sigSet, &instSet](Operation *op) -> WalkResult {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<SigOp>([&](auto sigOp) -> WalkResult {
          if (!sigSet.insert(sigOp.getName()).second)
            return sigOp.emitError("redefinition of signal named '")
                   << sigOp.getName() << "'!";

          return success();
        })
        .Case<OutputOp>([&](auto outputOp) -> WalkResult {
          if (outputOp.getName() && !sigSet.insert(*outputOp.getName()).second)
            return outputOp.emitError("redefinition of signal named '")
                   << *outputOp.getName() << "'!";

          return success();
        })
        .Case<InstOp>([&](auto instOp) -> WalkResult {
          if (!instSet.insert(instOp.getName()).second)
            return instOp.emitError("redefinition of instance named '")
                   << instOp.getName() << "'!";

          return success();
        })
        .Default([](auto op) -> WalkResult { return WalkResult::advance(); });
  });

  return failure(walkResult.wasInterrupted());
}

/// Returns the argument types of this function.
ArrayRef<Type> llhd::EntityOp::getArgumentTypes() {
  return getFunctionType().getInputs();
}

/// Returns the result types of this function.
ArrayRef<Type> llhd::EntityOp::getResultTypes() {
  return getFunctionType().getResults();
}

Region *llhd::EntityOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

//===----------------------------------------------------------------------===//
// ProcOp
//===----------------------------------------------------------------------===//

LogicalResult circt::llhd::ProcOp::verifyType() {
  // Fail if function returns more than zero values. This is because the
  // outputs of a process are specially marked arguments.
  if (getNumResults() > 0) {
    return emitOpError(
        "process has more than zero return types, this is not allowed");
  }

  // Check that all operands are of signal type
  for (int i = 0, e = getNumArguments(); i < e; ++i) {
    if (!isa<llhd::SigType>(getArgument(i).getType())) {
      return emitOpError("usage of invalid argument type, was ")
             << getArgument(i).getType() << ", expected LLHD signal type";
    }
  }
  return success();
}

/// Returns the argument types of this function.
ArrayRef<Type> llhd::ProcOp::getArgumentTypes() {
  return getFunctionType().getInputs();
}

/// Returns the result types of this function.
ArrayRef<Type> llhd::ProcOp::getResultTypes() {
  return getFunctionType().getResults();
}

LogicalResult circt::llhd::ProcOp::verifyBody() { return success(); }

LogicalResult llhd::ProcOp::verify() {
  // Check that the ins attribute is smaller or equal the number of
  // arguments
  uint64_t numArgs = getNumArguments();
  uint64_t numIns = getInsAttr().getInt();
  if (numArgs < numIns) {
    return emitOpError(
               "Cannot have more inputs than arguments, expected at most ")
           << numArgs << ", got " << numIns;
  }
  return success();
}

static ParseResult
parseProcArgumentList(OpAsmParser &parser, SmallVectorImpl<Type> &argTypes,
                      SmallVectorImpl<OpAsmParser::Argument> &argNames) {
  if (parser.parseLParen())
    return failure();

  // The argument list either has to consistently have ssa-id's followed by
  // types, or just be a type list.  It isn't ok to sometimes have SSA ID's
  // and sometimes not.
  auto parseArgument = [&]() -> ParseResult {
    llvm::SMLoc loc = parser.getCurrentLocation();

    // Parse argument name if present.
    OpAsmParser::Argument argument;
    Type argumentType;
    auto optArg = parser.parseOptionalArgument(argument);
    if (optArg.has_value()) {
      if (succeeded(optArg.value())) {
        // Reject this if the preceding argument was missing a name.
        if (argNames.empty() && !argTypes.empty())
          return parser.emitError(loc,
                                  "expected type instead of SSA identifier");
        argNames.push_back(argument);

        if (parser.parseColonType(argumentType))
          return failure();
      } else if (!argNames.empty()) {
        // Reject this if the preceding argument had a name.
        return parser.emitError(loc, "expected SSA identifier");
      } else if (parser.parseType(argumentType)) {
        return failure();
      }
    }

    // Add the argument type.
    argTypes.push_back(argumentType);
    argNames.back().type = argumentType;

    return success();
  };

  // Parse the function arguments.
  if (failed(parser.parseOptionalRParen())) {
    do {
      unsigned numTypedArguments = argTypes.size();
      if (parseArgument())
        return failure();

      llvm::SMLoc loc = parser.getCurrentLocation();
      if (argTypes.size() == numTypedArguments &&
          succeeded(parser.parseOptionalComma()))
        return parser.emitError(loc, "variadic arguments are not allowed");
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  return success();
}

ParseResult llhd::ProcOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr procName;
  SmallVector<OpAsmParser::Argument, 8> argNames;
  SmallVector<Type, 8> argTypes;
  Builder &builder = parser.getBuilder();

  if (parser.parseSymbolName(procName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parseProcArgumentList(parser, argTypes, argNames))
    return failure();

  result.addAttribute("ins", builder.getI64IntegerAttr(argTypes.size()));
  if (parser.parseArrow())
    return failure();

  if (parseProcArgumentList(parser, argTypes, argNames))
    return failure();

  auto type = builder.getFunctionType(argTypes, std::nullopt);
  result.addAttribute(circt::llhd::ProcOp::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, argNames))
    return failure();

  return success();
}

/// Print the signature of the `proc` unit. Assumes that it passed the
/// verification.
static void printProcArguments(OpAsmPrinter &p, Operation *op,
                               ArrayRef<Type> types, uint64_t numIns) {
  Region &body = op->getRegion(0);
  auto printList = [&](unsigned i, unsigned max) -> void {
    for (; i < max; ++i) {
      p << body.front().getArgument(i) << " : " << types[i];
      p.printOptionalAttrDict(::mlir::function_interface_impl::getArgAttrs(
          cast<mlir::FunctionOpInterface>(op), i));

      if (i < max - 1)
        p << ", ";
    }
  };

  p << '(';
  printList(0, numIns);
  p << ") -> (";
  printList(numIns, types.size());
  p << ')';
}

void llhd::ProcOp::print(OpAsmPrinter &printer) {
  FunctionType type = getFunctionType();
  printer << ' ';
  printer.printSymbolName(getName());
  printProcArguments(printer, getOperation(), type.getInputs(),
                     getInsAttr().getInt());
  printer << " ";
  printer.printRegion(getBody(), false, true);
}

Region *llhd::ProcOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

//===----------------------------------------------------------------------===//
// InstOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::InstOp::verify() {
  // Check that the callee attribute was specified.
  auto calleeAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!calleeAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");

  auto proc = (*this)->getParentOfType<ModuleOp>().lookupSymbol<llhd::ProcOp>(
      calleeAttr.getValue());
  if (proc) {
    auto type = proc.getFunctionType();

    if (proc.getIns() != getInputs().size())
      return emitOpError("incorrect number of inputs for proc instantiation");

    if (type.getNumInputs() != getNumOperands())
      return emitOpError("incorrect number of outputs for proc instantiation");

    for (size_t i = 0, e = type.getNumInputs(); i != e; ++i) {
      if (getOperand(i).getType() != type.getInput(i))
        return emitOpError("operand type mismatch");
    }

    return success();
  }

  auto entity =
      (*this)->getParentOfType<ModuleOp>().lookupSymbol<llhd::EntityOp>(
          calleeAttr.getValue());
  if (entity) {
    auto type = entity.getFunctionType();

    if (entity.getIns() != getInputs().size())
      return emitOpError("incorrect number of inputs for entity instantiation");

    if (type.getNumInputs() != getNumOperands())
      return emitOpError(
          "incorrect number of outputs for entity instantiation");

    for (size_t i = 0, e = type.getNumInputs(); i != e; ++i) {
      if (getOperand(i).getType() != type.getInput(i))
        return emitOpError("operand type mismatch");
    }

    return success();
  }

  auto module =
      (*this)->getParentOfType<ModuleOp>().lookupSymbol<hw::HWModuleOp>(
          calleeAttr.getValue());
  if (module) {

    if (module.getNumInputPorts() != getInputs().size())
      return emitOpError(
          "incorrect number of inputs for hw.module instantiation");

    if (module.getNumOutputPorts() + module.getNumInputPorts() !=
        getNumOperands())
      return emitOpError(
          "incorrect number of outputs for hw.module instantiation");

    // Check input types
    for (size_t i = 0, e = module.getNumInputPorts(); i != e; ++i) {
      if (cast<llhd::SigType>(getOperand(i).getType()).getUnderlyingType() !=
          module.getInputTypes()[i])
        return emitOpError("input type mismatch");
    }

    // Check output types
    for (size_t i = 0, e = module.getNumOutputPorts(); i != e; ++i) {
      if (cast<llhd::SigType>(
              getOperand(module.getNumInputPorts() + i).getType())
              .getUnderlyingType() != module.getOutputTypes()[i])
        return emitOpError("output type mismatch");
    }

    return success();
  }

  return emitOpError()
         << "'" << calleeAttr.getValue()
         << "' does not reference a valid proc, entity, or hw.module";
}

FunctionType llhd::InstOp::getCalleeType() {
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(getContext(), argTypes, ArrayRef<Type>());
}

//===----------------------------------------------------------------------===//
// ConnectOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::ConnectOp::canonicalize(llhd::ConnectOp op,
                                            PatternRewriter &rewriter) {
  if (op.getLhs() == op.getRhs())
    rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// RegOp
//===----------------------------------------------------------------------===//

ParseResult llhd::RegOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand signal;
  Type signalType;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> valueOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> triggerOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> delayOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> gateOperands;
  SmallVector<Type, 8> valueTypes;
  llvm::SmallVector<int64_t, 8> modesArray;
  llvm::SmallVector<int64_t, 8> gateMask;
  int64_t gateCount = 0;

  if (parser.parseOperand(signal))
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand value;
    OpAsmParser::UnresolvedOperand trigger;
    OpAsmParser::UnresolvedOperand delay;
    OpAsmParser::UnresolvedOperand gate;
    Type valueType;
    StringAttr modeAttr;
    NamedAttrList attrStorage;

    if (parser.parseLParen())
      return failure();
    if (parser.parseOperand(value) || parser.parseComma())
      return failure();
    if (parser.parseAttribute(modeAttr, parser.getBuilder().getNoneType(),
                              "modes", attrStorage))
      return failure();
    auto attrOptional = llhd::symbolizeRegMode(modeAttr.getValue());
    if (!attrOptional)
      return parser.emitError(parser.getCurrentLocation(),
                              "invalid string attribute");
    modesArray.push_back(static_cast<int64_t>(*attrOptional));
    if (parser.parseOperand(trigger))
      return failure();
    if (parser.parseKeyword("after") || parser.parseOperand(delay))
      return failure();
    if (succeeded(parser.parseOptionalKeyword("if"))) {
      gateMask.push_back(++gateCount);
      if (parser.parseOperand(gate))
        return failure();
      gateOperands.push_back(gate);
    } else {
      gateMask.push_back(0);
    }
    if (parser.parseColon() || parser.parseType(valueType) ||
        parser.parseRParen())
      return failure();
    valueOperands.push_back(value);
    triggerOperands.push_back(trigger);
    delayOperands.push_back(delay);
    valueTypes.push_back(valueType);
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(signalType))
    return failure();
  if (parser.resolveOperand(signal, signalType, result.operands))
    return failure();
  if (parser.resolveOperands(valueOperands, valueTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();
  for (auto operand : triggerOperands)
    if (parser.resolveOperand(operand, parser.getBuilder().getI1Type(),
                              result.operands))
      return failure();
  for (auto operand : delayOperands)
    if (parser.resolveOperand(
            operand, llhd::TimeType::get(parser.getBuilder().getContext()),
            result.operands))
      return failure();
  for (auto operand : gateOperands)
    if (parser.resolveOperand(operand, parser.getBuilder().getI1Type(),
                              result.operands))
      return failure();
  result.addAttribute("gateMask",
                      parser.getBuilder().getI64ArrayAttr(gateMask));
  result.addAttribute("modes", parser.getBuilder().getI64ArrayAttr(modesArray));
  llvm::SmallVector<int32_t, 5> operandSizes;
  operandSizes.push_back(1);
  operandSizes.push_back(valueOperands.size());
  operandSizes.push_back(triggerOperands.size());
  operandSizes.push_back(delayOperands.size());
  operandSizes.push_back(gateOperands.size());
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(operandSizes));

  return success();
}

void llhd::RegOp::print(OpAsmPrinter &printer) {
  printer << " " << getSignal();
  for (size_t i = 0, e = getValues().size(); i < e; ++i) {
    std::optional<llhd::RegMode> mode = llhd::symbolizeRegMode(
        cast<IntegerAttr>(getModes().getValue()[i]).getInt());
    if (!mode) {
      emitError("invalid RegMode");
      return;
    }
    printer << ", (" << getValues()[i] << ", \""
            << llhd::stringifyRegMode(*mode) << "\" " << getTriggers()[i]
            << " after " << getDelays()[i];
    if (hasGate(i))
      printer << " if " << getGateAt(i);
    printer << " : " << getValues()[i].getType() << ")";
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"modes", "gateMask", "operandSegmentSizes"});
  printer << " : " << getSignal().getType();
}

LogicalResult llhd::RegOp::verify() {
  // At least one trigger has to be present
  if (getTriggers().size() < 1)
    return emitError("At least one trigger quadruple has to be present.");

  // Values variadic operand must have the same size as the triggers variadic
  if (getValues().size() != getTriggers().size())
    return emitOpError("Number of 'values' is not equal to the number of "
                       "'triggers', got ")
           << getValues().size() << " modes, but " << getTriggers().size()
           << " triggers!";

  // Delay variadic operand must have the same size as the triggers variadic
  if (getDelays().size() != getTriggers().size())
    return emitOpError("Number of 'delays' is not equal to the number of "
                       "'triggers', got ")
           << getDelays().size() << " modes, but " << getTriggers().size()
           << " triggers!";

  // Array Attribute of RegModes must have the same number of elements as the
  // variadics
  if (getModes().size() != getTriggers().size())
    return emitOpError("Number of 'modes' is not equal to the number of "
                       "'triggers', got ")
           << getModes().size() << " modes, but " << getTriggers().size()
           << " triggers!";

  // Array Attribute 'gateMask' must have the same number of elements as the
  // triggers and values variadics
  if (getGateMask().size() != getTriggers().size())
    return emitOpError("Size of 'gateMask' is not equal to the size of "
                       "'triggers', got ")
           << getGateMask().size() << " modes, but " << getTriggers().size()
           << " triggers!";

  // Number of non-zero elements in 'gateMask' has to be the same as the size
  // of the gates variadic, also each number from 1 to size-1 has to occur
  // only once and in increasing order
  unsigned counter = 0;
  unsigned prevElement = 0;
  for (Attribute maskElem : getGateMask().getValue()) {
    int64_t val = cast<IntegerAttr>(maskElem).getInt();
    if (val < 0)
      return emitError("Element in 'gateMask' must not be negative!");
    if (val == 0)
      continue;
    if (val != ++prevElement)
      return emitError(
          "'gateMask' has to contain every number from 1 to the "
          "number of gates minus one exactly once in increasing order "
          "(may have zeros in-between).");
    counter++;
  }
  if (getGates().size() != counter)
    return emitError("The number of non-zero elements in 'gateMask' and the "
                     "size of the 'gates' variadic have to match.");

  // Each value must be either the same type as the 'signal' or the underlying
  // type of the 'signal'
  for (auto val : getValues()) {
    if (val.getType() != getSignal().getType() &&
        val.getType() !=
            cast<llhd::SigType>(getSignal().getType()).getUnderlyingType()) {
      return emitOpError(
          "type of each 'value' has to be either the same as the "
          "type of 'signal' or the underlying type of 'signal'");
    }
  }
  return success();
}

#include "circt/Dialect/LLHD/IR/LLHDEnums.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LLHD/IR/LLHD.cpp.inc"
