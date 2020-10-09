//===- Ops.cpp - Implement the RTL operations -----------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/RTL/Visitors.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace rtl;

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<RTLModulePortInfo> ports) {
  using namespace mlir::impl;

  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> argTypes;
  for (auto elt : ports)
    argTypes.push_back(elt.type);

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, /*resultTypes=*/{});
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // Record the names of the arguments if present.
  SmallString<8> attrNameBuf;
  SmallString<8> attrDirBuf;
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    if (ports[i].name.getValue().empty())
      continue;

    auto argAttr =
        NamedAttribute(builder.getIdentifier("rtl.name"), ports[i].name);

    auto dirAttr = NamedAttribute(builder.getIdentifier("rtl.direction"),
                                  ports[i].direction);

    result.addAttribute(getArgAttrName(i, attrNameBuf),
                        builder.getDictionaryAttr({argAttr, dirAttr}));
  }
  result.addRegion();
}

void rtl::RTLModuleOp::build(OpBuilder &builder, OperationState &result,
                             StringAttr name,
                             ArrayRef<RTLModulePortInfo> ports) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    body->addArgument(elt.type);

  rtl::RTLModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

FunctionType rtl::getModuleType(Operation *op) {
  auto typeAttr = op->getAttrOfType<TypeAttr>(RTLModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

StringAttr rtl::getRTLNameAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    if (argAttr.first != "rtl.name")
      continue;
    return argAttr.second.dyn_cast<StringAttr>();
  }
  return StringAttr();
}

StringAttr rtl::getRTLDirectionAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    if (argAttr.first != "rtl.direction")
      continue;
    return argAttr.second.dyn_cast<StringAttr>();
  }
  return StringAttr();
}

void rtl::RTLModuleOp::getRTLModulePortInfo(
    Operation *op, SmallVectorImpl<RTLModulePortInfo> &results) {
  auto argTypes = getModuleType(op).getInputs();

  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    auto argAttrs = ::mlir::impl::getArgAttrs(op, i);
    auto type = argTypes[i].dyn_cast<IntegerType>();

    results.push_back(
        {getRTLNameAttr(argAttrs), type, getRTLDirectionAttr(argAttrs)});
  }
}

static ParseResult parseRTLModuleOp(OpAsmParser &parser,
                                    OperationState &result) {
  using namespace mlir::impl;

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (parseFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                             argTypes, argAttrs, isVariadic, resultTypes,
                             resultAttrs))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());

  auto *context = result.getContext();

  // Postprocess each of the arguments.  If there was no 'rtl.name'
  // attribute, and if the argument name was non-numeric, then add the
  // rtl.name attribute with the textual name from the IR.  The name in the
  // text file is a load-bearing part of the IR, but we don't want the
  // verbosity in dumps of including it explicitly in the attribute
  // dictionary.
  for (size_t i = 0, e = argAttrs.size(); i != e; ++i) {
    auto &attrs = argAttrs[i];

    // If an explicit name attribute was present, don't add the implicit one.
    bool hasNameAttr = false;
    for (auto &elt : attrs)
      if (elt.first.str() == "rtl.name")
        hasNameAttr = true;
    if (hasNameAttr || entryArgs.empty())
      continue;

    auto &arg = entryArgs[i];

    // The name of an argument is of the form "%42" or "%id", and since
    // parsing succeeded, we know it always has one character.
    assert(arg.name.size() > 1 && arg.name[0] == '%' && "Unknown MLIR name");
    if (isdigit(arg.name[1]))
      continue;

    auto nameAttr = StringAttr::get(arg.name.drop_front(), context);
    attrs.push_back({Identifier::get("rtl.name", context), nameAttr});
  }

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (parser.parseOptionalRegion(
          *body, entryArgs, entryArgs.empty() ? ArrayRef<Type>() : argTypes))
    return failure();

  RTLModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

FunctionType getRTLModuleOpType(Operation *op) {
  auto typeAttr =
      op->getAttrOfType<TypeAttr>(rtl::RTLModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

static void printRTLModuleOp(OpAsmPrinter &p, Operation *op) {
  using namespace mlir::impl;

  FunctionType fnType = getRTLModuleOpType(op);
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op->getName() << ' ';
  p.printSymbolName(funcName);

  printFunctionSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size());
}

static void print(OpAsmPrinter &p, RTLModuleOp op) {
  printRTLModuleOp(p, op);

  // Print the body if this is not an external function.
  Region &body = op.getBody();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
}

static LogicalResult verifyRTLInstanceOp(RTLInstanceOp op) {
  auto moduleIR = op.getParentWithTrait<OpTrait::SymbolTable>();
  if (moduleIR == nullptr) {
    op.emitError("Must be contained within a SymbolTable region");
    return failure();
  }
  auto referencedModule =
      mlir::SymbolTable::lookupSymbolIn(moduleIR, op.moduleName());
  if (referencedModule == nullptr) {
    op.emitError(
        llvm::formatv("Cannot find module definition '{0}'", op.moduleName()));
    return failure();
  }
  if (!isa<rtl::RTLModuleOp>(referencedModule)) {
    op.emitError(llvm::formatv("Symbol resolved to '{0}', not a RTLModuleOp",
                               referencedModule->getName()));
    return failure();
  }
  return success();
}

/// Return true if the specified operation is a combinatorial logic op.
bool rtl::isCombinatorial(Operation *op) {
  struct IsCombClassifier
      : public CombinatorialVisitor<IsCombClassifier, bool> {
    bool visitInvalidComb(Operation *op) { return false; }
    bool visitUnhandledComb(Operation *op) { return true; }
  };

  return IsCombClassifier().dispatchCombinatorialVisitor(op);
}

static Attribute getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(value.getBitWidth(), context),
                          value);
}

namespace {
struct ConstantIntMatcher {
  APInt &value;
  ConstantIntMatcher(APInt &value) : value(value) {}
  bool match(Operation *op) {
    if (auto cst = dyn_cast<ConstantOp>(op)) {
      value = cst.value();
      return true;
    }
    return false;
  }
};
} // end anonymous namespace

static inline ConstantIntMatcher m_RConstant(APInt &value) {
  return ConstantIntMatcher(value);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyConstantOp(ConstantOp constant) {
  // If the result type has a bitwidth, then the attribute must match its width.
  auto intType = constant.getType().cast<IntegerType>();
  if (constant.value().getBitWidth() != intType.getWidth()) {
    constant.emitError(
        "firrtl.constant attribute bitwidth doesn't match return type");
    return failure();
  }

  return success();
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

/// Build a ConstantOp from an APInt, infering the result type from the
/// width of the APInt.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       const APInt &value) {

  auto type = IntegerType::get(value.getBitWidth(), IntegerType::Signless,
                               builder.getContext());
  auto attr = builder.getIntegerAttr(type, value);
  return build(builder, result, type, attr);
}

/// This builder allows construction of small signed integers like 0, 1, -1
/// matching a specified MLIR IntegerType.  This shouldn't be used for general
/// constant folding because it only works with values that can be expressed in
/// an int64_t.  Use APInt's instead.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       int64_t value, IntegerType type) {
  auto numBits = type.getWidth();
  build(builder, result, APInt(numBits, (uint64_t)value, /*isSigned=*/true));
}

/// Flattens a single input in `op` if `hasOneUse` is true and it can be defined
/// as an Op. Returns true if successful, and false otherwise.
/// Example: op(1, 2, op(3, 4), 5) -> op(1, 2, 3, 4, 5)  // returns true
template <typename Op>
static bool tryFlatteningOperands(Op op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();

  for (size_t i = 0, size = inputs.size(); i != size; ++i) {
    if (!inputs[i].hasOneUse())
      continue;
    auto flattenOp = inputs[i].template getDefiningOp<Op>();
    if (!flattenOp)
      continue;
    auto flattenOpInputs = flattenOp.inputs();

    SmallVector<Value, 4> newOperands;
    newOperands.reserve(size + flattenOpInputs.size());

    auto flattenOpIndex = inputs.begin() + i;
    newOperands.append(inputs.begin(), flattenOpIndex);
    newOperands.append(flattenOpInputs.begin(), flattenOpInputs.end());
    newOperands.append(flattenOpIndex + 1, inputs.end());

    rewriter.replaceOpWithNewOp<Op>(op, op.getType(), newOperands);
    return true;
  }
  return false;
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto intTy = getType().cast<IntegerType>();
  auto intCst = getValue();

  // Sugar i1 constants with 'true' and 'false'.
  if (intTy.getWidth() == 1)
    return setNameFn(getResult(), intCst.isNullValue() ? "false" : "true");

  // Otherwise, build a complex name with the value and type.
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << intCst << '_' << intTy;
  setNameFn(getResult(), specialName.str());
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Verify SExtOp and ZExtOp.
static LogicalResult verifyExtOp(Operation *op) {
  // The source must be smaller than the dest type.  Both are already known to
  // be signless integers.
  auto srcType = op->getOperand(0).getType().cast<IntegerType>();
  auto dstType = op->getResult(0).getType().cast<IntegerType>();
  if (srcType.getWidth() >= dstType.getWidth()) {
    op->emitOpError("extension must increase bitwidth of operand");
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

void ConcatOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange inputs) {
  unsigned resultWidth = 0;
  for (auto input : inputs) {
    resultWidth += input.getType().cast<IntegerType>().getWidth();
  }
  build(builder, result, builder.getIntegerType(resultWidth), inputs);
}

static LogicalResult verifyExtractOp(ExtractOp op) {
  unsigned srcWidth = op.input().getType().cast<IntegerType>().getWidth();
  unsigned dstWidth = op.getType().cast<IntegerType>().getWidth();
  if (op.lowBit() >= srcWidth || srcWidth - op.lowBit() < dstWidth)
    return op.emitOpError("from bit too large for input"), failure();

  return success();
}

OpFoldResult ExtractOp::fold(ArrayRef<Attribute> operands) {
  // If we are extracting the entire input, then return it.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  APInt value;
  if (mlir::matchPattern(input(), m_RConstant(value))) {
    unsigned dstWidth = getType().cast<IntegerType>().getWidth();
    return getIntAttr(value.lshr(lowBit()).trunc(dstWidth), getContext());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

static LogicalResult verifyUTVariadicRTLOp(Operation *op) {
  auto size = op->getOperands().size();
  if (size < 1)
    return op->emitOpError("requires 1 or more args");

  return success();
}

OpFoldResult AndOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();

  // and(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  APInt value;

  // and(..., 0) -> 0 -- annulment
  if (matchPattern(inputs().back(), m_RConstant(value)) && value.isNullValue())
    return inputs().back();

  return {};
}

void AndOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<AndOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(AndOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // and(..., '1) -> and(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isAllOnesValue()) {
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // and(..., x, x) -> and(..., x) -- idempotent
      if (inputs[size - 1] == inputs[size - 2]) {
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // and(..., c1, c2) -> and(..., c3) where c3 = c1 & c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value & value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), newOperands);
        return success();
      }

      // and(x, and(...)) -> and(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: and(..., x, not(x)) -> and(..., 0) -- complement
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();

  // or(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  APInt value;

  // or(..., '1) -> '1 -- annulment
  if (matchPattern(inputs().back(), m_RConstant(value)) &&
      value.isAllOnesValue())
    return inputs().back();
  return {};
}

void OrOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                       MLIRContext *context) {
  struct Folder final : public OpRewritePattern<OrOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(OrOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // or(..., 0) -> or(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {

        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
        return success();
      }

      // or(..., x, x) -> or(..., x) -- idempotent
      if (inputs[size - 1] == inputs[size - 2]) {
        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
        return success();
      }

      // or(..., c1, c2) -> or(..., c3) where c3 = c1 | c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value | value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), newOperands);
        return success();
      }

      // or(x, or(...)) -> or(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: or(..., x, not(x)) -> or(..., '1) -- complement
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();

  // xor(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  // xor(x, x) -> 0 -- idempotent
  if (size == 2u && inputs()[0] == inputs()[1])
    return IntegerAttr::get(getType(), 0);

  return {};
}

void XorOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<XorOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(XorOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // xor(..., 0) -> xor(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {

        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      if (inputs[size - 1] == inputs[size - 2]) {
        assert(size > 2 &&
               "expected idempotent case for 2 elements handled already.");
        // xor(..., x, x) -> xor (...) -- idempotent
        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(),
                                           inputs.drop_back(/*n=*/2));
        return success();
      }

      // xor(..., c1, c2) -> xor(..., c3) where c3 = c1 ^ c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value ^ value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(), newOperands);
        return success();
      }

      // xor(x, xor(...)) -> xor(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: xor(..., '1) -> not(xor(...))
      /// TODO: xor(..., x, not(x)) -> xor(..., '1)
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();

  // add(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  return {};
}

void AddOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<AddOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(AddOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // add(..., 0) -> add(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // add(..., c1, c2) -> add(..., c3) where c3 = c1 + c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value + value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      // add(x, add(...)) -> add(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: add(..., x, x) -> add(..., shl(x, 1))

      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
  auto size = inputs().size();

  // mul(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  APInt value;

  // mul(..., 0) -> 0 -- annulment
  if (matchPattern(inputs().back(), m_RConstant(value)) && value.isNullValue())
    return inputs().back();

  return {};
}

void MulOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<MulOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(MulOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // mul(..., 1) -> mul(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) && (value == 1u)) {
        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // mul(..., c1, c2) -> mul(..., c3) where c3 = c1 * c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value * value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newOperands);
        return success();
      }

      // mul(a, mul(...)) -> mul(a, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      return failure();
    }
  };
  results.insert<Folder>(context);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/RTL/RTL.cpp.inc"
