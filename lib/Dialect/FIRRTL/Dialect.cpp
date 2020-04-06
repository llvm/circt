//===- Dialect.cpp - Implement the FIRRTL dialect -------------------------===//
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "spt/Dialect/FIRRTL/IR/Types.h"

using namespace spt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// If the specified attribute set contains the firrtl.name attribute, return it.
static StringAttr getFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    if (!argAttr.first.is("firrtl.name"))
      continue;

    return argAttr.second.dyn_cast<StringAttr>();
  }

  return StringAttr();
}

namespace {

// We implement the OpAsmDialectInterface so that FIRRTL dialect operations
// automatically interpret the firrtl.name attribute on function arguments and
// on operations as their SSA name.
struct FIRRTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {
    if (op->getNumResults() > 0)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("firrtl.name"))
        setNameFn(op->getResult(0), nameAttr.getValue());
  }

  /// Get a special name to use when printing the entry block arguments of the
  /// region contained by an operation in this dialect.
  void getAsmBlockArgumentNames(Block *block,
                                OpAsmSetValueNameFn setNameFn) const override {
    // Check to see if the operation containing the arguments has 'firrtl.name'
    // attributes for them.  If so, use that as the name.
    auto *parentOp = block->getParentOp();

    for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
      // Scan for a 'firrtl.name' attribute.
      if (auto str = getFIRRTLNameAttr(impl::getArgAttrs(parentOp, i)))
        setNameFn(block->getArgument(i), str.getValue());
    }
  }
};
} // end anonymous namespace

FIRRTLDialect::FIRRTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {

  // Register types.
  addTypes<SIntType, UIntType, ClockType, ResetType, AnalogType,
           // Derived Types
           FlipType, BundleType, FVectorType>();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<FIRRTLOpAsmDialectInterface>();
}

FIRRTLDialect::~FIRRTLDialect() {}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  type.cast<FIRRTLType>().print(os.getStream());
}

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//

void CircuitOp::build(Builder *builder, OperationState &result,
                      StringAttr name) {
  // Add an attribute for the name.
  result.addAttribute(builder->getIdentifier("name"), name);

  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
  CircuitOp::ensureTerminator(*bodyRegion, *builder, result.location);
}

static void print(OpAsmPrinter &p, CircuitOp op) {
  p << op.getOperationName() << " ";
  p.printAttribute(op.nameAttr());

  p.printOptionalAttrDictWithKeyword(op.getAttrs(), {"name"});

  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static ParseResult parseCircuitOp(OpAsmParser &parser, OperationState &result) {
  // Parse the module name.
  StringAttr nameAttr;
  if (parser.parseAttribute(nameAttr, "name", result.attributes))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*regionArgs*/ {}, /*argTypes*/ {}))
    return failure();

  CircuitOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// FModuleOp
//===----------------------------------------------------------------------===//

void FModuleOp::build(Builder *builder, OperationState &result, StringAttr name,
                      ArrayRef<std::pair<StringAttr, FIRRTLType>> ports) {
  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> argTypes;
  for (auto elt : ports)
    argTypes.push_back(elt.second);

  // Record the argument and result types as an attribute.
  auto type = builder->getFunctionType(argTypes, /*resultTypes*/ {});
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // Record the names of the arguments if present.
  SmallString<8> attrNameBuf;
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    if (ports[i].first.getValue().empty())
      continue;

    auto argAttr =
        NamedAttribute(builder->getIdentifier("firrtl.name"), ports[i].first);

    result.addAttribute(getArgAttrName(i, attrNameBuf),
                        builder->getDictionaryAttr(argAttr));
  }

  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    body->addArgument(elt.second);

  FModuleOp::ensureTerminator(*bodyRegion, *builder, result.location);
}

// TODO: This ia a clone of mlir::impl::printFunctionSignature, refactor it to
// allow this customization.
static void printFunctionSignature2(OpAsmPrinter &p, Operation *op,
                                    ArrayRef<Type> argTypes, bool isVariadic,
                                    ArrayRef<Type> resultTypes) {
  Region &body = op->getRegion(0);
  bool isExternal = body.empty();

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    Value argumentValue;
    if (!isExternal) {
      argumentValue = body.front().getArgument(i);
      p.printOperand(argumentValue);
      p << ": ";
    }

    p.printType(argTypes[i]);

    auto argAttrs = ::mlir::impl::getArgAttrs(op, i);

    // If the argument has the firrtl.name attribute, and if it was used by
    // the printer exactly (not name mangled with a suffix etc) then we can
    // omit the firrtl.name attribute from the argument attribute dictionary.
    ArrayRef<StringRef> elidedAttrs;
    if (argumentValue) {
      if (auto nameAttr = getFIRRTLNameAttr(argAttrs)) {

        // Check to make sure the asmprinter is printing it correctly.
        SmallString<32> resultNameStr;
        llvm::raw_svector_ostream tmpStream(resultNameStr);
        p.printOperand(argumentValue, tmpStream);

        // If the name is the same as we would otherwise use, then we're good!
        if (tmpStream.str().drop_front() == nameAttr.getValue())
          elidedAttrs = {"firrtl.name"};
      }
    }
    p.printOptionalAttrDict(argAttrs, elidedAttrs);
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';
}

static void print(OpAsmPrinter &p, FModuleOp op) {
  using namespace mlir::impl;

  FunctionType fnType = op.getType();
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // TODO: Should refactor mlir::impl::printFunctionLikeOp to allow these
  // customizations.  Need to not print the terminator.

  // Print the operation and the function name.
  auto funcName =
      op.getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op.getOperationName() << ' ';
  p.printSymbolName(funcName);

  printFunctionSignature2(p, op, argTypes, /*isVariadic*/ false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size());

  // Print the body if this is not an external function.
  Region &body = op.getBody();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
}

static ParseResult parseFModuleOp(OpAsmParser &parser, OperationState &result) {
  using namespace mlir::impl;

  // TODO: Should refactor mlir::impl::parseFunctionLikeOp to allow these
  // customizations for implicit argument names.  Need to not print the
  // terminator.

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> argAttrs;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> resultAttrs;
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
  if (parseFunctionSignature(parser, /*allowVariadic*/ false, entryArgs,
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

  // Postprocess each of the arguments.  If there was no 'firrtl.name'
  // attribute, and if the argument name was non-numeric, then add the
  // firrtl.name attribute with the textual name from the IR.  The name in the
  // text file is a load-bearing part of the IR, but we don't want the
  // verbosity in dumps of including it explicitly in the attribute
  // dictionary.
  for (size_t i = 0, e = argAttrs.size(); i != e; ++i) {
    auto &arg = entryArgs[i];
    auto &attrs = argAttrs[i];

    // If an explicit name attribute was present, don't add the implicit one.
    bool hasNameAttr = false;
    for (auto &elt : attrs)
      if (elt.first.str() == "firrtl.name")
        hasNameAttr = true;
    if (hasNameAttr)
      continue;

    // The name of an argument is of the form "%42" or "%id", and since
    // parsing succeeded, we know it always has one character.
    assert(arg.name.size() > 1 && arg.name[0] == '%' && "Unknown MLIR name");
    if (isdigit(arg.name[1]))
      continue;

    auto nameAttr = StringAttr::get(arg.name.drop_front(), context);
    attrs.push_back({Identifier::get("firrtl.name", context), nameAttr});
  }

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (parser.parseOptionalRegion(
          *body, entryArgs, entryArgs.empty() ? ArrayRef<Type>() : argTypes))
    return failure();

  if (!body->empty())
    FModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

void WhenOp::createElseRegion() {
  assert(!hasElseRegion() && "already has an else region");
  OpBuilder builder(&elseRegion());
  WhenOp::ensureTerminator(elseRegion(), builder, getLoc());
}

void WhenOp::build(Builder *builder, OperationState &result, Value condition,
                   bool withElseRegion) {
  result.addOperands(condition);

  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  WhenOp::ensureTerminator(*thenRegion, *builder, result.location);
  if (withElseRegion)
    WhenOp::ensureTerminator(*elseRegion, *builder, result.location);
}

static void print(OpAsmPrinter &p, WhenOp op) {
  p << op.getOperationName() << " ";
  p.printOperand(op.condition());

  p.printRegion(op.thenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  // Print the 'else' regions if it has any blocks.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }

  // Print the attribute list.
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

static ParseResult parseWhenOp(OpAsmParser &parser, OperationState &result) {
  // Parse the module name.
  OpAsmParser::OperandType conditionOperand;
  if (parser.parseOperand(conditionOperand) ||
      parser.resolveOperand(conditionOperand,
                            UIntType::get(result.getContext(), 1),
                            result.operands))
    return failure();

  // Create the regions for 'then' and 'else'.  The latter must be created even
  // if it remains empty for the validity of the operation.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, {}, {}))
    return failure();
  WhenOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return failure();
    WhenOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

// Return the result of a subfield operation.
FIRRTLType SubfieldOp::getResultType(FIRRTLType inType, StringRef fieldName) {
  if (auto bundleType = inType.dyn_cast<BundleType>()) {
    for (auto &elt : bundleType.getElements()) {
      if (elt.first.strref() == fieldName)
        return elt.second;
    }
  }

  if (auto flipType = inType.dyn_cast<FlipType>())
    if (auto subType = getResultType(flipType.getElementType(), fieldName))
      return FlipType::get(subType);

  return {};
}

FIRRTLType SubindexOp::getResultType(FIRRTLType inType, unsigned fieldIdx) {
  if (auto vectorType = inType.dyn_cast<FVectorType>())
    if (fieldIdx < vectorType.getNumElements())
      return vectorType.getElementType();

  if (auto flipType = inType.dyn_cast<FlipType>())
    if (auto subType = getResultType(flipType.getElementType(), fieldIdx))
      return FlipType::get(subType);

  return {};
}

//===----------------------------------------------------------------------===//
// Binary Primitives
//===----------------------------------------------------------------------===//

/// If LHS and RHS are both UInt or SInt types, the return true and compute the
/// max width of them if known.  If unknown, return -1 in maxWidth.
static bool isSameIntegerType(FIRRTLType lhs, FIRRTLType rhs,
                              int32_t &maxWidth) {
  if (auto lu = lhs.dyn_cast<UIntType>())
    if (auto ru = rhs.dyn_cast<UIntType>()) {
      if (!lu.getWidth().hasValue())
        maxWidth = -1;
      else if (!ru.getWidth().hasValue())
        maxWidth = -1;
      else
        maxWidth = std::max(lu.getWidth().getValue(), ru.getWidth().getValue());
      return true;
    }

  if (auto ls = lhs.dyn_cast<SIntType>())
    if (auto rs = rhs.dyn_cast<SIntType>()) {
      if (!ls.getWidth().hasValue())
        maxWidth = -1;
      else if (!rs.getWidth().hasValue())
        maxWidth = -1;
      else
        maxWidth = std::max(ls.getWidth().getValue(), rs.getWidth().getValue());
      return true;
    }

  return false;
}

FIRRTLType firrtl::getAddResult(FIRRTLType lhs, FIRRTLType rhs) {
  int32_t width;
  if (isSameIntegerType(lhs, rhs, width)) {
    if (width != -1)
      ++width;
    return getIntegerType(lhs.getContext(), lhs.isa<SIntType>(), width);
  }

  return {};
}

FIRRTLType firrtl::getCompareResult(FIRRTLType lhs, FIRRTLType rhs) {
  if ((lhs.isa<UIntType>() && rhs.isa<UIntType>()) ||
      (lhs.isa<SIntType>() && rhs.isa<SIntType>()))
    return UIntType::get(lhs.getContext(), 1);
  return {};
}

FIRRTLType firrtl::getBitwiseBinaryResult(FIRRTLType lhs, FIRRTLType rhs) {
  int32_t width;
  if (isSameIntegerType(lhs, rhs, width))
    return UIntType::get(lhs.getContext(), width);

  return {};
}

FIRRTLType firrtl::getDShlResult(FIRRTLType lhs, FIRRTLType rhs) {
  int32_t width;
  auto rhsui = rhs.dyn_cast<UIntType>();
  if (!rhsui || !isSameIntegerType(lhs, lhs, width))
    return {};

  // If the left or right has unknown result type, then the operation does too.
  if (width == -1 || !rhsui.getWidth().hasValue())
    width = -1;
  else
    width = width + (1 << rhsui.getWidth().getValue()) - 1;
  return getIntegerType(lhs.getContext(), lhs.isa<SIntType>(), width);
}

FIRRTLType firrtl::getDShrResult(FIRRTLType lhs, FIRRTLType rhs) {
  int32_t width;
  if (!rhs.isa<UIntType>() || !isSameIntegerType(lhs, lhs, width))
    return {};
  return lhs;
}

//===----------------------------------------------------------------------===//
// Unary Primitives
//===----------------------------------------------------------------------===//

FIRRTLType firrtl::getAsClockResult(FIRRTLType input) {
  if (input.isa<UIntType>() || input.isa<SIntType>() || input.isa<ClockType>())
    return ClockType::get(input.getContext());
  return {};
}

FIRRTLType firrtl::getAsSIntResult(FIRRTLType input) {
  if (input.isa<ClockType>() || input.isa<ResetType>())
    return SIntType::get(input.getContext(), 1);
  if (input.isa<SIntType>())
    return input;
  if (auto ui = input.dyn_cast<UIntType>())
    return SIntType::get(input.getContext(), ui.getWidthOrSentinel());
  return {};
}

FIRRTLType firrtl::getAsUIntResult(FIRRTLType input) {
  if (input.isa<ClockType>() || input.isa<ResetType>())
    return UIntType::get(input.getContext(), 1);
  if (input.isa<UIntType>())
    return input;
  if (auto si = input.dyn_cast<SIntType>())
    return UIntType::get(input.getContext(), si.getWidthOrSentinel());
  return {};
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

FIRRTLType BitsPrimOp::getResultType(FIRRTLType input, int32_t high,
                                     int32_t low) {
  // High must be >= low and both most be non-negative.
  if (high < low || low < 0)
    return {};

  int32_t width;
  if (isSameIntegerType(input, input, width)) {
    // If the input has staticly known width, check it.  Both and low must be
    // strictly less than width.
    if (width != -1 && high >= width)
      return {};

    return UIntType::get(input.getContext(), high - low + 1);
  }

  return {};
}

FIRRTLType ShlPrimOp::getResultType(FIRRTLType input, int32_t amount) {
  int32_t width;
  if (amount < 0 || !isSameIntegerType(input, input, width))
    return {};

  if (width != -1)
    width += amount;

  return getIntegerType(input.getContext(), input.isa<SIntType>(), width);
}

FIRRTLType ShrPrimOp::getResultType(FIRRTLType input, int32_t amount) {
  int32_t width;
  if (amount < 0 || !isSameIntegerType(input, input, width))
    return {};

  if (width != -1) {
    if (amount > width)
      return {};
    width = std::max(1, width - amount);
  }

  return getIntegerType(input.getContext(), input.isa<SIntType>(), width);
}

#define GET_OP_CLASSES
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
