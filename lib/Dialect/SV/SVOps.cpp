//===- SVOps.cpp - Implement the SV operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the SV ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

using namespace circt;
using namespace sv;

/// Return true if the specified operation is an expression.
bool sv::isExpression(Operation *op) {
  return isa<sv::VerbatimExprOp>(op) || isa<sv::GetModportOp>(op) ||
         isa<sv::ReadInterfaceSignalOp>(op) || isa<sv::ConstantXOp>(op) ||
         isa<sv::ConstantZOp>(op);
}

LogicalResult sv::verifyInProceduralRegion(Operation *op) {
  if (op->getParentOp()->hasTrait<sv::ProceduralRegion>())
    return success();
  op->emitError() << op->getName() << " should be in a procedural region";
  return failure();
}

LogicalResult sv::verifyInNonProceduralRegion(Operation *op) {
  if (!op->getParentOp()->hasTrait<sv::ProceduralRegion>())
    return success();
  op->emitError() << op->getName() << " should be in a non-procedural region";
  return failure();
}

/// Returns the operation registered with the given symbol name with the regions
/// of 'symbolTableOp'. recurse through nested regions which don't contain the
/// symboltable trait. Returns nullptr if no valid symbol was found.
static Operation *lookupSymbolInNested(Operation *symbolTableOp,
                                       StringRef symbol) {
  Region &region = symbolTableOp->getRegion(0);
  if (region.empty())
    return nullptr;

  // Look for a symbol with the given name.
  Identifier symbolNameId = Identifier::get(SymbolTable::getSymbolAttrName(),
                                            symbolTableOp->getContext());
  for (Block &block : region)
    for (Operation &nestedOp : block) {
      auto nameAttr = nestedOp.getAttrOfType<StringAttr>(symbolNameId);
      if (nameAttr && nameAttr.getValue() == symbol)
        return &nestedOp;
      if (!nestedOp.hasTrait<OpTrait::SymbolTable>() &&
          nestedOp.getNumRegions()) {
        if (auto *nop = lookupSymbolInNested(&nestedOp, symbol))
          return nop;
      }
    }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseImplicitSSAName(OpAsmParser &parser,
                                        NamedAttrList &resultAttrs) {

  if (parser.parseOptionalAttrDict(resultAttrs))
    return failure();

  // If the attribute dictionary contains no 'name' attribute, infer it from
  // the SSA name (if specified).
  bool hadName = llvm::any_of(
      resultAttrs, [](NamedAttribute attr) { return attr.first == "name"; });

  // If there was no name specified, check to see if there was a useful name
  // specified in the asm file.
  if (hadName)
    return success();

  // If there is no explicit name attribute, get it from the SSA result name.
  // If numeric, just use an empty name.
  auto resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  auto nameAttr = parser.getBuilder().getStringAttr(resultName);
  auto *context = parser.getBuilder().getContext();
  resultAttrs.push_back({Identifier::get("name", context), nameAttr});
  return success();
}

static void printImplicitSSAName(OpAsmPrinter &p, Operation *op,
                                 DictionaryAttr attr) {
  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  bool namesDisagree = false;

  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto expectedName = op->getAttrOfType<StringAttr>("name").getValue();
  auto actualName = tmpStream.str().drop_front();
  if (actualName != expectedName) {
    // Anonymous names are printed as digits, which is fine.
    if (!expectedName.empty() || !isdigit(actualName[0]))
      namesDisagree = true;
  }

  if (namesDisagree)
    p.printOptionalAttrDict(op->getAttrs(), {"sym_name"});
  else
    p.printOptionalAttrDict(op->getAttrs(), {"name", "sym_name"});
}

//===----------------------------------------------------------------------===//
// VerbatimExprOp
//===----------------------------------------------------------------------===//

void VerbatimExprOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // If the string is macro like, then use a pretty name.  We only take the
  // string up to a weird character (like a paren) and currently ignore
  // parenthesized expressions.
  auto isOkCharacter = [](char c) { return llvm::isAlnum(c) || c == '_'; };
  auto name = string();
  // Ignore a leading ` in macro name.
  if (name.startswith("`"))
    name = name.drop_front();
  name = name.take_while(isOkCharacter);
  if (!name.empty())
    setNameFn(getResult(), name);
}

//===----------------------------------------------------------------------===//
// ConstantXOp / ConstantZOp
//===----------------------------------------------------------------------===//

void ConstantXOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "x_" << getType();
  setNameFn(getResult(), specialName.str());
}

void ConstantZOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "z_" << getType();
  setNameFn(getResult(), specialName.str());
}

//===----------------------------------------------------------------------===//
// RegOp
//===----------------------------------------------------------------------===//

void RegOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type elementType, StringAttr name, StringAttr sym_name) {
  if (!name)
    name = odsBuilder.getStringAttr("");
  odsState.addAttribute("name", name);
  if (sym_name)
    odsState.addAttribute("sym_name", sym_name);
  odsState.addTypes(hw::InOutType::get(elementType));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void RegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

// If this reg is only written to, delete the reg and all writers.
LogicalResult RegOp::canonicalize(RegOp op, PatternRewriter &rewriter) {
  // If the reg has a symbol, then we can't delete it.
  if (op.sym_nameAttr())
    return failure();
  // Check that all operations on the wire are sv.connects. All other wire
  // operations will have been handled by other canonicalization.
  for (auto &use : op.getResult().getUses())
    if (!isa<ConnectOp>(use.getOwner()))
      return failure();

  // Remove all uses of the wire.
  for (auto &use : op.getResult().getUses())
    rewriter.eraseOp(use.getOwner());

  // Remove the wire.
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Control flow like-operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IfDefOp

void IfDefOp::build(OpBuilder &odsBuilder, OperationState &result,
                    StringRef cond, std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  build(odsBuilder, result, odsBuilder.getStringAttr(cond), thenCtor, elseCtor);
}

void IfDefOp::build(OpBuilder &odsBuilder, OperationState &result,
                    StringAttr cond, std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  assert(!cond.getValue().empty() && cond.getValue().front() != '!' &&
         "Should only use simple Verilog identifiers in ifdef conditions");
  OpBuilder::InsertionGuard guard(odsBuilder);

  result.addAttribute("cond", cond);
  odsBuilder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (thenCtor)
    thenCtor();

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    odsBuilder.createBlock(elseRegion);
    elseCtor();
  }
}

static bool isEmptyBlockExceptForTerminator(Block *block) {
  assert(block && "Blcok must be non-null");
  return block->empty() || block->front().hasTrait<OpTrait::IsTerminator>();
}

// If both thenRegion and elseRegion are empty, erase op.
LogicalResult IfDefOp::canonicalize(IfDefOp op, PatternRewriter &rewriter) {
  if (!isEmptyBlockExceptForTerminator(op.getThenBlock()))
    return failure();

  if (op.hasElse() && !isEmptyBlockExceptForTerminator(op.getElseBlock()))
    return failure();

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// IfDefProceduralOp

void IfDefProceduralOp::build(OpBuilder &odsBuilder, OperationState &result,
                              StringRef cond, std::function<void()> thenCtor,
                              std::function<void()> elseCtor) {
  IfDefOp::build(odsBuilder, result, cond, std::move(thenCtor),
                 std::move(elseCtor));
}

//===----------------------------------------------------------------------===//
// IfOp

void IfOp::build(OpBuilder &odsBuilder, OperationState &result, Value cond,
                 std::function<void()> thenCtor,
                 std::function<void()> elseCtor) {
  OpBuilder::InsertionGuard guard(odsBuilder);

  result.addOperands(cond);
  odsBuilder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (thenCtor)
    thenCtor();

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    odsBuilder.createBlock(elseRegion);
    elseCtor();
  }
}

/// Replaces the given op with the contents of the given single-block region.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *fromBlock = &region.front();
  // Merge it in above the specified operation.
  op->getBlock()->getOperations().splice(Block::iterator(op),
                                         fromBlock->getOperations());
}

LogicalResult IfOp::canonicalize(IfOp op, PatternRewriter &rewriter) {
  if (auto constant = op.cond().getDefiningOp<hw::ConstantOp>()) {

    if (constant.getValue().isAllOnesValue())
      replaceOpWithRegion(rewriter, op, op.thenRegion());
    else if (!op.elseRegion().empty())
      replaceOpWithRegion(rewriter, op, op.elseRegion());

    rewriter.eraseOp(op);

    return success();
  }

  // Erase empty if's.

  // If there is stuff in the then block, leave this operation alone.
  if (!op.getThenBlock()->empty())
    return failure();

  // If not and there is no else, then this operation is just useless.
  if (!op.hasElse() || op.getElseBlock()->empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  // Otherwise, invert the condition and move the 'else' block to the 'then'
  // region.
  auto full =
      rewriter.create<hw::ConstantOp>(op.getLoc(), op.cond().getType(), -1);
  Value ops[] = {full, op.cond()};
  auto cond =
      rewriter.createOrFold<comb::XorOp>(op.getLoc(), op.cond().getType(), ops);
  op.setOperand(cond);

  auto *thenBlock = op.getThenBlock(), *elseBlock = op.getElseBlock();

  // Move the body of the then block over to the else.
  thenBlock->getOperations().splice(thenBlock->end(),
                                    elseBlock->getOperations());
  rewriter.eraseBlock(elseBlock);
  return success();
}

//===----------------------------------------------------------------------===//
// AlwaysOp

AlwaysOp::Condition AlwaysOp::getCondition(size_t idx) {
  return Condition{EventControl(events()[idx].cast<IntegerAttr>().getInt()),
                   getOperand(idx)};
}

void AlwaysOp::build(OpBuilder &odsBuilder, OperationState &result,
                     ArrayRef<EventControl> events, ArrayRef<Value> clocks,
                     std::function<void()> bodyCtor) {
  assert(events.size() == clocks.size() &&
         "mismatch between event and clock list");
  OpBuilder::InsertionGuard guard(odsBuilder);

  SmallVector<Attribute> eventAttrs;
  for (auto event : events)
    eventAttrs.push_back(
        odsBuilder.getI32IntegerAttr(static_cast<int32_t>(event)));
  result.addAttribute("events", odsBuilder.getArrayAttr(eventAttrs));
  result.addOperands(clocks);

  // Set up the body.  Moves the insert point
  odsBuilder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (bodyCtor)
    bodyCtor();
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
static LogicalResult verifyAlwaysOp(AlwaysOp op) {
  if (op.events().size() != op.getNumOperands())
    return op.emitError("different number of operands and events");
  return success();
}

static ParseResult
parseEventList(OpAsmParser &p, Attribute &eventsAttr,
               SmallVectorImpl<OpAsmParser::OperandType> &clocksOperands) {

  // Parse zero or more conditions intoevents and clocksOperands.
  SmallVector<Attribute> events;

  auto loc = p.getCurrentLocation();
  StringRef keyword;
  if (!p.parseOptionalKeyword(&keyword)) {
    while (1) {
      auto kind = symbolizeEventControl(keyword);
      if (!kind.hasValue())
        return p.emitError(loc, "expected 'posedge', 'negedge', or 'edge'");
      auto eventEnum = static_cast<int32_t>(kind.getValue());
      events.push_back(p.getBuilder().getI32IntegerAttr(eventEnum));

      clocksOperands.push_back({});
      if (p.parseOperand(clocksOperands.back()))
        return failure();

      if (failed(p.parseOptionalComma()))
        break;
      if (p.parseKeyword(&keyword))
        return failure();
    }
  }
  eventsAttr = p.getBuilder().getArrayAttr(events);
  return success();
}

static void printEventList(OpAsmPrinter &p, AlwaysOp op, ArrayAttr portsAttr,
                           OperandRange operands) {
  for (size_t i = 0, e = op.getNumConditions(); i != e; ++i) {
    if (i != 0)
      p << ", ";
    auto cond = op.getCondition(i);
    p << stringifyEventControl(cond.event);
    p << ' ';
    p.printOperand(cond.value);
  }
}

//===----------------------------------------------------------------------===//
// AlwaysFFOp

void AlwaysFFOp::build(OpBuilder &odsBuilder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(odsBuilder);

  result.addAttribute("clockEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute(
      "resetStyle",
      odsBuilder.getI32IntegerAttr(static_cast<int32_t>(ResetType::NoReset)));

  // Set up the body.  Moves Insert Point
  odsBuilder.createBlock(result.addRegion());

  if (bodyCtor)
    bodyCtor();

  // Set up the reset region.
  result.addRegion();
}

void AlwaysFFOp::build(OpBuilder &odsBuilder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       ResetType resetStyle, EventControl resetEdge,
                       Value reset, std::function<void()> bodyCtor,
                       std::function<void()> resetCtor) {
  OpBuilder::InsertionGuard guard(odsBuilder);

  result.addAttribute("clockEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute("resetStyle", odsBuilder.getI32IntegerAttr(
                                        static_cast<int32_t>(resetStyle)));
  result.addAttribute("resetEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(resetEdge)));
  result.addOperands(reset);

  // Set up the body.  Moves Insert Point.
  odsBuilder.createBlock(result.addRegion());

  if (bodyCtor)
    bodyCtor();

  // Set up the reset.  Moves Insert Point.
  odsBuilder.createBlock(result.addRegion());

  if (resetCtor)
    resetCtor();
}

//===----------------------------------------------------------------------===//
// InitialOp
//===----------------------------------------------------------------------===//

void InitialOp::build(OpBuilder &odsBuilder, OperationState &result,
                      std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(odsBuilder);

  odsBuilder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (bodyCtor)
    bodyCtor();
}

//===----------------------------------------------------------------------===//
// CaseZOp
//===----------------------------------------------------------------------===//

/// Return the specified bit, bit 0 is the least significant bit.
auto CaseZOp::CasePattern::getBit(size_t bitNumber) const -> PatternBit {
  return PatternBit(unsigned(attr.getValue()[bitNumber * 2]) +
                    2 * unsigned(attr.getValue()[bitNumber * 2 + 1]));
}

bool CaseZOp::CasePattern::isDefault() const {
  for (size_t i = 0, e = getWidth(); i != e; ++i)
    if (getBit(i) != PatternAny)
      return false;
  return true;
}

// Get a CasePattern from a specified list of PatternBits.  Bits are
// specified in most least significant order - element zero is the least
// significant bit.
CaseZOp::CasePattern::CasePattern(ArrayRef<PatternBit> bits,
                                  MLIRContext *context) {
  APInt pattern(bits.size() * 2, 0);
  for (auto elt : llvm::reverse(bits)) {
    pattern <<= 2;
    pattern |= unsigned(elt);
  }
  auto patternType = IntegerType::get(context, bits.size() * 2);
  attr = IntegerAttr::get(patternType, pattern);
}

auto CaseZOp::getCases() -> SmallVector<CaseInfo, 4> {
  SmallVector<CaseInfo, 4> result;
  assert(casePatterns().size() == getNumRegions() &&
         "case pattern / region count mismatch");
  size_t nextRegion = 0;
  for (auto elt : casePatterns()) {
    result.push_back({CasePattern(elt.cast<IntegerAttr>()),
                      &getRegion(nextRegion++).front()});
  }

  return result;
}

static ParseResult parseCaseZOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::OperandType condOperand;
  Type condType;

  auto loc = parser.getCurrentLocation();
  if (parser.parseOperand(condOperand) || parser.parseColonType(condType) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperand(condOperand, condType, result.operands))
    return failure();

  // Check the integer type.
  if (!result.operands[0].getType().isSignlessInteger())
    return parser.emitError(loc, "condition must have signless integer type");
  auto condWidth = condType.getIntOrFloatBitWidth();

  // Parse all the cases.
  SmallVector<Attribute> casePatterns;
  SmallVector<CaseZOp::PatternBit, 16> caseBits;
  while (1) {
    if (succeeded(parser.parseOptionalKeyword("default"))) {
      // Fill the pattern with Any.
      caseBits.assign(condWidth, CaseZOp::PatternAny);
    } else if (failed(parser.parseOptionalKeyword("case"))) {
      // Not default or case, must be the end of the cases.
      break;
    } else {
      // Parse the pattern.  It always starts with b, so it is an MLIR keyword.
      StringRef caseVal;
      loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&caseVal))
        return failure();

      if (caseVal.front() != 'b')
        return parser.emitError(loc, "expected case value starting with 'b'");
      caseVal = caseVal.drop_front();

      // Parse and decode each bit, we reverse the list later for MSB->LSB.
      for (; !caseVal.empty(); caseVal = caseVal.drop_front()) {
        CaseZOp::PatternBit bit;
        switch (caseVal.front()) {
        case '0':
          bit = CaseZOp::PatternZero;
          break;
        case '1':
          bit = CaseZOp::PatternOne;
          break;
        case 'x':
          bit = CaseZOp::PatternAny;
          break;
        default:
          return parser.emitError(loc, "unexpected case bit '")
                 << caseVal.front() << "'";
        }
        caseBits.push_back(bit);
      }

      if (caseVal.size() > condWidth)
        return parser.emitError(loc, "too many bits specified in pattern");
      std::reverse(caseBits.begin(), caseBits.end());

      // High zeros may be missing.
      if (caseBits.size() < condWidth)
        caseBits.append(condWidth - caseBits.size(), CaseZOp::PatternZero);
    }

    auto resultPattern = CaseZOp::CasePattern(caseBits, builder.getContext());
    casePatterns.push_back(resultPattern.attr);
    caseBits.clear();

    // Parse the case body.
    auto caseRegion = std::make_unique<Region>();
    if (parser.parseColon() || parser.parseRegion(*caseRegion))
      return failure();
    result.addRegion(std::move(caseRegion));
  }

  result.addAttribute("casePatterns", builder.getArrayAttr(casePatterns));
  return success();
}

static void printCaseZOp(OpAsmPrinter &p, CaseZOp op) {
  p << "sv.casez" << ' ' << op.cond() << " : " << op.cond().getType();
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"casePatterns"});

  for (auto caseInfo : op.getCases()) {
    p.printNewline();
    auto pattern = caseInfo.pattern;
    if (pattern.isDefault()) {
      p << "default";
    } else {
      p << "case b";
      for (size_t i = 0, e = pattern.getWidth(); i != e; ++i)
        p << CaseZOp::getLetter(pattern.getBit(e - i - 1),
                                /*isVerilog=*/false);
    }

    p << ':';
    p.printRegion(*caseInfo.block->getParent(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

static LogicalResult verifyCaseZOp(CaseZOp op) {
  // Ensure that the number of regions and number of case values match.
  if (op.casePatterns().size() != op.getNumRegions())
    return op.emitOpError("case pattern / region count mismatch");
  return success();
}

//===----------------------------------------------------------------------===//
// TypeDecl operations
//===----------------------------------------------------------------------===//

void InterfaceOp::build(OpBuilder &builder, OperationState &result,
                        StringRef sym_name, std::function<void()> body) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute("sym_name", builder.getStringAttr(sym_name));
  builder.createBlock(result.addRegion());
  if (body)
    body();
}

ModportType InterfaceOp::getModportType(StringRef modportName) {
  assert(lookupSymbol<InterfaceModportOp>(modportName) &&
         "Modport symbol not found.");
  auto *ctxt = getContext();
  return ModportType::get(
      getContext(),
      SymbolRefAttr::get(ctxt, sym_name(),
                         {SymbolRefAttr::get(ctxt, modportName)}));
}

Type InterfaceOp::getSignalType(StringRef signalName) {
  InterfaceSignalOp signal = lookupSymbol<InterfaceSignalOp>(signalName);
  assert(signal && "Interface signal symbol not found.");
  return signal.type();
}

static ParseResult parseModportStructs(OpAsmParser &parser,
                                       ArrayAttr &portsAttr) {
  if (parser.parseLParen())
    return failure();

  auto context = parser.getBuilder().getContext();

  SmallVector<Attribute, 8> ports;
  do {
    StringAttr direction;
    FlatSymbolRefAttr signal;
    if (parser.parseAttribute(direction) || parser.parseAttribute(signal))
      return failure();

    ports.push_back(ModportStructAttr::get(direction, signal, context));
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  portsAttr = ArrayAttr::get(context, ports);
  return success();
}

static void printModportStructs(OpAsmPrinter &p, Operation *,
                                ArrayAttr portsAttr) {
  p << " (";
  llvm::interleaveComma(portsAttr, p, [&](Attribute attr) {
    auto port = attr.cast<ModportStructAttr>();
    p << port.direction();
    p << ' ';
    p.printSymbolName(port.signal().getRootReference());
  });
  p << ')';
}

void InterfaceSignalOp::build(mlir::OpBuilder &builder,
                              ::mlir::OperationState &state, StringRef name,
                              mlir::Type type) {
  build(builder, state, name, mlir::TypeAttr::get(type));
}

void InterfaceModportOp::build(OpBuilder &builder, OperationState &state,
                               StringRef name, ArrayRef<StringRef> inputs,
                               ArrayRef<StringRef> outputs) {
  auto *ctxt = builder.getContext();
  SmallVector<Attribute, 8> directions;
  StringAttr inputDir = StringAttr::get(ctxt, "input");
  StringAttr outputDir = StringAttr::get(ctxt, "output");
  for (auto input : inputs)
    directions.push_back(ModportStructAttr::get(
        inputDir, SymbolRefAttr::get(ctxt, input), ctxt));
  for (auto output : outputs)
    directions.push_back(ModportStructAttr::get(
        outputDir, SymbolRefAttr::get(ctxt, output), ctxt));
  build(builder, state, name, ArrayAttr::get(ctxt, directions));
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
static LogicalResult verifyInterfaceInstanceOp(InterfaceInstanceOp op) {
  auto symtable = SymbolTable::getNearestSymbolTable(op);
  if (!symtable)
    return op.emitError("sv.interface.instance must exist within a region "
                        "which has a symbol table.");
  auto ifaceTy = op.getType();
  auto referencedOp =
      SymbolTable::lookupSymbolIn(symtable, ifaceTy.getInterface());
  if (!referencedOp)
    return op.emitError("Symbol not found: ") << ifaceTy.getInterface() << ".";
  if (!isa<InterfaceOp>(referencedOp))
    return op.emitError("Symbol ")
           << ifaceTy.getInterface() << " is not an InterfaceOp.";
  return success();
}

/// Ensure that the symbol being instantiated exists and is an
/// InterfaceModportOp.
static LogicalResult verifyGetModportOp(GetModportOp op) {
  auto symtable = SymbolTable::getNearestSymbolTable(op);
  if (!symtable)
    return op.emitError("sv.interface.instance must exist within a region "
                        "which has a symbol table.");
  auto ifaceTy = op.getType();
  auto referencedOp =
      SymbolTable::lookupSymbolIn(symtable, ifaceTy.getModport());
  if (!referencedOp)
    return op.emitError("Symbol not found: ") << ifaceTy.getModport() << ".";
  if (!isa<InterfaceModportOp>(referencedOp))
    return op.emitError("Symbol ")
           << ifaceTy.getModport() << " is not an InterfaceModportOp.";
  return success();
}

void GetModportOp::build(OpBuilder &builder, OperationState &state, Value value,
                         StringRef field) {
  auto ifaceTy = value.getType().dyn_cast<InterfaceType>();
  assert(ifaceTy && "GetModportOp expects an InterfaceType.");
  auto fieldAttr = SymbolRefAttr::get(builder.getContext(), field);
  auto modportSym =
      SymbolRefAttr::get(builder.getContext(),
                         ifaceTy.getInterface().getRootReference(), fieldAttr);
  build(builder, state, ModportType::get(builder.getContext(), modportSym),
        value, fieldAttr);
}

void ReadInterfaceSignalOp::build(OpBuilder &builder, OperationState &state,
                                  Value iface, StringRef signalName) {
  auto ifaceTy = iface.getType().dyn_cast<InterfaceType>();
  assert(ifaceTy && "ReadInterfaceSignalOp expects an InterfaceType.");
  auto fieldAttr = SymbolRefAttr::get(builder.getContext(), signalName);
  InterfaceOp ifaceDefOp = SymbolTable::lookupNearestSymbolFrom<InterfaceOp>(
      iface.getDefiningOp(), ifaceTy.getInterface());
  assert(ifaceDefOp &&
         "ReadInterfaceSignalOp could not resolve an InterfaceOp.");
  build(builder, state, ifaceDefOp.getSignalType(signalName), iface, fieldAttr);
}

ParseResult parseIfaceTypeAndSignal(OpAsmParser &p, Type &ifaceTy,
                                    FlatSymbolRefAttr &signalName) {
  SymbolRefAttr fullSym;
  if (p.parseAttribute(fullSym) || fullSym.getNestedReferences().size() != 1)
    return failure();

  auto *ctxt = p.getBuilder().getContext();
  ifaceTy = InterfaceType::get(
      ctxt, FlatSymbolRefAttr::get(ctxt, fullSym.getRootReference()));
  signalName = FlatSymbolRefAttr::get(ctxt, fullSym.getLeafReference());
  return success();
}

void printIfaceTypeAndSignal(OpAsmPrinter &p, Operation *op, Type type,
                             FlatSymbolRefAttr signalName) {
  InterfaceType ifaceTy = type.dyn_cast<InterfaceType>();
  assert(ifaceTy && "Expected an InterfaceType");
  auto sym = SymbolRefAttr::get(op->getContext(),
                                ifaceTy.getInterface().getRootReference(),
                                {signalName});
  p << sym;
}

LogicalResult verifySignalExists(Value ifaceVal, FlatSymbolRefAttr signalName) {
  auto ifaceTy = ifaceVal.getType().dyn_cast<InterfaceType>();
  if (!ifaceTy)
    return failure();
  InterfaceOp iface = SymbolTable::lookupNearestSymbolFrom<InterfaceOp>(
      ifaceVal.getDefiningOp(), ifaceTy.getInterface());
  if (!iface)
    return failure();
  InterfaceSignalOp signal = iface.lookupSymbol<InterfaceSignalOp>(signalName);
  if (!signal)
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// WireOp
//===----------------------------------------------------------------------===//

void WireOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   Type elementType, StringAttr name, StringAttr sym_name) {
  if (!name)
    name = odsBuilder.getStringAttr("");
  if (sym_name)
    odsState.addAttribute("sym_name", sym_name);

  odsState.addAttribute("name", name);
  odsState.addTypes(InOutType::get(elementType));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void WireOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

// If this wire is only written to, delete the wire and all writers.
LogicalResult WireOp::canonicalize(WireOp wire, PatternRewriter &rewriter) {
  // If the wire has a symbol, then we can't delete it.
  if (wire.sym_nameAttr())
    return failure();

  // Wires have inout type, so they'll have connects and read_inout operations
  // that work on them.  If anything unexpected is found then leave it alone.
  SmallVector<sv::ReadInOutOp> reads;
  sv::ConnectOp write;

  for (auto *user : wire->getUsers()) {
    if (auto read = dyn_cast<sv::ReadInOutOp>(user)) {
      reads.push_back(read);
      continue;
    }

    // Otherwise must be a connect, and we must not have seen a write yet.
    auto connect = dyn_cast<sv::ConnectOp>(user);
    // Either the wire has more than one write or another kind of Op (other than
    // ConectOp and ReadInOutOp), then can't optimize.
    if (!connect || write)
      return failure();
    write = connect;
  }

  Value connected;
  if (!write) {
    // If no write and only reads, then replace with XOp.
    connected = rewriter.create<ConstantXOp>(
        wire.getLoc(),
        wire.getResult().getType().cast<InOutType>().getElementType());
  } else if (isa<hw::HWModuleOp>(write->getParentOp()))
    connected = write.src();
  else
    // If the write is happening at the module level then we don't have any
    // use-before-def checking to do, so we only handle that for now.
    return failure();

  // Ok, we can do this.  Replace all the reads with the connected value.
  for (auto read : reads)
    rewriter.replaceOp(read, connected);

  // And remove the write and wire itself.
  if (write)
    rewriter.eraseOp(write);
  rewriter.eraseOp(wire);
  return success();
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
static LogicalResult verifyWireOp(WireOp op) {
  if (!isa<hw::HWModuleOp>(op->getParentOp()))
    return op.emitError("sv.wire must not be in an always or initial block");
  return success();
}

//===----------------------------------------------------------------------===//
// ReadInOutOp
//===----------------------------------------------------------------------===//

void ReadInOutOp::build(OpBuilder &builder, OperationState &result,
                        Value input) {
  auto resultType = input.getType().cast<InOutType>().getElementType();
  build(builder, result, resultType, input);
}

//===----------------------------------------------------------------------===//
// ArrayIndexInOutOp
//===----------------------------------------------------------------------===//

void ArrayIndexInOutOp::build(OpBuilder &builder, OperationState &result,
                              Value input, Value index) {
  auto resultType = input.getType().cast<InOutType>().getElementType();
  resultType = getAnyHWArrayElementType(resultType);
  assert(resultType && "input should have 'inout of an array' type");
  build(builder, result, InOutType::get(resultType), input, index);
}

//===----------------------------------------------------------------------===//
// Other ops.
//===----------------------------------------------------------------------===//

static LogicalResult verifyAliasOp(AliasOp op) {
  // Must have at least two operands.
  if (op.operands().size() < 2)
    return op.emitOpError("alias must have at least two operands");

  return success();
}

//===----------------------------------------------------------------------===//
// PAssignOp
//===----------------------------------------------------------------------===//

// reg s <= cond ? val : s simplification.
// Don't assign a register's value to itself, conditionally assign the new value
// instead.
LogicalResult PAssignOp::canonicalize(PAssignOp op, PatternRewriter &rewriter) {
  auto mux = op.src().getDefiningOp<comb::MuxOp>();
  if (!mux)
    return failure();

  auto reg = dyn_cast<sv::RegOp>(op.dest().getDefiningOp());
  if (!reg)
    return failure();

  bool trueBranch; // did we find the register on the true branch?
  auto tvread = mux.trueValue().getDefiningOp<sv::ReadInOutOp>();
  auto fvread = mux.falseValue().getDefiningOp<sv::ReadInOutOp>();
  if (tvread && reg == tvread.input().getDefiningOp<sv::RegOp>())
    trueBranch = true;
  else if (fvread && reg == fvread.input().getDefiningOp<sv::RegOp>())
    trueBranch = false;
  else
    return failure();

  // Check that this is the only write of the register
  for (auto &use : reg->getUses()) {
    if (isa<ReadInOutOp>(use.getOwner()))
      continue;
    if (use.getOwner() == op)
      continue;
    return failure();
  }

  // Replace a non-blocking procedural assign in a procedural region with a
  // conditional procedural assign.  We've ensured that this is the only write
  // of the register.
  if (trueBranch) {
    auto one =
        rewriter.create<hw::ConstantOp>(mux.getLoc(), mux.cond().getType(), -1);
    Value ops[] = {mux.cond(), one};
    auto cond = rewriter.createOrFold<comb::XorOp>(mux.getLoc(),
                                                   mux.cond().getType(), ops);
    rewriter.create<sv::IfOp>(mux.getLoc(), cond, [&]() {
      rewriter.create<PAssignOp>(op.getLoc(), reg, mux.falseValue());
    });
  } else {
    rewriter.create<sv::IfOp>(mux.getLoc(), mux.cond(), [&]() {
      rewriter.create<PAssignOp>(op.getLoc(), reg, mux.trueValue());
    });
  }

  // Remove the wire.
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// BindOp
//===----------------------------------------------------------------------===//

hw::InstanceOp BindOp::getReferencedInstance() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;

  /// Lookup the instance for the symbol.  This returns null on
  /// invalid IR.
  auto inst = lookupSymbolInNested(topLevelModuleOp, bind());
  return dyn_cast_or_null<hw::InstanceOp>(inst);
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
static LogicalResult verifyBindOp(BindOp op) {
  auto inst = op.getReferencedInstance();
  if (!inst)
    return op.emitError("Referenced instance doesn't exist");
  if (!inst->getAttr("doNotPrint"))
    return op.emitError("Referenced instance isn't marked as doNotPrint");
  return success();
}

//===----------------------------------------------------------------------===//
// Helpers to elide "label" attributes.
//===----------------------------------------------------------------------===//

static ParseResult parseElideLabel(OpAsmParser &p, NamedAttrList &resultAttrs) {

  auto result = p.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("label"))
    resultAttrs.append("label", p.getBuilder().getStringAttr(""));

  return result;
}

static void printElideLabel(OpAsmPrinter &p, Operation *op,
                            DictionaryAttr attr) {

  SmallVector<StringRef, 1> elides;
  // Elide "label" if it is an empty string.
  if (op->getAttrOfType<StringAttr>("label").getValue().empty())
    elides.push_back("label");

  p.printOptionalAttrDict(op->getAttrs(), elides);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.cpp.inc"
#include "circt/Dialect/SV/SVEnums.cpp.inc"
#include "circt/Dialect/SV/SVStructs.cpp.inc"
