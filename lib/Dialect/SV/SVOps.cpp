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
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallString.h"

using namespace circt;
using namespace sv;

/// Return true if the specified operation is an expression.
bool sv::isExpression(Operation *op) {
  return isa<sv::TextualValueOp>(op) || isa<sv::GetModportOp>(op) ||
         isa<sv::ReadInterfaceSignalOp>(op);
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

  auto resultName = parser.getResultName(0);
  if (!resultName.first.empty() && !isdigit(resultName.first[0])) {
    StringRef name = resultName.first;
    auto nameAttr = parser.getBuilder().getStringAttr(name);
    auto *context = parser.getBuilder().getContext();
    resultAttrs.push_back({Identifier::get("name", context), nameAttr});
  }

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
  auto expectedName = op->getAttrOfType<StringAttr>("name");
  if (!expectedName ||
      tmpStream.str().drop_front() != expectedName.getValue()) {
    namesDisagree = true;
  }

  if (namesDisagree)
    p.printOptionalAttrDict(op->getAttrs());
  else
    p.printOptionalAttrDict(op->getAttrs(), {"name"});
}

//===----------------------------------------------------------------------===//
// RegOp
//===----------------------------------------------------------------------===//

void RegOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type elementType, StringAttr name) {
  if (name)
    odsState.addAttribute("name", name);

  odsState.addTypes(rtl::InOutType::get(elementType));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void RegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  if (auto nameAttr = (*this)->getAttrOfType<StringAttr>("name"))
    setNameFn(getResult(), nameAttr.getValue());
}

void RegOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  // If this wire is only written to, delete the wire and all writers.
  struct DropDeadConnect final : public OpRewritePattern<RegOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(RegOp op,
                                  PatternRewriter &rewriter) const override {

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
  };
  results.insert<DropDeadConnect>(context);
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
  result.addAttribute("cond", cond);
  Region *thenRegion = result.addRegion();
  IfDefOp::ensureTerminator(*thenRegion, odsBuilder, result.location);

  // Fill in the body of the #ifdef.
  if (thenCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*thenRegion->begin());
    thenCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    IfDefOp::ensureTerminator(*elseRegion, odsBuilder, result.location);
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*elseRegion->begin());
    elseCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
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
  result.addOperands(cond);
  Region *body = result.addRegion();
  IfOp::ensureTerminator(*body, odsBuilder, result.location);

  // Fill in the body of the #ifdef.
  if (thenCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*body->begin());
    thenCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    IfDefOp::ensureTerminator(*elseRegion, odsBuilder, result.location);
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*elseRegion->begin());
    elseCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
}

/// Replaces the given op with the contents of the given single-block region.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  rewriter.mergeBlockBefore(block, op);
  rewriter.eraseOp(op);
  rewriter.eraseOp(terminator);
}

void IfOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                       MLIRContext *context) {
  struct RemoveStaticCondition : public OpRewritePattern<IfOp> {
    using OpRewritePattern<IfOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(IfOp op,
                                  PatternRewriter &rewriter) const override {
      auto constant = op.cond().getDefiningOp<rtl::ConstantOp>();
      if (!constant)
        return failure();

      if (constant.getValue().isAllOnesValue())
        replaceOpWithRegion(rewriter, op, op.thenRegion());
      else if (!op.elseRegion().empty())
        replaceOpWithRegion(rewriter, op, op.elseRegion());
      else
        rewriter.eraseOp(op);

      return success();
    }
  };
  results.insert<RemoveStaticCondition>(context);
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

  SmallVector<Attribute> eventAttrs;
  for (auto event : events)
    eventAttrs.push_back(
        odsBuilder.getI32IntegerAttr(static_cast<int32_t>(event)));
  result.addAttribute("events", odsBuilder.getArrayAttr(eventAttrs));
  result.addOperands(clocks);

  // Set up the body.
  Region *body = result.addRegion();
  AlwaysOp::ensureTerminator(*body, odsBuilder, result.location);

  // Fill in the body of the #ifdef.
  if (bodyCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*body->begin());
    bodyCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
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
  result.addAttribute("clockEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute(
      "resetStyle",
      odsBuilder.getI32IntegerAttr(static_cast<int32_t>(ResetType::NoReset)));

  // Set up the body.
  Region *bodyBlk = result.addRegion();
  AlwaysFFOp::ensureTerminator(*bodyBlk, odsBuilder, result.location);

  if (bodyCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*bodyBlk->begin());
    bodyCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }

  // Set up the reset region.
  result.addRegion();
}

void AlwaysFFOp::build(OpBuilder &odsBuilder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       ResetType resetStyle, EventControl resetEdge,
                       Value reset, std::function<void()> bodyCtor,
                       std::function<void()> resetCtor) {
  result.addAttribute("clockEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute("resetStyle", odsBuilder.getI32IntegerAttr(
                                        static_cast<int32_t>(resetStyle)));
  result.addAttribute("resetEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(resetEdge)));
  result.addOperands(reset);

  // Set up the body.
  Region *bodyRegion = result.addRegion();

  AlwaysFFOp::ensureTerminator(*bodyRegion, odsBuilder, result.location);
  if (bodyCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*bodyRegion->begin());
    bodyCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }

  // Set up the reset.
  Region *resetRegion = result.addRegion();

  AlwaysFFOp::ensureTerminator(*resetRegion, odsBuilder, result.location);
  if (resetCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*resetRegion->begin());
    resetCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
}

//===----------------------------------------------------------------------===//
// InitialOp
//===----------------------------------------------------------------------===//

void InitialOp::build(OpBuilder &odsBuilder, OperationState &result,
                      std::function<void()> bodyCtor) {
  Region *body = result.addRegion();
  InitialOp::ensureTerminator(*body, odsBuilder, result.location);

  // Fill in the body of the #ifdef.
  if (bodyCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*body->begin());
    bodyCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
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
    CaseZOp::ensureTerminator(*caseRegion, builder, result.location);
    result.addRegion(std::move(caseRegion));
  }

  result.addAttribute("casePatterns", builder.getArrayAttr(casePatterns));
  return success();
}

static void printCaseZOp(OpAsmPrinter &p, CaseZOp op) {
  p << "sv.casez" << ' ' << op.cond() << " : " << op.cond().getType();
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"casePatterns"});

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
    bool printTerminator = true;
    if (auto *term = caseInfo.block->getTerminator()) {
      printTerminator =
          !term->getAttrDictionary().empty() || term->getNumOperands() != 0;
    }
    p.printRegion(*caseInfo.block->getParent(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printTerminator);
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

ModportType InterfaceOp::getModportType(StringRef modportName) {
  InterfaceModportOp modportOp = lookupSymbol<InterfaceModportOp>(modportName);
  assert(modportOp && "Modport symbol not found.");
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
                   Type elementType, StringAttr name) {
  if (name)
    odsState.addAttribute("name", name);

  odsState.addTypes(InOutType::get(elementType));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void WireOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  if (auto nameAttr = (*this)->getAttrOfType<StringAttr>("name"))
    setNameFn(getResult(), nameAttr.getValue());
}

void WireOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  // If this wire is only written to, delete the wire and all writers.
  struct DropDeadConnect final : public OpRewritePattern<WireOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(WireOp op,
                                  PatternRewriter &rewriter) const override {

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
  };
  results.insert<DropDeadConnect>(context);
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
static LogicalResult verifyWireOp(WireOp op) {
  if (!isa<rtl::RTLModuleOp>(op->getParentOp()))
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
  resultType = getAnyRTLArrayElementType(resultType);
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
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.cpp.inc"
#include "circt/Dialect/SV/SVEnums.cpp.inc"
#include "circt/Dialect/SV/SVStructs.cpp.inc"
