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
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/CustomDirectiveImpl.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace circt;
using namespace sv;
using mlir::TypedAttr;

/// Return true if the specified expression is 2-state.  This is determined by
/// looking at the defining op.  This can look as far through the dataflow as it
/// wants, but for now, it is just looking at the single value.
bool sv::is2StateExpression(Value v) {
  if (auto *op = v.getDefiningOp()) {
    if (auto attr = op->getAttrOfType<UnitAttr>("twoState"))
      return (bool)attr;
  }
  // Plain constants are obviously safe
  return v.getDefiningOp<hw::ConstantOp>();
}

/// Return true if the specified operation is an expression.
bool sv::isExpression(Operation *op) {
  return isa<VerbatimExprOp, VerbatimExprSEOp, GetModportOp,
             ReadInterfaceSignalOp, ConstantXOp, ConstantZOp, ConstantStrOp,
             MacroRefExprOp, MacroRefExprSEOp>(op);
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
  StringAttr symbolNameId = StringAttr::get(symbolTableOp->getContext(),
                                            SymbolTable::getSymbolAttrName());
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

/// Verifies symbols referenced by macro identifiers.
static LogicalResult
verifyMacroIdentSymbolUses(Operation *op, FlatSymbolRefAttr attr,
                           SymbolTableCollection &symbolTable) {
  auto *refOp = symbolTable.lookupNearestSymbolFrom(op, attr);
  if (!refOp)
    return op->emitError("references an undefined symbol: ") << attr;
  if (!isa<MacroDeclOp>(refOp))
    return op->emitError("must reference a macro declaration");
  return success();
}

//===----------------------------------------------------------------------===//
// VerbatimExprOp
//===----------------------------------------------------------------------===//

/// Get the asm name for sv.verbatim.expr and sv.verbatim.expr.se.
static void
getVerbatimExprAsmResultNames(Operation *op,
                              function_ref<void(Value, StringRef)> setNameFn) {
  // If the string is macro like, then use a pretty name.  We only take the
  // string up to a weird character (like a paren) and currently ignore
  // parenthesized expressions.
  auto isOkCharacter = [](char c) { return llvm::isAlnum(c) || c == '_'; };
  auto name = op->getAttrOfType<StringAttr>("format_string").getValue();
  // Ignore a leading ` in macro name.
  if (name.starts_with("`"))
    name = name.drop_front();
  name = name.take_while(isOkCharacter);
  if (!name.empty())
    setNameFn(op->getResult(0), name);
}

void VerbatimExprOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getVerbatimExprAsmResultNames(getOperation(), std::move(setNameFn));
}

void VerbatimExprSEOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getVerbatimExprAsmResultNames(getOperation(), std::move(setNameFn));
}

//===----------------------------------------------------------------------===//
// MacroRefExprOp
//===----------------------------------------------------------------------===//

void MacroRefExprOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getMacroName());
}

void MacroRefExprSEOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getMacroName());
}

static MacroDeclOp getReferencedMacro(const hw::HWSymbolCache *cache,
                                      Operation *op,
                                      FlatSymbolRefAttr macroName) {
  if (cache)
    if (auto *result = cache->getDefinition(macroName.getAttr()))
      return cast<MacroDeclOp>(result);

  auto topLevelModuleOp = op->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol<MacroDeclOp>(macroName.getValue());
}

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
MacroDeclOp MacroRefExprOp::getReferencedMacro(const hw::HWSymbolCache *cache) {
  return ::getReferencedMacro(cache, *this, getMacroNameAttr());
}

MacroDeclOp
MacroRefExprSEOp::getReferencedMacro(const hw::HWSymbolCache *cache) {
  return ::getReferencedMacro(cache, *this, getMacroNameAttr());
}

//===----------------------------------------------------------------------===//
// MacroErrorOp
//===----------------------------------------------------------------------===//

std::string MacroErrorOp::getMacroIdentifier() {
  const auto *prefix = "_ERROR";
  auto msg = getMessage();
  if (!msg || msg->empty())
    return prefix;

  std::string id(prefix);
  id.push_back('_');
  for (auto c : *msg) {
    if (llvm::isAlnum(c))
      id.push_back(c);
    else
      id.push_back('_');
  }
  return id;
}

//===----------------------------------------------------------------------===//
// MacroDeclOp
//===----------------------------------------------------------------------===//

MacroDeclOp MacroDefOp::getReferencedMacro(const hw::HWSymbolCache *cache) {
  return ::getReferencedMacro(cache, *this, getMacroNameAttr());
}

MacroDeclOp MacroRefOp::getReferencedMacro(const hw::HWSymbolCache *cache) {
  return ::getReferencedMacro(cache, *this, getMacroNameAttr());
}

/// Ensure that the symbol being instantiated exists and is a MacroDefOp.
LogicalResult
MacroRefExprOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMacroIdentSymbolUses(*this, getMacroNameAttr(), symbolTable);
}

/// Ensure that the symbol being instantiated exists and is a MacroDefOp.
LogicalResult
MacroRefExprSEOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMacroIdentSymbolUses(*this, getMacroNameAttr(), symbolTable);
}

/// Ensure that the symbol being instantiated exists and is a MacroDefOp.
LogicalResult MacroDefOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMacroIdentSymbolUses(*this, getMacroNameAttr(), symbolTable);
}

/// Ensure that the symbol being instantiated exists and is a MacroDefOp.
LogicalResult MacroRefOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMacroIdentSymbolUses(*this, getMacroNameAttr(), symbolTable);
}

//===----------------------------------------------------------------------===//
// MacroDeclOp
//===----------------------------------------------------------------------===//

StringRef MacroDeclOp::getMacroIdentifier() {
  return getVerilogName().value_or(getSymName());
}

//===----------------------------------------------------------------------===//
// ConstantXOp / ConstantZOp
//===----------------------------------------------------------------------===//

void ConstantXOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "x_i" << getWidth();
  setNameFn(getResult(), specialName.str());
}

LogicalResult ConstantXOp::verify() {
  // We don't allow zero width constant or unknown width.
  if (getWidth() <= 0)
    return emitError("unsupported type");
  return success();
}

void ConstantZOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "z_i" << getWidth();
  setNameFn(getResult(), specialName.str());
}

LogicalResult ConstantZOp::verify() {
  // We don't allow zero width constant or unknown type.
  if (getWidth() <= 0)
    return emitError("unsupported type");
  return success();
}

//===----------------------------------------------------------------------===//
// LocalParamOp
//===----------------------------------------------------------------------===//

void LocalParamOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the localparam has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

LogicalResult LocalParamOp::verify() {
  // Verify that this is a valid parameter value.
  return hw::checkParameterInContext(
      getValue(), (*this)->getParentOfType<hw::HWModuleOp>(), *this);
}

//===----------------------------------------------------------------------===//
// RegOp
//===----------------------------------------------------------------------===//

static ParseResult
parseImplicitInitType(OpAsmParser &p, mlir::Type regType,
                      std::optional<OpAsmParser::UnresolvedOperand> &initValue,
                      mlir::Type &initType) {
  if (!initValue.has_value())
    return success();

  hw::InOutType ioType = dyn_cast<hw::InOutType>(regType);
  if (!ioType)
    return p.emitError(p.getCurrentLocation(), "expected inout type for reg");

  initType = ioType.getElementType();
  return success();
}

static void printImplicitInitType(OpAsmPrinter &p, Operation *op,
                                  mlir::Type regType, mlir::Value initValue,
                                  mlir::Type initType) {}

void RegOp::build(OpBuilder &builder, OperationState &odsState,
                  Type elementType, StringAttr name, hw::InnerSymAttr innerSym,
                  mlir::Value initValue) {
  if (!name)
    name = builder.getStringAttr("");
  odsState.addAttribute("name", name);
  if (innerSym)
    odsState.addAttribute(hw::InnerSymbolTable::getInnerSymbolAttrName(),
                          innerSym);
  odsState.addTypes(hw::InOutType::get(elementType));
  if (initValue)
    odsState.addOperands(initValue);
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void RegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

std::optional<size_t> RegOp::getTargetResultIndex() { return 0; }

// If this reg is only written to, delete the reg and all writers.
LogicalResult RegOp::canonicalize(RegOp op, PatternRewriter &rewriter) {
  // Block if op has SV attributes.
  if (hasSVAttributes(op))
    return failure();

  // If the reg has a symbol, then we can't delete it.
  if (op.getInnerSymAttr())
    return failure();
  // Check that all operations on the wire are sv.assigns. All other wire
  // operations will have been handled by other canonicalization.
  for (auto *user : op.getResult().getUsers())
    if (!isa<AssignOp>(user))
      return failure();

  // Remove all uses of the wire.
  for (auto *user : llvm::make_early_inc_range(op.getResult().getUsers()))
    rewriter.eraseOp(user);

  // Remove the wire.
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// LogicOp
//===----------------------------------------------------------------------===//

void LogicOp::build(OpBuilder &builder, OperationState &odsState,
                    Type elementType, StringAttr name,
                    hw::InnerSymAttr innerSym) {
  if (!name)
    name = builder.getStringAttr("");
  odsState.addAttribute("name", name);
  if (innerSym)
    odsState.addAttribute(hw::InnerSymbolTable::getInnerSymbolAttrName(),
                          innerSym);
  odsState.addTypes(hw::InOutType::get(elementType));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void LogicOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the logic has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

std::optional<size_t> LogicOp::getTargetResultIndex() { return 0; }

//===----------------------------------------------------------------------===//
// Control flow like-operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IfDefOp
//===----------------------------------------------------------------------===//

void IfDefOp::build(OpBuilder &builder, OperationState &result, StringRef cond,
                    std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  build(builder, result, builder.getStringAttr(cond), std::move(thenCtor),
        std::move(elseCtor));
}

void IfDefOp::build(OpBuilder &builder, OperationState &result, StringAttr cond,
                    std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  build(builder, result, FlatSymbolRefAttr::get(builder.getContext(), cond),
        std::move(thenCtor), std::move(elseCtor));
}

void IfDefOp::build(OpBuilder &builder, OperationState &result,
                    FlatSymbolRefAttr cond, std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  build(builder, result, MacroIdentAttr::get(builder.getContext(), cond),
        std::move(thenCtor), std::move(elseCtor));
}

void IfDefOp::build(OpBuilder &builder, OperationState &result,
                    MacroIdentAttr cond, std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute("cond", cond);
  builder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (thenCtor)
    thenCtor();

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    builder.createBlock(elseRegion);
    elseCtor();
  }
}

LogicalResult IfDefOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMacroIdentSymbolUses(*this, getCond().getIdent(), symbolTable);
}

// If both thenRegion and elseRegion are empty, erase op.
template <class Op>
static LogicalResult canonicalizeIfDefLike(Op op, PatternRewriter &rewriter) {
  if (!op.getThenBlock()->empty())
    return failure();

  if (op.hasElse() && !op.getElseBlock()->empty())
    return failure();

  rewriter.eraseOp(op);
  return success();
}

LogicalResult IfDefOp::canonicalize(IfDefOp op, PatternRewriter &rewriter) {
  return canonicalizeIfDefLike(op, rewriter);
}

//===----------------------------------------------------------------------===//
// IfDefProceduralOp
//===----------------------------------------------------------------------===//

void IfDefProceduralOp::build(OpBuilder &builder, OperationState &result,
                              StringRef cond, std::function<void()> thenCtor,
                              std::function<void()> elseCtor) {
  build(builder, result, builder.getStringAttr(cond), std::move(thenCtor),
        std::move(elseCtor));
}

void IfDefProceduralOp::build(OpBuilder &builder, OperationState &result,
                              StringAttr cond, std::function<void()> thenCtor,
                              std::function<void()> elseCtor) {
  build(builder, result, FlatSymbolRefAttr::get(builder.getContext(), cond),
        std::move(thenCtor), std::move(elseCtor));
}

void IfDefProceduralOp::build(OpBuilder &builder, OperationState &result,
                              FlatSymbolRefAttr cond,
                              std::function<void()> thenCtor,
                              std::function<void()> elseCtor) {
  build(builder, result, MacroIdentAttr::get(builder.getContext(), cond),
        std::move(thenCtor), std::move(elseCtor));
}

void IfDefProceduralOp::build(OpBuilder &builder, OperationState &result,
                              MacroIdentAttr cond,
                              std::function<void()> thenCtor,
                              std::function<void()> elseCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute("cond", cond);
  builder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (thenCtor)
    thenCtor();

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    builder.createBlock(elseRegion);
    elseCtor();
  }
}

LogicalResult IfDefProceduralOp::canonicalize(IfDefProceduralOp op,
                                              PatternRewriter &rewriter) {
  return canonicalizeIfDefLike(op, rewriter);
}

LogicalResult
IfDefProceduralOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMacroIdentSymbolUses(*this, getCond().getIdent(), symbolTable);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 std::function<void()> thenCtor,
                 std::function<void()> elseCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(cond);
  builder.createBlock(result.addRegion());

  // Fill in the body of the if.
  if (thenCtor)
    thenCtor();

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    builder.createBlock(elseRegion);
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
  // Block if op has SV attributes.
  if (hasSVAttributes(op))
    return failure();

  if (auto constant = op.getCond().getDefiningOp<hw::ConstantOp>()) {

    if (constant.getValue().isAllOnes())
      replaceOpWithRegion(rewriter, op, op.getThenRegion());
    else if (!op.getElseRegion().empty())
      replaceOpWithRegion(rewriter, op, op.getElseRegion());

    rewriter.eraseOp(op);

    return success();
  }

  // Erase empty if-else block.
  if (!op.getThenBlock()->empty() && op.hasElse() &&
      op.getElseBlock()->empty()) {
    rewriter.eraseBlock(op.getElseBlock());
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
  // region if the condition is a 2-state operation.  This changes x prop
  // behavior so it needs to be guarded.
  if (is2StateExpression(op.getCond())) {
    auto cond = comb::createOrFoldNot(op.getLoc(), op.getCond(), rewriter);
    op.setOperand(cond);

    auto *thenBlock = op.getThenBlock(), *elseBlock = op.getElseBlock();

    // Move the body of the then block over to the else.
    thenBlock->getOperations().splice(thenBlock->end(),
                                      elseBlock->getOperations());
    rewriter.eraseBlock(elseBlock);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AlwaysOp
//===----------------------------------------------------------------------===//

AlwaysOp::Condition AlwaysOp::getCondition(size_t idx) {
  return Condition{EventControl(cast<IntegerAttr>(getEvents()[idx]).getInt()),
                   getOperand(idx)};
}

void AlwaysOp::build(OpBuilder &builder, OperationState &result,
                     ArrayRef<sv::EventControl> events, ArrayRef<Value> clocks,
                     std::function<void()> bodyCtor) {
  assert(events.size() == clocks.size() &&
         "mismatch between event and clock list");
  OpBuilder::InsertionGuard guard(builder);

  SmallVector<Attribute> eventAttrs;
  for (auto event : events)
    eventAttrs.push_back(
        builder.getI32IntegerAttr(static_cast<int32_t>(event)));
  result.addAttribute("events", builder.getArrayAttr(eventAttrs));
  result.addOperands(clocks);

  // Set up the body.  Moves the insert point
  builder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (bodyCtor)
    bodyCtor();
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult AlwaysOp::verify() {
  if (getEvents().size() != getNumOperands())
    return emitError("different number of operands and events");
  return success();
}

static ParseResult parseEventList(
    OpAsmParser &p, Attribute &eventsAttr,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &clocksOperands) {

  // Parse zero or more conditions intoevents and clocksOperands.
  SmallVector<Attribute> events;

  auto loc = p.getCurrentLocation();
  StringRef keyword;
  if (!p.parseOptionalKeyword(&keyword)) {
    while (1) {
      auto kind = sv::symbolizeEventControl(keyword);
      if (!kind.has_value())
        return p.emitError(loc, "expected 'posedge', 'negedge', or 'edge'");
      auto eventEnum = static_cast<int32_t>(*kind);
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
//===----------------------------------------------------------------------===//

void AlwaysFFOp::build(OpBuilder &builder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute(
      "clockEdge", builder.getI32IntegerAttr(static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute(
      "resetStyle",
      builder.getI32IntegerAttr(static_cast<int32_t>(ResetType::NoReset)));

  // Set up the body.  Moves Insert Point
  builder.createBlock(result.addRegion());

  if (bodyCtor)
    bodyCtor();

  // Set up the reset region.
  result.addRegion();
}

void AlwaysFFOp::build(OpBuilder &builder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       ResetType resetStyle, EventControl resetEdge,
                       Value reset, std::function<void()> bodyCtor,
                       std::function<void()> resetCtor) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute(
      "clockEdge", builder.getI32IntegerAttr(static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute("resetStyle", builder.getI32IntegerAttr(
                                        static_cast<int32_t>(resetStyle)));
  result.addAttribute(
      "resetEdge", builder.getI32IntegerAttr(static_cast<int32_t>(resetEdge)));
  result.addOperands(reset);

  // Set up the body.  Moves Insert Point.
  builder.createBlock(result.addRegion());

  if (bodyCtor)
    bodyCtor();

  // Set up the reset.  Moves Insert Point.
  builder.createBlock(result.addRegion());

  if (resetCtor)
    resetCtor();
}

//===----------------------------------------------------------------------===//
// AlwaysCombOp
//===----------------------------------------------------------------------===//

void AlwaysCombOp::build(OpBuilder &builder, OperationState &result,
                         std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(builder);

  builder.createBlock(result.addRegion());

  if (bodyCtor)
    bodyCtor();
}

//===----------------------------------------------------------------------===//
// InitialOp
//===----------------------------------------------------------------------===//

void InitialOp::build(OpBuilder &builder, OperationState &result,
                      std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(builder);

  builder.createBlock(result.addRegion());

  // Fill in the body of the #ifdef.
  if (bodyCtor)
    bodyCtor();
}

//===----------------------------------------------------------------------===//
// CaseOp
//===----------------------------------------------------------------------===//

/// Return the letter for the specified pattern bit, e.g. "0", "1", "x" or "z".
char sv::getLetter(CasePatternBit bit) {
  switch (bit) {
  case CasePatternBit::Zero:
    return '0';
  case CasePatternBit::One:
    return '1';
  case CasePatternBit::AnyX:
    return 'x';
  case CasePatternBit::AnyZ:
    return 'z';
  }
  llvm_unreachable("invalid casez PatternBit");
}

/// Return the specified bit, bit 0 is the least significant bit.
auto CaseBitPattern::getBit(size_t bitNumber) const -> CasePatternBit {
  return CasePatternBit(unsigned(intAttr.getValue()[bitNumber * 2]) +
                        2 * unsigned(intAttr.getValue()[bitNumber * 2 + 1]));
}

bool CaseBitPattern::hasX() const {
  for (size_t i = 0, e = getWidth(); i != e; ++i)
    if (getBit(i) == CasePatternBit::AnyX)
      return true;
  return false;
}

bool CaseBitPattern::hasZ() const {
  for (size_t i = 0, e = getWidth(); i != e; ++i)
    if (getBit(i) == CasePatternBit::AnyZ)
      return true;
  return false;
}
static SmallVector<CasePatternBit> getPatternBitsForValue(const APInt &value) {
  SmallVector<CasePatternBit> result;
  result.reserve(value.getBitWidth());
  for (size_t i = 0, e = value.getBitWidth(); i != e; ++i)
    result.push_back(CasePatternBit(value[i]));

  return result;
}

// Get a CaseBitPattern from a specified list of PatternBits.  Bits are
// specified in most least significant order - element zero is the least
// significant bit.
CaseBitPattern::CaseBitPattern(const APInt &value, MLIRContext *context)
    : CaseBitPattern(getPatternBitsForValue(value), context) {}

// Get a CaseBitPattern from a specified list of PatternBits.  Bits are
// specified in most least significant order - element zero is the least
// significant bit.
CaseBitPattern::CaseBitPattern(ArrayRef<CasePatternBit> bits,
                               MLIRContext *context)
    : CasePattern(CPK_bit) {
  APInt pattern(bits.size() * 2, 0);
  for (auto elt : llvm::reverse(bits)) {
    pattern <<= 2;
    pattern |= unsigned(elt);
  }
  auto patternType = IntegerType::get(context, bits.size() * 2);
  intAttr = IntegerAttr::get(patternType, pattern);
}

auto CaseOp::getCases() -> SmallVector<CaseInfo, 4> {
  SmallVector<CaseInfo, 4> result;
  assert(getCasePatterns().size() == getNumRegions() &&
         "case pattern / region count mismatch");
  size_t nextRegion = 0;
  for (auto elt : getCasePatterns()) {
    llvm::TypeSwitch<Attribute>(elt)
        .Case<hw::EnumFieldAttr>([&](auto enumAttr) {
          result.push_back({std::make_unique<CaseEnumPattern>(enumAttr),
                            &getRegion(nextRegion++).front()});
        })
        .Case<CaseExprPatternAttr>([&](auto exprAttr) {
          result.push_back({std::make_unique<CaseExprPattern>(getContext()),
                            &getRegion(nextRegion++).front()});
        })
        .Case<IntegerAttr>([&](auto intAttr) {
          result.push_back({std::make_unique<CaseBitPattern>(intAttr),
                            &getRegion(nextRegion++).front()});
        })
        .Case<CaseDefaultPattern::AttrType>([&](auto) {
          result.push_back({std::make_unique<CaseDefaultPattern>(getContext()),
                            &getRegion(nextRegion++).front()});
        })
        .Default([](auto) {
          assert(false && "invalid case pattern attribute type");
        });
  }

  return result;
}

StringRef CaseEnumPattern::getFieldValue() const {
  return cast<hw::EnumFieldAttr>(enumAttr).getField();
}

/// Parse case op.
/// case op ::= `sv.case` case-style? validation-qualifier? cond `:` type
///             attr-dict case-pattern^*
/// case-style ::= `case` | `casex` | `casez`
/// validation-qualifier (see SV Spec 12.5.3) ::= `unique` | `unique0`
///                                             | `priority`
/// case-pattern ::= `case` bit-pattern `:` region
ParseResult CaseOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::UnresolvedOperand condOperand;
  Type condType;

  auto loc = parser.getCurrentLocation();

  StringRef keyword;
  if (!parser.parseOptionalKeyword(&keyword, {"case", "casex", "casez"})) {
    auto kind = symbolizeCaseStmtType(keyword);
    auto caseEnum = static_cast<int32_t>(kind.value());
    result.addAttribute("caseStyle", builder.getI32IntegerAttr(caseEnum));
  }

  // Parse validation qualifier.
  if (!parser.parseOptionalKeyword(
          &keyword, {"plain", "priority", "unique", "unique0"})) {
    auto kind = symbolizeValidationQualifierTypeEnum(keyword);
    result.addAttribute("validationQualifier",
                        ValidationQualifierTypeEnumAttr::get(
                            builder.getContext(), kind.value()));
  }

  if (parser.parseOperand(condOperand) || parser.parseColonType(condType) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperand(condOperand, condType, result.operands))
    return failure();

  // Check the integer type.
  Type canonicalCondType = hw::getCanonicalType(condType);
  hw::EnumType enumType = dyn_cast<hw::EnumType>(canonicalCondType);
  unsigned condWidth = 0;
  if (!enumType) {
    if (!result.operands[0].getType().isSignlessInteger())
      return parser.emitError(loc, "condition must have signless integer type");
    condWidth = condType.getIntOrFloatBitWidth();
  }

  // Parse all the cases.
  SmallVector<Attribute> casePatterns;
  SmallVector<CasePatternBit, 16> caseBits;
  while (1) {
    mlir::OptionalParseResult caseValueParseResult;
    OpAsmParser::UnresolvedOperand caseValueOperand;
    if (succeeded(parser.parseOptionalKeyword("default"))) {
      casePatterns.push_back(CaseDefaultPattern(parser.getContext()).attr());
    } else if (failed(parser.parseOptionalKeyword("case"))) {
      // Not default or case, must be the end of the cases.
      break;
    } else if (enumType) {
      // Enumerated case; parse the case value.
      StringRef caseVal;

      if (parser.parseKeyword(&caseVal))
        return failure();

      if (!enumType.contains(caseVal))
        return parser.emitError(loc)
               << "case value '" + caseVal + "' is not a member of enum type "
               << enumType;
      casePatterns.push_back(
          hw::EnumFieldAttr::get(parser.getEncodedSourceLoc(loc),
                                 builder.getStringAttr(caseVal), condType));
    } else if ((caseValueParseResult =
                    parser.parseOptionalOperand(caseValueOperand))
                   .has_value()) {
      if (failed(caseValueParseResult.value()) ||
          parser.resolveOperand(caseValueOperand, condType, result.operands))
        return failure();
      casePatterns.push_back(CaseExprPattern(parser.getContext()).attr());
    } else {
      // Parse the pattern.  It always starts with b, so it is an MLIR
      // keyword.
      StringRef caseVal;
      loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&caseVal))
        return failure();

      if (caseVal.front() != 'b')
        return parser.emitError(loc, "expected case value starting with 'b'");
      caseVal = caseVal.drop_front();

      // Parse and decode each bit, we reverse the list later for MSB->LSB.
      for (; !caseVal.empty(); caseVal = caseVal.drop_front()) {
        CasePatternBit bit;
        switch (caseVal.front()) {
        case '0':
          bit = CasePatternBit::Zero;
          break;
        case '1':
          bit = CasePatternBit::One;
          break;
        case 'x':
          bit = CasePatternBit::AnyX;
          break;
        case 'z':
          bit = CasePatternBit::AnyZ;
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
        caseBits.append(condWidth - caseBits.size(), CasePatternBit::Zero);

      auto resultPattern = CaseBitPattern(caseBits, builder.getContext());
      casePatterns.push_back(resultPattern.attr());
      caseBits.clear();
    }

    // Parse the case body.
    auto caseRegion = std::make_unique<Region>();
    if (parser.parseColon() || parser.parseRegion(*caseRegion))
      return failure();
    result.addRegion(std::move(caseRegion));
  }

  result.addAttribute("casePatterns", builder.getArrayAttr(casePatterns));
  return success();
}

void CaseOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getCaseStyle() == CaseStmtType::CaseXStmt)
    p << "casex ";
  else if (getCaseStyle() == CaseStmtType::CaseZStmt)
    p << "casez ";

  if (getValidationQualifier() !=
      ValidationQualifierTypeEnum::ValidationQualifierPlain)
    p << stringifyValidationQualifierTypeEnum(getValidationQualifier()) << ' ';

  p << getCond() << " : " << getCond().getType();
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"casePatterns", "caseStyle", "validationQualifier"});

  size_t caseValueIndex = 0;
  for (auto &caseInfo : getCases()) {
    p.printNewline();
    auto &pattern = caseInfo.pattern;

    llvm::TypeSwitch<CasePattern *>(pattern.get())
        .Case<CaseBitPattern>([&](auto bitPattern) {
          p << "case b";
          for (size_t bit = 0, e = bitPattern->getWidth(); bit != e; ++bit)
            p << getLetter(bitPattern->getBit(e - bit - 1));
        })
        .Case<CaseEnumPattern>([&](auto enumPattern) {
          p << "case " << enumPattern->getFieldValue();
        })
        .Case<CaseExprPattern>([&](auto) {
          p << "case ";
          p.printOperand(getCaseValues()[caseValueIndex++]);
        })
        .Case<CaseDefaultPattern>([&](auto) { p << "default"; })
        .Default([&](auto) { assert(false && "unhandled case pattern"); });

    p << ": ";
    p.printRegion(*caseInfo.block->getParent(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

LogicalResult CaseOp::verify() {
  if (!(hw::isHWIntegerType(getCond().getType()) ||
        hw::isHWEnumType(getCond().getType())))
    return emitError("condition must have either integer or enum type");

  // Ensure that the number of regions and number of case values match.
  if (getCasePatterns().size() != getNumRegions())
    return emitOpError("case pattern / region count mismatch");
  return success();
}

/// This ctor allows you to build a CaseZ with some number of cases, getting
/// a callback for each case.
void CaseOp::build(
    OpBuilder &builder, OperationState &result, CaseStmtType caseStyle,
    ValidationQualifierTypeEnum validationQualifier, Value cond,
    size_t numCases,
    std::function<std::unique_ptr<CasePattern>(size_t)> caseCtor) {
  result.addOperands(cond);
  result.addAttribute("caseStyle",
                      CaseStmtTypeAttr::get(builder.getContext(), caseStyle));
  result.addAttribute("validationQualifier",
                      ValidationQualifierTypeEnumAttr::get(
                          builder.getContext(), validationQualifier));
  SmallVector<Attribute> casePatterns;

  OpBuilder::InsertionGuard guard(builder);

  // Fill in the cases with the callback.
  for (size_t i = 0, e = numCases; i != e; ++i) {
    builder.createBlock(result.addRegion());
    casePatterns.push_back(caseCtor(i)->attr());
  }

  result.addAttribute("casePatterns", builder.getArrayAttr(casePatterns));
}

// Strength reduce case styles based on the bit patterns.
LogicalResult CaseOp::canonicalize(CaseOp op, PatternRewriter &rewriter) {
  if (op.getCaseStyle() == CaseStmtType::CaseStmt)
    return failure();
  if (isa<hw::EnumType>(op.getCond().getType()))
    return failure();

  auto caseInfo = op.getCases();
  bool noXZ = llvm::all_of(caseInfo, [](const CaseInfo &ci) {
    return !ci.pattern.get()->hasX() && !ci.pattern.get()->hasZ();
  });
  bool noX = llvm::all_of(caseInfo, [](const CaseInfo &ci) {
    if (isa<CaseDefaultPattern>(ci.pattern))
      return true;
    return !ci.pattern.get()->hasX();
  });
  bool noZ = llvm::all_of(caseInfo, [](const CaseInfo &ci) {
    if (isa<CaseDefaultPattern>(ci.pattern))
      return true;
    return !ci.pattern.get()->hasZ();
  });

  if (op.getCaseStyle() == CaseStmtType::CaseXStmt) {
    if (noXZ) {
      rewriter.modifyOpInPlace(op, [&]() {
        op.setCaseStyleAttr(
            CaseStmtTypeAttr::get(op.getContext(), CaseStmtType::CaseStmt));
      });
      return success();
    }
    if (noX) {
      rewriter.modifyOpInPlace(op, [&]() {
        op.setCaseStyleAttr(
            CaseStmtTypeAttr::get(op.getContext(), CaseStmtType::CaseZStmt));
      });
      return success();
    }
  }

  if (op.getCaseStyle() == CaseStmtType::CaseZStmt && noZ) {
    rewriter.modifyOpInPlace(op, [&]() {
      op.setCaseStyleAttr(
          CaseStmtTypeAttr::get(op.getContext(), CaseStmtType::CaseStmt));
    });
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// OrderedOutputOp
//===----------------------------------------------------------------------===//

void OrderedOutputOp::build(OpBuilder &builder, OperationState &result,
                            std::function<void()> body) {
  OpBuilder::InsertionGuard guard(builder);

  builder.createBlock(result.addRegion());

  // Fill in the body of the ordered block.
  if (body)
    body();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(OpBuilder &builder, OperationState &result,
                  int64_t lowerBound, int64_t upperBound, int64_t step,
                  IntegerType type, StringRef name,
                  llvm::function_ref<void(BlockArgument)> body) {
  auto lb = hw::ConstantOp::create(builder, result.location, type, lowerBound);
  auto ub = hw::ConstantOp::create(builder, result.location, type, upperBound);
  auto st = hw::ConstantOp::create(builder, result.location, type, step);
  build(builder, result, lb, ub, st, name, body);
}
void ForOp::build(OpBuilder &builder, OperationState &result, Value lowerBound,
                  Value upperBound, Value step, StringRef name,
                  llvm::function_ref<void(BlockArgument)> body) {
  OpBuilder::InsertionGuard guard(builder);
  build(builder, result, lowerBound, upperBound, step, name);
  auto *region = result.regions.front().get();
  builder.createBlock(region);
  BlockArgument blockArgument =
      region->addArgument(lowerBound.getType(), result.location);

  if (body)
    body(blockArgument);
}

void ForOp::getAsmBlockArgumentNames(mlir::Region &region,
                                     mlir::OpAsmSetValueNameFn setNameFn) {
  auto *block = &region.front();
  setNameFn(block->getArgument(0), getInductionVarNameAttr());
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  Type type;

  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;
  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;

  // Parse the induction variable followed by '='.
  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      // Parse loop bounds.
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  regionArgs.push_back(inductionVariable);

  // Resolve input operands.
  regionArgs.front().type = type;
  if (parser.resolveOperand(lb, type, result.operands) ||
      parser.resolveOperand(ub, type, result.operands) ||
      parser.resolveOperand(step, type, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (!inductionVariable.ssaName.name.empty()) {
    if (!isdigit(inductionVariable.ssaName.name[1]))
      // Retrive from its SSA name.
      result.attributes.append(
          {builder.getStringAttr("inductionVarName"),
           builder.getStringAttr(inductionVariable.ssaName.name.drop_front())});
  }

  return success();
}

void ForOp::print(OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();
  p << " : " << getInductionVar().getType() << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(), {"inductionVarName"});
}

LogicalResult ForOp::canonicalize(ForOp op, PatternRewriter &rewriter) {
  APInt lb, ub, step;
  if (matchPattern(op.getLowerBound(), mlir::m_ConstantInt(&lb)) &&
      matchPattern(op.getUpperBound(), mlir::m_ConstantInt(&ub)) &&
      matchPattern(op.getStep(), mlir::m_ConstantInt(&step)) &&
      lb + step == ub) {
    // Unroll the loop if it's executed only once.
    rewriter.replaceAllUsesWith(op.getInductionVar(), op.getLowerBound());
    replaceOpWithRegion(rewriter, op, op.getBodyRegion());
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Assignment statements
//===----------------------------------------------------------------------===//

LogicalResult BPAssignOp::verify() {
  if (isa<sv::WireOp>(getDest().getDefiningOp()))
    return emitOpError(
        "Verilog disallows procedural assignment to a net type (did you intend "
        "to use a variable type, e.g., sv.reg?)");
  return success();
}

LogicalResult PAssignOp::verify() {
  if (isa<sv::WireOp>(getDest().getDefiningOp()))
    return emitOpError(
        "Verilog disallows procedural assignment to a net type (did you intend "
        "to use a variable type, e.g., sv.reg?)");
  return success();
}

namespace {
// This represents a slice of an array.
struct ArraySlice {
  Value array;
  Value start;
  size_t size; // Represent a range array[start, start + size).

  // Get a struct from the value. Return std::nullopt if the value doesn't
  // represent an array slice.
  static std::optional<ArraySlice> getArraySlice(Value v) {
    auto *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    return TypeSwitch<Operation *, std::optional<ArraySlice>>(op)
        .Case<hw::ArrayGetOp, ArrayIndexInOutOp>(
            [](auto arrayIndex) -> std::optional<ArraySlice> {
              hw::ConstantOp constant =
                  arrayIndex.getIndex()
                      .template getDefiningOp<hw::ConstantOp>();
              if (!constant)
                return std::nullopt;
              return ArraySlice{/*array=*/arrayIndex.getInput(),
                                /*start=*/constant,
                                /*end=*/1};
            })
        .Case<hw::ArraySliceOp>([](hw::ArraySliceOp slice)
                                    -> std::optional<ArraySlice> {
          auto constant = slice.getLowIndex().getDefiningOp<hw::ConstantOp>();
          if (!constant)
            return std::nullopt;
          return ArraySlice{
              /*array=*/slice.getInput(), /*start=*/constant,
              /*end=*/
              hw::type_cast<hw::ArrayType>(slice.getType()).getNumElements()};
        })
        .Case<sv::IndexedPartSelectInOutOp>(
            [](sv::IndexedPartSelectInOutOp index)
                -> std::optional<ArraySlice> {
              auto constant = index.getBase().getDefiningOp<hw::ConstantOp>();
              if (!constant || index.getDecrement())
                return std::nullopt;
              return ArraySlice{/*array=*/index.getInput(),
                                /*start=*/constant,
                                /*end=*/index.getWidth()};
            })
        .Default([](auto) { return std::nullopt; });
  }

  // Create a pair of ArraySlice from source and destination of assignments.
  static std::optional<std::pair<ArraySlice, ArraySlice>>
  getAssignedRange(Operation *op) {
    assert((isa<PAssignOp, BPAssignOp>(op) && "assignments are expected"));
    auto srcRange = ArraySlice::getArraySlice(op->getOperand(1));
    if (!srcRange)
      return std::nullopt;
    auto destRange = ArraySlice::getArraySlice(op->getOperand(0));
    if (!destRange)
      return std::nullopt;

    return std::make_pair(*destRange, *srcRange);
  }
};
} // namespace

// This canonicalization merges neiboring assignments of array elements into
// array slice assignments. e.g.
// a[0] <= b[1]
// a[1] <= b[2]
// ->
// a[1:0] <= b[2:1]
template <typename AssignTy>
static LogicalResult mergeNeiboringAssignments(AssignTy op,
                                               PatternRewriter &rewriter) {
  // Get assigned ranges of each assignment.
  auto assignedRangeOpt = ArraySlice::getAssignedRange(op);
  if (!assignedRangeOpt)
    return failure();

  auto [dest, src] = *assignedRangeOpt;
  AssignTy nextAssign = dyn_cast_or_null<AssignTy>(op->getNextNode());
  bool changed = false;
  SmallVector<Location> loc{op.getLoc()};
  // Check that a next operation is a same kind of the assignment.
  while (nextAssign) {
    auto nextAssignedRange = ArraySlice::getAssignedRange(nextAssign);
    if (!nextAssignedRange)
      break;
    auto [nextDest, nextSrc] = *nextAssignedRange;
    // Check that these assignments are mergaable.
    if (dest.array != nextDest.array || src.array != nextSrc.array ||
        !hw::isOffset(dest.start, nextDest.start, dest.size) ||
        !hw::isOffset(src.start, nextSrc.start, src.size))
      break;

    dest.size += nextDest.size;
    src.size += nextSrc.size;
    changed = true;
    loc.push_back(nextAssign.getLoc());
    rewriter.eraseOp(nextAssign);
    nextAssign = dyn_cast_or_null<AssignTy>(op->getNextNode());
  }

  if (!changed)
    return failure();

  // From here, construct assignments of array slices.
  auto resultType = hw::ArrayType::get(
      hw::type_cast<hw::ArrayType>(src.array.getType()).getElementType(),
      src.size);
  auto newDest = sv::IndexedPartSelectInOutOp::create(
      rewriter, op.getLoc(), dest.array, dest.start, dest.size);
  auto newSrc = hw::ArraySliceOp::create(rewriter, op.getLoc(), resultType,
                                         src.array, src.start);
  auto newLoc = rewriter.getFusedLoc(loc);
  auto newOp = rewriter.replaceOpWithNewOp<AssignTy>(op, newDest, newSrc);
  newOp->setLoc(newLoc);
  return success();
}

LogicalResult PAssignOp::canonicalize(PAssignOp op, PatternRewriter &rewriter) {
  return mergeNeiboringAssignments(op, rewriter);
}

LogicalResult BPAssignOp::canonicalize(BPAssignOp op,
                                       PatternRewriter &rewriter) {
  return mergeNeiboringAssignments(op, rewriter);
}

//===----------------------------------------------------------------------===//
// TypeDecl operations
//===----------------------------------------------------------------------===//

void InterfaceOp::build(OpBuilder &builder, OperationState &result,
                        StringRef sym_name, std::function<void()> body) {
  OpBuilder::InsertionGuard guard(builder);

  result.addAttribute(::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(sym_name));
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
      SymbolRefAttr::get(ctxt, getSymName(),
                         {SymbolRefAttr::get(ctxt, modportName)}));
}

Type InterfaceOp::getSignalType(StringRef signalName) {
  InterfaceSignalOp signal = lookupSymbol<InterfaceSignalOp>(signalName);
  assert(signal && "Interface signal symbol not found.");
  return signal.getType();
}

static ParseResult parseModportStructs(OpAsmParser &parser,
                                       ArrayAttr &portsAttr) {

  auto *context = parser.getBuilder().getContext();

  SmallVector<Attribute, 8> ports;
  auto parseElement = [&]() -> ParseResult {
    auto direction = ModportDirectionAttr::parse(parser, {});
    if (!direction)
      return failure();

    FlatSymbolRefAttr signal;
    if (parser.parseAttribute(signal))
      return failure();

    ports.push_back(ModportStructAttr::get(
        context, cast<ModportDirectionAttr>(direction), signal));
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseElement))
    return failure();

  portsAttr = ArrayAttr::get(context, ports);
  return success();
}

static void printModportStructs(OpAsmPrinter &p, Operation *,
                                ArrayAttr portsAttr) {
  p << "(";
  llvm::interleaveComma(portsAttr, p, [&](Attribute attr) {
    auto port = cast<ModportStructAttr>(attr);
    p << stringifyEnum(port.getDirection().getValue());
    p << ' ';
    p.printSymbolName(port.getSignal().getRootReference().getValue());
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
  auto inputDir = ModportDirectionAttr::get(ctxt, ModportDirection::input);
  auto outputDir = ModportDirectionAttr::get(ctxt, ModportDirection::output);
  for (auto input : inputs)
    directions.push_back(ModportStructAttr::get(
        ctxt, inputDir, SymbolRefAttr::get(ctxt, input)));
  for (auto output : outputs)
    directions.push_back(ModportStructAttr::get(
        ctxt, outputDir, SymbolRefAttr::get(ctxt, output)));
  build(builder, state, name, ArrayAttr::get(ctxt, directions));
}

std::optional<size_t> InterfaceInstanceOp::getTargetResultIndex() {
  // Inner symbols on instance operations target the op not any result.
  return std::nullopt;
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InterfaceInstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult InterfaceInstanceOp::verify() {
  if (getName().empty())
    return emitOpError("requires non-empty name");
  return success();
}

LogicalResult
InterfaceInstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *symtable = SymbolTable::getNearestSymbolTable(*this);
  if (!symtable)
    return emitError("sv.interface.instance must exist within a region "
                     "which has a symbol table.");
  auto ifaceTy = getType();
  auto *referencedOp =
      symbolTable.lookupSymbolIn(symtable, ifaceTy.getInterface());
  if (!referencedOp)
    return emitError("Symbol not found: ") << ifaceTy.getInterface() << ".";
  if (!isa<InterfaceOp>(referencedOp))
    return emitError("Symbol ")
           << ifaceTy.getInterface() << " is not an InterfaceOp.";
  return success();
}

/// Ensure that the symbol being instantiated exists and is an
/// InterfaceModportOp.
LogicalResult
GetModportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *symtable = SymbolTable::getNearestSymbolTable(*this);
  if (!symtable)
    return emitError("sv.interface.instance must exist within a region "
                     "which has a symbol table.");

  auto ifaceTy = getType();
  auto *referencedOp =
      symbolTable.lookupSymbolIn(symtable, ifaceTy.getModport());
  if (!referencedOp)
    return emitError("Symbol not found: ") << ifaceTy.getModport() << ".";
  if (!isa<InterfaceModportOp>(referencedOp))
    return emitError("Symbol ")
           << ifaceTy.getModport() << " is not an InterfaceModportOp.";
  return success();
}

void GetModportOp::build(OpBuilder &builder, OperationState &state, Value value,
                         StringRef field) {
  auto ifaceTy = dyn_cast<InterfaceType>(value.getType());
  assert(ifaceTy && "GetModportOp expects an InterfaceType.");
  auto fieldAttr = SymbolRefAttr::get(builder.getContext(), field);
  auto modportSym =
      SymbolRefAttr::get(ifaceTy.getInterface().getRootReference(), fieldAttr);
  build(builder, state, ModportType::get(builder.getContext(), modportSym),
        value, fieldAttr);
}

/// Lookup the op for the modport declaration.  This returns null on invalid
/// IR.
InterfaceModportOp
GetModportOp::getReferencedDecl(const hw::HWSymbolCache &cache) {
  return dyn_cast_or_null<InterfaceModportOp>(
      cache.getDefinition(getFieldAttr()));
}

void ReadInterfaceSignalOp::build(OpBuilder &builder, OperationState &state,
                                  Value iface, StringRef signalName) {
  auto ifaceTy = dyn_cast<InterfaceType>(iface.getType());
  assert(ifaceTy && "ReadInterfaceSignalOp expects an InterfaceType.");
  auto fieldAttr = SymbolRefAttr::get(builder.getContext(), signalName);
  InterfaceOp ifaceDefOp = SymbolTable::lookupNearestSymbolFrom<InterfaceOp>(
      iface.getDefiningOp(), ifaceTy.getInterface());
  assert(ifaceDefOp &&
         "ReadInterfaceSignalOp could not resolve an InterfaceOp.");
  build(builder, state, ifaceDefOp.getSignalType(signalName), iface, fieldAttr);
}

/// Lookup the op for the signal declaration.  This returns null on invalid
/// IR.
InterfaceSignalOp
ReadInterfaceSignalOp::getReferencedDecl(const hw::HWSymbolCache &cache) {
  return dyn_cast_or_null<InterfaceSignalOp>(
      cache.getDefinition(getSignalNameAttr()));
}

ParseResult parseIfaceTypeAndSignal(OpAsmParser &p, Type &ifaceTy,
                                    FlatSymbolRefAttr &signalName) {
  SymbolRefAttr fullSym;
  if (p.parseAttribute(fullSym) || fullSym.getNestedReferences().size() != 1)
    return failure();

  auto *ctxt = p.getBuilder().getContext();
  ifaceTy = InterfaceType::get(
      ctxt, FlatSymbolRefAttr::get(fullSym.getRootReference()));
  signalName = FlatSymbolRefAttr::get(fullSym.getLeafReference());
  return success();
}

void printIfaceTypeAndSignal(OpAsmPrinter &p, Operation *op, Type type,
                             FlatSymbolRefAttr signalName) {
  InterfaceType ifaceTy = dyn_cast<InterfaceType>(type);
  assert(ifaceTy && "Expected an InterfaceType");
  auto sym = SymbolRefAttr::get(ifaceTy.getInterface().getRootReference(),
                                {signalName});
  p << sym;
}

LogicalResult verifySignalExists(Value ifaceVal, FlatSymbolRefAttr signalName) {
  auto ifaceTy = dyn_cast<InterfaceType>(ifaceVal.getType());
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

Operation *
InterfaceInstanceOp::getReferencedInterface(const hw::HWSymbolCache *cache) {
  FlatSymbolRefAttr interface = getInterfaceType().getInterface();
  if (cache)
    if (auto *result = cache->getDefinition(interface))
      return result;

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;

  return topLevelModuleOp.lookupSymbol(interface);
}

LogicalResult AssignInterfaceSignalOp::verify() {
  return verifySignalExists(getIface(), getSignalNameAttr());
}

LogicalResult ReadInterfaceSignalOp::verify() {
  return verifySignalExists(getIface(), getSignalNameAttr());
}

//===----------------------------------------------------------------------===//
// WireOp
//===----------------------------------------------------------------------===//

void WireOp::build(OpBuilder &builder, OperationState &odsState,
                   Type elementType, StringAttr name,
                   hw::InnerSymAttr innerSym) {
  if (!name)
    name = builder.getStringAttr("");
  if (innerSym)
    odsState.addAttribute(hw::InnerSymbolTable::getInnerSymbolAttrName(),
                          innerSym);

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

std::optional<size_t> WireOp::getTargetResultIndex() { return 0; }

// If this wire is only written to, delete the wire and all writers.
LogicalResult WireOp::canonicalize(WireOp wire, PatternRewriter &rewriter) {
  // Block if op has SV attributes.
  if (hasSVAttributes(wire))
    return failure();

  // If the wire has a symbol, then we can't delete it.
  if (wire.getInnerSymAttr())
    return failure();

  // Wires have inout type, so they'll have assigns and read_inout operations
  // that work on them.  If anything unexpected is found then leave it alone.
  SmallVector<sv::ReadInOutOp> reads;
  sv::AssignOp write;

  for (auto *user : wire->getUsers()) {
    if (auto read = dyn_cast<sv::ReadInOutOp>(user)) {
      reads.push_back(read);
      continue;
    }

    // Otherwise must be an assign, and we must not have seen a write yet.
    auto assign = dyn_cast<sv::AssignOp>(user);
    // Either the wire has more than one write or another kind of Op (other than
    // AssignOp and ReadInOutOp), then can't optimize.
    if (!assign || write)
      return failure();

    // If the assign op has SV attributes, we don't want to delete the
    // assignment.
    if (hasSVAttributes(assign))
      return failure();

    write = assign;
  }

  Value connected;
  if (!write) {
    // If no write and only reads, then replace with ZOp.
    // SV 6.6: "If no driver is connected to a net, its
    // value shall be high-impedance (z) unless the net is a trireg"
    connected = ConstantZOp::create(
        rewriter, wire.getLoc(),
        cast<InOutType>(wire.getResult().getType()).getElementType());
  } else if (isa<hw::HWModuleOp>(write->getParentOp()))
    connected = write.getSrc();
  else
    // If the write is happening at the module level then we don't have any
    // use-before-def checking to do, so we only handle that for now.
    return failure();

  // If the wire has a name attribute, propagate the name to the expression.
  if (auto *connectedOp = connected.getDefiningOp())
    if (!wire.getName().empty())
      rewriter.modifyOpInPlace(connectedOp, [&] {
        connectedOp->setAttr("sv.namehint", wire.getNameAttr());
      });

  // Ok, we can do this.  Replace all the reads with the connected value.
  for (auto read : reads)
    rewriter.replaceOp(read, connected);

  // And remove the write and wire itself.
  if (write)
    rewriter.eraseOp(write);
  rewriter.eraseOp(wire);
  return success();
}

//===----------------------------------------------------------------------===//
// IndexedPartSelectInOutOp
//===----------------------------------------------------------------------===//

// A helper function to infer a return type of IndexedPartSelectInOutOp.
static Type getElementTypeOfWidth(Type type, int32_t width) {
  auto elemTy = cast<hw::InOutType>(type).getElementType();
  if (isa<IntegerType>(elemTy))
    return hw::InOutType::get(IntegerType::get(type.getContext(), width));
  if (isa<hw::ArrayType>(elemTy))
    return hw::InOutType::get(hw::ArrayType::get(
        cast<hw::ArrayType>(elemTy).getElementType(), width));
  return {};
}

LogicalResult IndexedPartSelectInOutOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Adaptor adaptor(operands, attrs, properties, regions);
  auto width = adaptor.getWidthAttr();
  if (!width)
    return failure();

  auto typ = getElementTypeOfWidth(operands[0].getType(),
                                   width.getValue().getZExtValue());
  if (!typ)
    return failure();
  results.push_back(typ);
  return success();
}

LogicalResult IndexedPartSelectInOutOp::verify() {
  unsigned inputWidth = 0, resultWidth = 0;
  auto opWidth = getWidth();
  auto inputElemTy = cast<InOutType>(getInput().getType()).getElementType();
  auto resultElemTy = cast<InOutType>(getType()).getElementType();
  if (auto i = dyn_cast<IntegerType>(inputElemTy))
    inputWidth = i.getWidth();
  else if (auto i = hw::type_cast<hw::ArrayType>(inputElemTy))
    inputWidth = i.getNumElements();
  else
    return emitError("input element type must be Integer or Array");

  if (auto resType = dyn_cast<IntegerType>(resultElemTy))
    resultWidth = resType.getWidth();
  else if (auto resType = hw::type_cast<hw::ArrayType>(resultElemTy))
    resultWidth = resType.getNumElements();
  else
    return emitError("result element type must be Integer or Array");

  if (opWidth > inputWidth)
    return emitError("slice width should not be greater than input width");
  if (opWidth != resultWidth)
    return emitError("result width must be equal to slice width");
  return success();
}

OpFoldResult IndexedPartSelectInOutOp::fold(FoldAdaptor) {
  if (getType() == getInput().getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// IndexedPartSelectOp
//===----------------------------------------------------------------------===//

LogicalResult IndexedPartSelectOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Adaptor adaptor(operands, attrs, properties, regions);
  auto width = adaptor.getWidthAttr();
  if (!width)
    return failure();

  results.push_back(IntegerType::get(context, width.getInt()));
  return success();
}

LogicalResult IndexedPartSelectOp::verify() {
  auto opWidth = getWidth();

  unsigned resultWidth = cast<IntegerType>(getType()).getWidth();
  unsigned inputWidth = cast<IntegerType>(getInput().getType()).getWidth();

  if (opWidth > inputWidth)
    return emitError("slice width should not be greater than input width");
  if (opWidth != resultWidth)
    return emitError("result width must be equal to slice width");
  return success();
}

//===----------------------------------------------------------------------===//
// StructFieldInOutOp
//===----------------------------------------------------------------------===//

LogicalResult StructFieldInOutOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Adaptor adaptor(operands, attrs, properties, regions);
  auto field = adaptor.getFieldAttr();
  if (!field)
    return failure();
  auto structType =
      hw::type_cast<hw::StructType>(getInOutElementType(operands[0].getType()));
  auto resultType = structType.getFieldType(field);
  if (!resultType)
    return failure();

  results.push_back(hw::InOutType::get(resultType));
  return success();
}

//===----------------------------------------------------------------------===//
// Other ops.
//===----------------------------------------------------------------------===//

LogicalResult AliasOp::verify() {
  // Must have at least two operands.
  if (getAliases().size() < 2)
    return emitOpError("alias must have at least two operands");

  return success();
}

//===----------------------------------------------------------------------===//
// BindOp
//===----------------------------------------------------------------------===//

/// Instances must be at the top level of the hw.module (or within a `ifdef)
// and are typically at the end of it, so we scan backwards to find them.
template <class Op>
static Op findInstanceSymbolInBlock(StringAttr name, Block *body) {
  for (auto &op : llvm::reverse(body->getOperations())) {
    if (auto instance = dyn_cast<Op>(op)) {
      if (auto innerSym = instance.getInnerSym())
        if (innerSym->getSymName() == name)
          return instance;
    }

    if (auto ifdef = dyn_cast<IfDefOp>(op)) {
      if (auto result =
              findInstanceSymbolInBlock<Op>(name, ifdef.getThenBlock()))
        return result;
      if (ifdef.hasElse())
        if (auto result =
                findInstanceSymbolInBlock<Op>(name, ifdef.getElseBlock()))
          return result;
    }
  }
  return {};
}

hw::InstanceOp BindOp::getReferencedInstance(const hw::HWSymbolCache *cache) {
  // If we have a cache, directly look up the referenced instance.
  if (cache) {
    auto result = cache->getInnerDefinition(getInstance());
    return cast<hw::InstanceOp>(result.getOp());
  }

  // Otherwise, resolve the instance by looking up the module ...
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return {};

  auto hwModule = dyn_cast_or_null<hw::HWModuleOp>(
      topLevelModuleOp.lookupSymbol(getInstance().getModule()));
  if (!hwModule)
    return {};

  // ... then look up the instance within it.
  return findInstanceSymbolInBlock<hw::InstanceOp>(getInstance().getName(),
                                                   hwModule.getBodyBlock());
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult BindOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = (*this)->getParentOfType<mlir::ModuleOp>();
  auto hwModule = dyn_cast_or_null<hw::HWModuleOp>(
      symbolTable.lookupSymbolIn(module, getInstance().getModule()));
  if (!hwModule)
    return emitError("Referenced module doesn't exist ")
           << getInstance().getModule() << "::" << getInstance().getName();

  auto inst = findInstanceSymbolInBlock<hw::InstanceOp>(
      getInstance().getName(), hwModule.getBodyBlock());
  if (!inst)
    return emitError("Referenced instance doesn't exist ")
           << getInstance().getModule() << "::" << getInstance().getName();
  if (!inst.getDoNotPrint())
    return emitError("Referenced instance isn't marked as doNotPrint");
  return success();
}

void BindOp::build(OpBuilder &builder, OperationState &odsState, StringAttr mod,
                   StringAttr name) {
  auto ref = hw::InnerRefAttr::get(mod, name);
  odsState.addAttribute("instance", ref);
}

//===----------------------------------------------------------------------===//
// SVVerbatimModuleOp
//===----------------------------------------------------------------------===//

SmallVector<hw::PortInfo> SVVerbatimModuleOp::getPortList() {
  SmallVector<hw::PortInfo> retval;
  auto modTy = getModuleType();
  auto emptyDict = DictionaryAttr::get(getContext());
  auto unknownLoc = UnknownLoc::get(getContext());
  auto locs = getAllPortLocs();
  auto attrs = getAllPortAttrs();

  for (unsigned i = 0, e = modTy.getNumPorts(); i < e; ++i) {
    Location loc = (i < locs.size()) ? locs[i] : unknownLoc;
    DictionaryAttr portAttrs =
        (i < attrs.size()) ? cast<DictionaryAttr>(attrs[i]) : emptyDict;

    retval.push_back({modTy.getPorts()[i],
                      modTy.isOutput(i) ? modTy.getOutputIdForPortId(i)
                                        : modTy.getInputIdForPortId(i),
                      portAttrs, loc});
  }
  return retval;
}

hw::ModuleType SVVerbatimModuleOp::getHWModuleType() {
  return llvm::cast<hw::ModuleType>(getModuleTypeAttr().getValue());
}

size_t SVVerbatimModuleOp::getNumPorts() {
  return getModuleType().getNumPorts();
}

size_t SVVerbatimModuleOp::getNumInputPorts() {
  return getModuleType().getNumInputs();
}

size_t SVVerbatimModuleOp::getNumOutputPorts() {
  return getModuleType().getNumOutputs();
}

hw::PortInfo SVVerbatimModuleOp::getPort(size_t idx) {
  return getPortList()[idx];
}

size_t SVVerbatimModuleOp::getPortIdForInputId(size_t idx) {
  return getModuleType().getPortIdForInputId(idx);
}

size_t SVVerbatimModuleOp::getPortIdForOutputId(size_t idx) {
  return getModuleType().getPortIdForOutputId(idx);
}

ArrayRef<Attribute> SVVerbatimModuleOp::getAllPortAttrs() {
  if (auto attrs = getPerPortAttrs())
    return attrs->getValue();

  // Return empty ArrayRef - callers should handle the empty case
  return {};
}

void SVVerbatimModuleOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  assert(attrs.empty() || attrs.size() == getNumPorts());
  if (attrs.empty()) {
    removePerPortAttrsAttr();
  } else {
    setPerPortAttrsAttr(ArrayAttr::get(getContext(), attrs));
  }
}

void SVVerbatimModuleOp::removeAllPortAttrs() { removePerPortAttrsAttr(); }

SmallVector<Location> SVVerbatimModuleOp::getAllPortLocs() {
  SmallVector<Location> locs;
  if (auto portLocs = getPortLocs()) {
    locs.reserve(portLocs->size());
    for (auto loc : *portLocs)
      locs.push_back(cast<Location>(loc));
  }
  return locs;
}

void SVVerbatimModuleOp::setAllPortLocsAttrs(ArrayRef<Attribute> locs) {
  assert(locs.empty() || locs.size() == getNumPorts());
  if (locs.empty()) {
    removePortLocsAttr();
  } else {
    setPortLocsAttr(ArrayAttr::get(getContext(), locs));
  }
}

void SVVerbatimModuleOp::setHWModuleType(hw::ModuleType type) {
  setModuleTypeAttr(TypeAttr::get(type));
  removePerPortAttrsAttr();
  removePortLocsAttr();
}

Attribute SVVerbatimModuleOp::getPortAttrs(size_t idx) {
  assert(idx < getNumPorts());
  auto allAttrs = getAllPortAttrs();
  if (idx < allAttrs.size())
    return allAttrs[idx];
  return DictionaryAttr::get(getContext());
}

void SVVerbatimModuleOp::setPortAttrs(size_t idx, Attribute attrs) {
  assert(idx < getNumPorts());
  auto allAttrs = getAllPortAttrs();
  SmallVector<Attribute> newAttrs(allAttrs.begin(), allAttrs.end());
  if (newAttrs.size() <= idx)
    newAttrs.resize(getNumPorts(), DictionaryAttr::get(getContext()));
  newAttrs[idx] = attrs;
  setAllPortAttrs(newAttrs);
}

Location SVVerbatimModuleOp::getPortLoc(size_t idx) {
  assert(idx < getNumPorts());
  auto allLocs = getAllPortLocs();
  if (idx < allLocs.size())
    return allLocs[idx];
  return UnknownLoc::get(getContext());
}

void SVVerbatimModuleOp::setPortLoc(size_t idx, Location loc) {
  assert(idx < getNumPorts());
  auto allLocs = getAllPortLocs();
  SmallVector<Attribute> newLocs;
  newLocs.reserve(getNumPorts());
  for (size_t i = 0; i < getNumPorts(); ++i) {
    if (i == idx)
      newLocs.push_back(loc);
    else if (i < allLocs.size())
      newLocs.push_back(allLocs[i]);
    else
      newLocs.push_back(UnknownLoc::get(getContext()));
  }
  setAllPortLocsAttrs(newLocs);
}

void SVVerbatimModuleOp::print(OpAsmPrinter &p) {
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = (*this)->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';

  p.printSymbolName(SymbolTable::getSymbolName(*this).getValue());

  printOptionalParameterList(p, *this, getParameters());

  Region emptyRegion;
  hw::module_like_impl::printModuleSignatureNew(
      p, emptyRegion, getModuleType(), getAllPortAttrs(), getAllPortLocs());

  SmallVector<StringRef> omittedAttrs = {
      SymbolTable::getSymbolAttrName(),   SymbolTable::getVisibilityAttrName(),
      getModuleTypeAttrName().getValue(), getPerPortAttrsAttrName().getValue(),
      getPortLocsAttrName().getValue(),   getParametersAttrName().getValue()};

  mlir::function_interface_impl::printFunctionAttributes(p, *this,
                                                         omittedAttrs);
}

ParseResult SVVerbatimModuleOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  using namespace mlir::function_interface_impl;
  auto builder = parser.getBuilder();

  // parse optional visibility
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // parse optional parameters
  ArrayAttr parameters;
  if (parseOptionalParameterList(parser, parameters))
    return failure();

  // parse module-like op signature
  SmallVector<hw::module_like_impl::PortParse> ports;
  TypeAttr modType;
  if (failed(
          hw::module_like_impl::parseModuleSignature(parser, ports, modType)))
    return failure();

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  result.addAttribute("module_type", modType);
  result.addAttribute("parameters", parameters);

  SmallVector<Attribute> portAttrs, portLocs;
  for (auto &port : ports) {
    if (port.attrs && !cast<DictionaryAttr>(port.attrs).empty())
      portAttrs.push_back(port.attrs);
    else
      portAttrs.push_back(builder.getDictionaryAttr({}));

    if (port.sourceLoc && !isa<UnknownLoc>(*port.sourceLoc))
      portLocs.push_back(*port.sourceLoc);
    else
      portLocs.push_back(builder.getUnknownLoc());
  }

  if (!portAttrs.empty())
    result.addAttribute("per_port_attrs", builder.getArrayAttr(portAttrs));
  if (!portLocs.empty())
    result.addAttribute("port_locs", builder.getArrayAttr(portLocs));

  // parse verbatim content
  if (!result.attributes.get("content"))
    return parser.emitError(parser.getCurrentLocation(),
                            "sv.verbatim.module requires 'content' attribute");

  // parse output file
  auto outputFileAttr = result.attributes.get("output_file");
  if (!outputFileAttr)
    return parser.emitError(
        parser.getCurrentLocation(),
        "sv.verbatim.module requires 'output_file' attribute");

  if (!isa<hw::OutputFileAttr>(outputFileAttr))
    return parser.emitError(
        parser.getCurrentLocation(),
        "sv.verbatim.module 'output_file' attribute must be an OutputFileAttr");

  return success();
}

void SVVerbatimModuleOp::setAllPortNames(ArrayRef<Attribute> names) {
  auto moduleType = getModuleType();
  assert(names.size() == moduleType.getNumPorts() &&
         "Number of names must match number of ports");

  SmallVector<hw::ModulePort> newPorts;
  for (auto [i, port] : llvm::enumerate(moduleType.getPorts())) {
    auto newName = cast<StringAttr>(names[i]);
    newPorts.push_back({newName, port.type, port.dir});
  }

  auto newModuleType = hw::ModuleType::get(getContext(), newPorts);
  setModuleTypeAttr(TypeAttr::get(newModuleType));
}

LogicalResult SVVerbatimModuleOp::verify() {
  // must have verbatim content
  if (getContent().empty())
    return emitOpError("missing or empty content attribute");

  auto moduleType = getModuleType();
  auto numPorts = moduleType.getNumPorts();

  if (auto attrs = getPerPortAttrs()) {
    if (attrs->size() != numPorts)
      return emitOpError("port attributes array size (")
             << attrs->size() << ") doesn't match number of ports (" << numPorts
             << ")";
  }

  if (auto locs = getPortLocs()) {
    if (locs->size() != numPorts)
      return emitOpError("port locations array size (")
             << locs->size() << ") doesn't match number of ports (" << numPorts
             << ")";
  }

  auto outputFileAttr = getOutputFile();
  if (!outputFileAttr || outputFileAttr.getFilename().getValue().empty())
    return emitOpError("output_file attribute cannot be empty");

  return success();
}

LogicalResult
SVVerbatimModuleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (auto additionalFiles = getAdditionalFiles()) {
    for (auto fileRef : *additionalFiles) {
      auto symbolRef = llvm::cast<SymbolRefAttr>(fileRef);
      auto *referencedOp =
          symbolTable.lookupNearestSymbolFrom(getOperation(), symbolRef);
      if (!isa<circt::emit::FileOp>(referencedOp))
        return emitError("Symbol ")
               << symbolRef << " is not an emit.file operation.";
    }
  }

  return success();
}

StringAttr SVVerbatimModuleOp::getVerilogModuleNameAttr() {
  if (auto vName = getVerilogNameAttr()) {
    return vName;
  }
  return (*this)->getAttrOfType<StringAttr>(
      ::mlir::SymbolTable::getSymbolAttrName());
}

hw::ModuleType
SVVerbatimModuleOp::getModuleTypeFromPorts(mlir::MLIRContext *context,
                                           ArrayRef<hw::PortInfo> ports) {
  SmallVector<hw::ModulePort> modulePorts;
  for (const auto &port : ports) {
    Type portType = port.type;
    hw::ModulePort::Direction portDir = port.dir;
    if (auto inoutType = dyn_cast<hw::InOutType>(port.type)) {
      portType = inoutType.getElementType();
      portDir = hw::ModulePort::Direction::InOut;
    }
    modulePorts.push_back({port.name, portType, portDir});
  }
  return hw::ModuleType::get(context, modulePorts);
}

//===----------------------------------------------------------------------===//
// BindInterfaceOp
//===----------------------------------------------------------------------===//

sv::InterfaceInstanceOp
BindInterfaceOp::getReferencedInstance(const hw::HWSymbolCache *cache) {
  // If we have a cache, directly look up the referenced instance.
  if (cache) {
    auto result = cache->getInnerDefinition(getInstance());
    return cast<sv::InterfaceInstanceOp>(result.getOp());
  }

  // Otherwise, resolve the instance by looking up the module ...
  auto *symbolTable = SymbolTable::getNearestSymbolTable(*this);
  if (!symbolTable)
    return {};
  auto *parentOp =
      lookupSymbolInNested(symbolTable, getInstance().getModule().getValue());
  if (!parentOp)
    return {};

  // ... then look up the instance within it.
  return findInstanceSymbolInBlock<sv::InterfaceInstanceOp>(
      getInstance().getName(), &parentOp->getRegion(0).front());
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
LogicalResult
BindInterfaceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *parentOp =
      symbolTable.lookupNearestSymbolFrom(*this, getInstance().getModule());
  if (!parentOp)
    return emitError("Referenced module doesn't exist ")
           << getInstance().getModule() << "::" << getInstance().getName();

  auto inst = findInstanceSymbolInBlock<sv::InterfaceInstanceOp>(
      getInstance().getName(), &parentOp->getRegion(0).front());
  if (!inst)
    return emitError("Referenced interface doesn't exist ")
           << getInstance().getModule() << "::" << getInstance().getName();
  if (!inst.getDoNotPrint())
    return emitError("Referenced interface isn't marked as doNotPrint");
  return success();
}

//===----------------------------------------------------------------------===//
// XMROp
//===----------------------------------------------------------------------===//

ParseResult parseXMRPath(::mlir::OpAsmParser &parser, ArrayAttr &pathAttr,
                         StringAttr &terminalAttr) {
  SmallVector<Attribute> strings;
  ParseResult ret = parser.parseCommaSeparatedList([&]() {
    StringAttr result;
    StringRef keyword;
    if (succeeded(parser.parseOptionalKeyword(&keyword))) {
      strings.push_back(parser.getBuilder().getStringAttr(keyword));
      return success();
    }
    if (succeeded(parser.parseAttribute(
            result, parser.getBuilder().getType<NoneType>()))) {
      strings.push_back(result);
      return success();
    }
    return failure();
  });
  if (succeeded(ret)) {
    pathAttr = parser.getBuilder().getArrayAttr(
        ArrayRef<Attribute>(strings).drop_back());
    terminalAttr = cast<StringAttr>(*strings.rbegin());
  }
  return ret;
}

void printXMRPath(OpAsmPrinter &p, XMROp op, ArrayAttr pathAttr,
                  StringAttr terminalAttr) {
  llvm::interleaveComma(pathAttr, p);
  p << ", " << terminalAttr;
}

/// Ensure that the symbol being instantiated exists and is a HierPathOp.
LogicalResult XMRRefOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *table = SymbolTable::getNearestSymbolTable(*this);
  auto path = dyn_cast_or_null<hw::HierPathOp>(
      symbolTable.lookupSymbolIn(table, getRefAttr()));
  if (!path)
    return emitError("Referenced path doesn't exist ") << getRefAttr();

  return success();
}

hw::HierPathOp XMRRefOp::getReferencedPath(const hw::HWSymbolCache *cache) {
  if (cache)
    if (auto *result = cache->getDefinition(getRefAttr().getAttr()))
      return cast<hw::HierPathOp>(result);

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol<hw::HierPathOp>(getRefAttr().getValue());
}

//===----------------------------------------------------------------------===//
// Verification Ops.
//===----------------------------------------------------------------------===//

static LogicalResult eraseIfZeroOrNotZero(Operation *op, Value value,
                                          PatternRewriter &rewriter,
                                          bool eraseIfZero) {
  if (auto constant = value.getDefiningOp<hw::ConstantOp>())
    if (constant.getValue().isZero() == eraseIfZero) {
      rewriter.eraseOp(op);
      return success();
    }

  return failure();
}

template <class Op, bool EraseIfZero = false>
static LogicalResult canonicalizeImmediateVerifOp(Op op,
                                                  PatternRewriter &rewriter) {
  return eraseIfZeroOrNotZero(op, op.getExpression(), rewriter, EraseIfZero);
}

void AssertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<AssertOp>);
}

void AssumeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<AssumeOp>);
}

void CoverOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<CoverOp, /* EraseIfZero = */ true>);
}

template <class Op, bool EraseIfZero = false>
static LogicalResult canonicalizeConcurrentVerifOp(Op op,
                                                   PatternRewriter &rewriter) {
  return eraseIfZeroOrNotZero(op, op.getProperty(), rewriter, EraseIfZero);
}

void AssertConcurrentOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add(canonicalizeConcurrentVerifOp<AssertConcurrentOp>);
}

void AssumeConcurrentOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add(canonicalizeConcurrentVerifOp<AssumeConcurrentOp>);
}

void CoverConcurrentOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add(
      canonicalizeConcurrentVerifOp<CoverConcurrentOp, /* EraseIfZero */ true>);
}

//===----------------------------------------------------------------------===//
// SV generate ops
//===----------------------------------------------------------------------===//

/// Parse cases formatted like:
///  case (pattern, "name") { ... }
bool parseCaseRegions(OpAsmParser &p, ArrayAttr &patternsArray,
                      ArrayAttr &caseNamesArray,
                      SmallVectorImpl<std::unique_ptr<Region>> &caseRegions) {
  SmallVector<Attribute> patterns;
  SmallVector<Attribute> names;
  while (!p.parseOptionalKeyword("case")) {
    Attribute pattern;
    StringAttr name;
    std::unique_ptr<Region> region = std::make_unique<Region>();
    if (p.parseLParen() || p.parseAttribute(pattern) || p.parseComma() ||
        p.parseAttribute(name) || p.parseRParen() || p.parseRegion(*region))
      return true;
    patterns.push_back(pattern);
    names.push_back(name);
    if (region->empty())
      region->push_back(new Block());
    caseRegions.push_back(std::move(region));
  }
  patternsArray = p.getBuilder().getArrayAttr(patterns);
  caseNamesArray = p.getBuilder().getArrayAttr(names);
  return false;
}

/// Print cases formatted like:
///  case (pattern, "name") { ... }
void printCaseRegions(OpAsmPrinter &p, Operation *, ArrayAttr patternsArray,
                      ArrayAttr namesArray,
                      MutableArrayRef<Region> caseRegions) {
  assert(patternsArray.size() == caseRegions.size());
  assert(patternsArray.size() == namesArray.size());
  for (size_t i = 0, e = caseRegions.size(); i < e; ++i) {
    p.printNewline();
    p << "case (" << patternsArray[i] << ", " << namesArray[i] << ") ";
    p.printRegion(caseRegions[i]);
  }
  p.printNewline();
}

LogicalResult GenerateCaseOp::verify() {
  size_t numPatterns = getCasePatterns().size();
  if (getCaseRegions().size() != numPatterns ||
      getCaseNames().size() != numPatterns)
    return emitOpError(
        "Size of caseRegions, patterns, and caseNames must match");

  StringSet<> usedNames;
  for (Attribute name : getCaseNames()) {
    StringAttr nameStr = dyn_cast<StringAttr>(name);
    if (!nameStr)
      return emitOpError("caseNames must all be string attributes");
    if (usedNames.contains(nameStr.getValue()))
      return emitOpError("caseNames must be unique");
    usedNames.insert(nameStr.getValue());
  }

  // mlir::FailureOr<Type> condType = evaluateParametricType();

  return success();
}

ModportStructAttr ModportStructAttr::get(MLIRContext *context,
                                         ModportDirection direction,
                                         FlatSymbolRefAttr signal) {
  return get(context, ModportDirectionAttr::get(context, direction), signal);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto builder = parser.getBuilder();
  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<hw::module_like_impl::PortParse> ports;
  TypeAttr modType;
  if (failed(
          hw::module_like_impl::parseModuleSignature(parser, ports, modType)))
    return failure();

  result.addAttribute(FuncOp::getModuleTypeAttrName(result.name), modType);

  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  auto unknownLoc = builder.getUnknownLoc();
  SmallVector<Attribute> attrs, inputLocs, outputLocs;
  auto nonEmptyLocsFn = [unknownLoc](Attribute attr) {
    return attr && cast<Location>(attr) != unknownLoc;
  };

  for (auto &port : ports) {
    attrs.push_back(port.attrs ? port.attrs : builder.getDictionaryAttr({}));
    auto loc = port.sourceLoc ? Location(*port.sourceLoc) : unknownLoc;
    (port.direction == hw::PortInfo::Direction::Output ? outputLocs : inputLocs)
        .push_back(loc);
  }

  result.addAttribute(FuncOp::getPerArgumentAttrsAttrName(result.name),
                      builder.getArrayAttr(attrs));

  if (llvm::any_of(outputLocs, nonEmptyLocsFn))
    result.addAttribute(FuncOp::getResultLocsAttrName(result.name),
                        builder.getArrayAttr(outputLocs));
  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // Add the entry block arguments.
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  for (auto &port : ports)
    if (port.direction != hw::ModulePort::Direction::Output)
      entryArgs.push_back(port);

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();

  mlir::OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs,
                                 /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  } else {
    if (llvm::any_of(inputLocs, nonEmptyLocsFn))
      result.addAttribute(FuncOp::getInputLocsAttrName(result.name),
                          builder.getArrayAttr(inputLocs));
  }

  return success();
}

void FuncOp::getAsmBlockArgumentNames(mlir::Region &region,
                                      mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  // Assign port names to the bbargs.
  auto func = cast<FuncOp>(region.getParentOp());

  auto *block = &region.front();

  auto names = func.getModuleType().getInputNames();
  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
    // Let mlir deterministically convert names to valid identifiers
    setNameFn(block->getArgument(i), cast<StringAttr>(names[i]));
  }
}

Type FuncOp::getExplicitlyReturnedType() {
  if (!getPerArgumentAttrs() || getNumOutputs() == 0)
    return {};

  // Check if the last port is used as an explicit return.
  auto lastArgument = getModuleType().getPorts().back();
  auto lastArgumentAttr = dyn_cast<DictionaryAttr>(
      getPerArgumentAttrsAttr()[getPerArgumentAttrsAttr().size() - 1]);

  if (lastArgument.dir == hw::ModulePort::Output && lastArgumentAttr &&
      lastArgumentAttr.getAs<UnitAttr>(getExplicitlyReturnedAttrName()))
    return lastArgument.type;
  return {};
}

ArrayRef<Attribute> FuncOp::getAllPortAttrs() {
  if (getPerArgumentAttrs())
    return getPerArgumentAttrs()->getValue();
  return {};
}

void FuncOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  setPerArgumentAttrsAttr(ArrayAttr::get(getContext(), attrs));
}

void FuncOp::removeAllPortAttrs() { setPerArgumentAttrsAttr({}); }
SmallVector<Location> FuncOp::getAllPortLocs() {
  SmallVector<Location> portLocs;
  portLocs.reserve(getNumPorts());
  auto resultLocs = getResultLocsAttr();
  unsigned inputCount = 0;
  auto modType = getModuleType();
  auto unknownLoc = UnknownLoc::get(getContext());
  auto *body = getBodyBlock();
  auto inputLocs = getInputLocsAttr();
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    if (modType.isOutput(i)) {
      auto loc = resultLocs
                     ? cast<Location>(
                           resultLocs.getValue()[portLocs.size() - inputCount])
                     : unknownLoc;
      portLocs.push_back(loc);
    } else {
      auto loc = body ? body->getArgument(inputCount).getLoc()
                      : (inputLocs ? cast<Location>(inputLocs[inputCount])
                                   : unknownLoc);
      portLocs.push_back(loc);
      ++inputCount;
    }
  }
  return portLocs;
}

void FuncOp::setAllPortLocsAttrs(llvm::ArrayRef<mlir::Attribute> locs) {
  SmallVector<Attribute> resultLocs, inputLocs;
  unsigned inputCount = 0;
  auto modType = getModuleType();
  auto *body = getBodyBlock();
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    if (modType.isOutput(i))
      resultLocs.push_back(locs[i]);
    else if (body)
      body->getArgument(inputCount++).setLoc(cast<Location>(locs[i]));
    else // Need to store locations in an attribute if declaration.
      inputLocs.push_back(locs[i]);
  }
  setResultLocsAttr(ArrayAttr::get(getContext(), resultLocs));
  if (!body)
    setInputLocsAttr(ArrayAttr::get(getContext(), inputLocs));
}

SmallVector<hw::PortInfo> FuncOp::getPortList() { return getPortList(false); }

hw::PortInfo FuncOp::getPort(size_t idx) {
  auto modTy = getHWModuleType();
  auto emptyDict = DictionaryAttr::get(getContext());
  LocationAttr loc = getPortLoc(idx);
  DictionaryAttr attrs = dyn_cast_or_null<DictionaryAttr>(getPortAttrs(idx));
  if (!attrs)
    attrs = emptyDict;
  return {modTy.getPorts()[idx],
          modTy.isOutput(idx) ? modTy.getOutputIdForPortId(idx)
                              : modTy.getInputIdForPortId(idx),
          attrs, loc};
}

SmallVector<hw::PortInfo> FuncOp::getPortList(bool excludeExplicitReturn) {
  auto modTy = getModuleType();
  auto emptyDict = DictionaryAttr::get(getContext());
  auto skipLastArgument = getExplicitlyReturnedType() && excludeExplicitReturn;
  SmallVector<hw::PortInfo> retval;
  auto portAttr = getAllPortLocs();
  for (unsigned i = 0, e = skipLastArgument ? modTy.getNumPorts() - 1
                                            : modTy.getNumPorts();
       i < e; ++i) {
    DictionaryAttr attrs = emptyDict;
    if (auto perArgumentAttr = getPerArgumentAttrs())
      if (auto argumentAttr =
              dyn_cast_or_null<DictionaryAttr>((*perArgumentAttr)[i]))
        attrs = argumentAttr;

    retval.push_back({modTy.getPorts()[i],
                      modTy.isOutput(i) ? modTy.getOutputIdForPortId(i)
                                        : modTy.getInputIdForPortId(i),
                      attrs, portAttr[i]});
  }
  return retval;
}

void FuncOp::print(OpAsmPrinter &p) {
  FuncOp op = *this;
  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);
  hw::module_like_impl::printModuleSignatureNew(
      p, op.getBody(), op.getModuleType(),
      op.getPerArgumentAttrsAttr()
          ? ArrayRef<Attribute>(op.getPerArgumentAttrsAttr().getValue())
          : ArrayRef<Attribute>{},
      getAllPortLocs());

  mlir::function_interface_impl::printFunctionAttributes(
      p, op,
      {visibilityAttrName, getModuleTypeAttrName(),
       getPerArgumentAttrsAttrName(), getInputLocsAttrName(),
       getResultLocsAttrName()});
  // Print the body if this is not an external function.
  Region &body = op->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto func = getParentOp<sv::FuncOp>();
  auto funcResults = func.getResultTypes();
  auto returnedValues = getOperands();
  if (funcResults.size() != returnedValues.size())
    return emitOpError("must have same number of operands as region results.");
  // Check that the types of our operands and the region's results match.
  for (size_t i = 0, e = funcResults.size(); i < e; ++i) {
    if (funcResults[i] != returnedValues[i].getType()) {
      emitOpError("output types must match function. In "
                  "operand ")
          << i << ", expected " << funcResults[i] << ", but got "
          << returnedValues[i].getType() << ".";
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Call Ops
//===----------------------------------------------------------------------===//

static Value
getExplicitlyReturnedValueImpl(sv::FuncOp op,
                               mlir::Operation::result_range results) {
  if (!op.getExplicitlyReturnedType())
    return {};
  return results.back();
}

Value FuncCallOp::getExplicitlyReturnedValue(sv::FuncOp op) {
  return getExplicitlyReturnedValueImpl(op, getResults());
}

Value FuncCallProceduralOp::getExplicitlyReturnedValue(sv::FuncOp op) {
  return getExplicitlyReturnedValueImpl(op, getResults());
}

LogicalResult
FuncCallProceduralOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto referencedOp = dyn_cast_or_null<sv::FuncOp>(
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr()));
  if (!referencedOp)
    return emitError("cannot find function declaration '")
           << getCallee() << "'";
  return success();
}

LogicalResult FuncCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto referencedOp = dyn_cast_or_null<sv::FuncOp>(
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr()));
  if (!referencedOp)
    return emitError("cannot find function declaration '")
           << getCallee() << "'";

  // Non-procedural call cannot have output arguments.
  if (referencedOp.getNumOutputs() != 1 ||
      !referencedOp.getExplicitlyReturnedType()) {
    auto diag = emitError()
                << "function called in a non-procedural region must "
                   "return a single result";
    diag.attachNote(referencedOp.getLoc()) << "doesn't satisfy the constraint";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FuncDPIImportOp
//===----------------------------------------------------------------------===//

LogicalResult
FuncDPIImportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto referencedOp = dyn_cast_or_null<sv::FuncOp>(
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr()));

  if (!referencedOp)
    return emitError("cannot find function declaration '")
           << getCallee() << "'";
  if (!referencedOp.isDeclaration())
    return emitError("imported function must be a declaration but '")
           << getCallee() << "' is defined";
  return success();
}

//===----------------------------------------------------------------------===//
// Assert Property Like ops
//===----------------------------------------------------------------------===//

namespace AssertPropertyLikeOp {
// Check that a clock is never given without an event
// and that an event is never given with a clock.
static LogicalResult verify(Value clock, bool eventExists, mlir::Location loc) {
  if ((!clock && eventExists) || (clock && !eventExists))
    return mlir::emitError(
        loc, "Every clock must be associated to an even and vice-versa!");
  return success();
}
} // namespace AssertPropertyLikeOp

LogicalResult AssertPropertyOp::verify() {
  return AssertPropertyLikeOp::verify(getClock(), getEvent().has_value(),
                                      getLoc());
}

LogicalResult AssumePropertyOp::verify() {
  return AssertPropertyLikeOp::verify(getClock(), getEvent().has_value(),
                                      getLoc());
}

LogicalResult CoverPropertyOp::verify() {
  return AssertPropertyLikeOp::verify(getClock(), getEvent().has_value(),
                                      getLoc());
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.cpp.inc"
