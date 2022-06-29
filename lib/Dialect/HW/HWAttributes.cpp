//===- HWAttributes.cpp - Implement HW attributes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace circt;
using namespace circt::hw;

// Internal method used for .mlir file parsing, defined below.
static Attribute parseParamExprWithOpcode(StringRef opcode, DialectAsmParser &p,
                                          Type type);

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/HW/HWAttributes.cpp.inc"

void HWDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/HW/HWAttributes.cpp.inc"
      >();
}

Attribute HWDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult = generatedAttributeParser(p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;

  // Parse "#hw.param.expr.add" as ParamExprAttr.
  if (attrName.startswith(ParamExprAttr::getMnemonic())) {
    auto string = attrName.drop_front(ParamExprAttr::getMnemonic().size());
    if (string.front() == '.')
      return parseParamExprWithOpcode(string.drop_front(), p, type);
  }

  p.emitError(p.getNameLoc(), "Unexpected hw attribute '" + attrName + "'");
  return {};
}

void HWDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

//===----------------------------------------------------------------------===//
// OutputFileAttr
//===----------------------------------------------------------------------===//

static std::string canonicalizeFilename(const Twine &directory,
                                        const Twine &filename) {
  SmallString<128> fullPath;

  // If the filename is an absolute path, we don't need the directory.
  if (llvm::sys::path::is_absolute(filename))
    filename.toVector(fullPath);
  else
    llvm::sys::path::append(fullPath, directory, filename);

  // If this is a directory target, we need to ensure it ends with a `/`
  if (filename.isTriviallyEmpty() && !fullPath.endswith("/"))
    fullPath += "/";

  return std::string(fullPath);
}

OutputFileAttr OutputFileAttr::getFromFilename(MLIRContext *context,
                                               const Twine &filename,
                                               bool excludeFromFileList,
                                               bool includeReplicatedOps) {
  return OutputFileAttr::getFromDirectoryAndFilename(
      context, "", filename, excludeFromFileList, includeReplicatedOps);
}

OutputFileAttr OutputFileAttr::getFromDirectoryAndFilename(
    MLIRContext *context, const Twine &directory, const Twine &filename,
    bool excludeFromFileList, bool includeReplicatedOps) {
  auto canonicalized = canonicalizeFilename(directory, filename);
  return OutputFileAttr::get(StringAttr::get(context, canonicalized),
                             BoolAttr::get(context, excludeFromFileList),
                             BoolAttr::get(context, includeReplicatedOps));
}

OutputFileAttr OutputFileAttr::getAsDirectory(MLIRContext *context,
                                              const Twine &directory,
                                              bool excludeFromFileList,
                                              bool includeReplicatedOps) {
  return getFromDirectoryAndFilename(context, directory, "",
                                     excludeFromFileList, includeReplicatedOps);
}

bool OutputFileAttr::isDirectory() {
  return getFilename().getValue().endswith("/");
}

/// Option         ::= 'excludeFromFileList' | 'includeReplicatedOp'
/// OutputFileAttr ::= 'output_file<' directory ',' name (',' Option)* '>'
Attribute OutputFileAttr::parse(AsmParser &p, Type type) {
  StringAttr filename;
  if (p.parseLess() || p.parseAttribute<StringAttr>(filename))
    return Attribute();

  // Parse the additional keyword attributes.  Its easier to let people specify
  // these more than once than to detect the problem and do something about it.
  bool excludeFromFileList = false;
  bool includeReplicatedOps = false;
  while (true) {
    if (p.parseOptionalComma())
      break;
    if (!p.parseOptionalKeyword("excludeFromFileList"))
      excludeFromFileList = true;
    else if (!p.parseKeyword("includeReplicatedOps",
                             "or 'excludeFromFileList'"))
      includeReplicatedOps = true;
    else
      return Attribute();
  }

  if (p.parseGreater())
    return Attribute();

  auto *context = p.getContext();

  return OutputFileAttr::get(context, filename,
                             BoolAttr::get(context, excludeFromFileList),
                             BoolAttr::get(context, includeReplicatedOps));
}

void OutputFileAttr::print(AsmPrinter &p) const {
  p << "<" << getFilename();
  if (getExcludeFromFilelist().getValue())
    p << ", excludeFromFileList";
  if (getIncludeReplicatedOps().getValue())
    p << ", includeReplicatedOps";
  p << ">";
}

//===----------------------------------------------------------------------===//
// FileListAttr
//===----------------------------------------------------------------------===//

FileListAttr FileListAttr::getFromFilename(MLIRContext *context,
                                           const Twine &filename) {
  auto canonicalized = canonicalizeFilename("", filename);
  return FileListAttr::get(StringAttr::get(context, canonicalized));
}

//===----------------------------------------------------------------------===//
// EnumValueAttr
//===----------------------------------------------------------------------===//

Attribute EnumValueAttr::parse(AsmParser &p, Type) {
  StringRef enumerator;
  EnumType type;
  if (p.parseLess() || p.parseKeyword(&enumerator) || p.parseComma() ||
      p.parseType(type) || p.parseGreater())
    return Attribute();
  return EnumValueAttr::get(p.getEncodedSourceLoc(p.getCurrentLocation()),
                            StringAttr::get(p.getContext(), enumerator), type);
}

void EnumValueAttr::print(AsmPrinter &p) const {
  p << "<" << getValue().getValue() << ", ";
  p.printType(getType().getValue());
  p << ">";
}

Attribute EnumValueAttr::get(Location loc, StringAttr value,
                             hw::EnumType type) {
  // Check whether the provided value is a member of the enum type.
  if (!type.contains(value.getValue())) {
    emitError(loc) << "enum value '" << value.getValue()
                   << "' is not a member of enum type " << type;
    return Attribute();
  }

  return Base::get(value.getContext(), value, TypeAttr::get(type));
}

//===----------------------------------------------------------------------===//
// InnerRefAttr
//===----------------------------------------------------------------------===//

Attribute InnerRefAttr::parse(AsmParser &p, Type type) {
  SymbolRefAttr attr;
  if (p.parseLess() || p.parseAttribute<SymbolRefAttr>(attr) ||
      p.parseGreater())
    return Attribute();
  if (attr.getNestedReferences().size() != 1)
    return Attribute();
  return InnerRefAttr::get(attr.getRootReference(), attr.getLeafReference());
}

void InnerRefAttr::print(AsmPrinter &p) const {
  p << "<";
  p.printSymbolName(getModule().getValue());
  p << "::";
  p.printSymbolName(getName().getValue());
  p << ">";
}

/// Get an InnerRefAttr, and add the sym to the op if not already
/// there. Also reponsibility of client to ensure the symName is unique.
InnerRefAttr InnerRefAttr::getFromOperation(mlir::Operation *op,
                                            mlir::StringAttr symName,
                                            mlir::StringAttr moduleName) {
  char attrName[] = "inner_sym";
  auto attr = op->getAttrOfType<StringAttr>(attrName);
  if (!attr) {
    attr = symName;
    op->setAttr(attrName, attr);
  }
  return InnerRefAttr::get(moduleName, attr);
}

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclAttr::parse(AsmParser &p, Type type) {
  llvm::errs() << "Should never parse raw\n";
  abort();
}

void ParamDeclAttr::print(AsmPrinter &p) const {
  p << "<" << getName() << ": " << getType();
  if (getValue())
    p << " = " << getValue();
  p << ">";
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclRefAttr::parse(AsmParser &p, Type type) {
  StringAttr name;
  if (p.parseLess() || p.parseAttribute(name) || p.parseGreater())
    return Attribute();

  return ParamDeclRefAttr::get(name, type);
}

void ParamDeclRefAttr::print(AsmPrinter &p) const {
  p << "<" << getName() << ">";
}

//===----------------------------------------------------------------------===//
// ParamVerbatimAttr
//===----------------------------------------------------------------------===//

Attribute ParamVerbatimAttr::parse(AsmParser &p, Type type) {
  StringAttr text;
  if (p.parseLess() || p.parseAttribute(text) || p.parseGreater())
    return Attribute();

  return ParamVerbatimAttr::get(p.getContext(), text, type);
}

void ParamVerbatimAttr::print(AsmPrinter &p) const {
  p << "<" << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// ParamExprAttr
//===----------------------------------------------------------------------===//

/// Given a binary function, if the two operands are known constant integers,
/// use the specified fold function to compute the result.
static Attribute foldBinaryOp(
    ArrayRef<Attribute> operands,
    llvm::function_ref<APInt(const APInt &, const APInt &)> calculate) {
  assert(operands.size() == 2 && "binary operator always has two operands");
  if (auto lhs = operands[0].dyn_cast<IntegerAttr>())
    if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
      return IntegerAttr::get(lhs.getType(),
                              calculate(lhs.getValue(), rhs.getValue()));
  return {};
}

/// Given a unary function, if the operand is a known constant integer,
/// use the specified fold function to compute the result.
static Attribute
foldUnaryOp(ArrayRef<Attribute> operands,
            llvm::function_ref<APInt(const APInt &)> calculate) {
  assert(operands.size() == 1 && "unary operator always has one operand");
  if (auto intAttr = operands[0].dyn_cast<IntegerAttr>())
    return IntegerAttr::get(intAttr.getType(), calculate(intAttr.getValue()));
  return {};
}

/// If the specified attribute is a ParamExprAttr with the specified opcode,
/// return it.  Otherwise return null.
static ParamExprAttr dyn_castPE(PEO opcode, Attribute value) {
  if (auto expr = value.dyn_cast<ParamExprAttr>())
    if (expr.getOpcode() == opcode)
      return expr;
  return {};
}

/// This implements a < comparison for two operands to an associative operation
/// imposing an ordering upon them.
///
/// The ordering provided puts more complex things to the start of the list,
/// from left to right:
///    expressions :: verbatims :: decl.refs :: constant
///
static bool paramExprOperandSortPredicate(Attribute lhs, Attribute rhs) {
  // Simplify the code below - we never have to care about exactly equal values.
  if (lhs == rhs)
    return false;

  // All expressions are "less than" a constant, since they appear on the right.
  if (rhs.isa<IntegerAttr>()) {
    // We don't bother to order constants w.r.t. each other since they will be
    // folded - they can all compare equal.
    return !lhs.isa<IntegerAttr>();
  }
  if (lhs.isa<IntegerAttr>())
    return false;

  // Next up are named parameters.
  if (auto rhsParam = rhs.dyn_cast<ParamDeclRefAttr>()) {
    // Parameters are sorted lexically w.r.t. each other.
    if (auto lhsParam = lhs.dyn_cast<ParamDeclRefAttr>())
      return lhsParam.getName().getValue() < rhsParam.getName().getValue();
    // They otherwise appear on the right of other things.
    return true;
  }
  if (lhs.isa<ParamDeclRefAttr>())
    return false;

  // Next up are verbatim parameters.
  if (auto rhsParam = rhs.dyn_cast<ParamVerbatimAttr>()) {
    // Verbatims are sorted lexically w.r.t. each other.
    if (auto lhsParam = lhs.dyn_cast<ParamVerbatimAttr>())
      return lhsParam.getValue().getValue() < rhsParam.getValue().getValue();
    // They otherwise appear on the right of other things.
    return true;
  }
  if (lhs.isa<ParamVerbatimAttr>())
    return false;

  // The only thing left are nested expressions.
  auto lhsExpr = lhs.cast<ParamExprAttr>(), rhsExpr = rhs.cast<ParamExprAttr>();
  // Sort by the string form of the opcode, e.g. add, .. mul,... then xor.
  if (lhsExpr.getOpcode() != rhsExpr.getOpcode())
    return stringifyPEO(lhsExpr.getOpcode()) <
           stringifyPEO(rhsExpr.getOpcode());

  // If they are the same opcode, then sort by arity: more complex to the left.
  ArrayRef<Attribute> lhsOperands = lhsExpr.getOperands(),
                      rhsOperands = rhsExpr.getOperands();
  if (lhsOperands.size() != rhsOperands.size())
    return lhsOperands.size() > rhsOperands.size();

  // We know the two subexpressions are different (they'd otherwise be pointer
  // equivalent) so just go compare all of the elements.
  for (size_t i = 0, e = lhsOperands.size(); i != e; ++i) {
    if (paramExprOperandSortPredicate(lhsOperands[i], rhsOperands[i]))
      return true;
    if (paramExprOperandSortPredicate(rhsOperands[i], lhsOperands[i]))
      return false;
  }

  llvm_unreachable("expressions should never be equivalent");
  return false;
}

/// Given a fully associative variadic integer operation, constant fold any
/// constant operands and move them to the right.  If the whole expression is
/// constant, then return that, otherwise update the operands list.
static Attribute simplifyAssocOp(
    PEO opcode, SmallVector<Attribute, 4> &operands,
    llvm::function_ref<APInt(const APInt &, const APInt &)> calculateFn,
    llvm::function_ref<bool(const APInt &)> identityConstantFn,
    llvm::function_ref<bool(const APInt &)> destructiveConstantFn = {}) {
  auto type = operands[0].getType();
  assert(isHWIntegerType(type));
  if (operands.size() == 1)
    return operands[0];

  // Flatten any of the same operation into the operand list:
  // `(add x, (add y, z))` => `(add x, y, z)`.
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto subexpr = dyn_castPE(opcode, operands[i])) {
      std::swap(operands[i], operands.back());
      operands.pop_back();
      --e;
      --i;
      operands.append(subexpr.getOperands().begin(),
                      subexpr.getOperands().end());
    }
  }

  // Impose an ordering on the operands, pushing subexpressions to the left and
  // constants to the right, with verbatims and parameters in the middle - but
  // predictably ordered w.r.t. each other.
  llvm::stable_sort(operands, paramExprOperandSortPredicate);

  // Merge any constants, they will appear at the back of the operand list now.
  if (operands.back().isa<IntegerAttr>()) {
    while (operands.size() >= 2 &&
           operands[operands.size() - 2].isa<IntegerAttr>()) {
      APInt c1 = operands.pop_back_val().cast<IntegerAttr>().getValue();
      APInt c2 = operands.pop_back_val().cast<IntegerAttr>().getValue();
      auto resultConstant = IntegerAttr::get(type, calculateFn(c1, c2));
      operands.push_back(resultConstant);
    }

    auto resultCst = operands.back().cast<IntegerAttr>();

    // If the resulting constant is the destructive constant (e.g. `x*0`), then
    // return it.
    if (destructiveConstantFn && destructiveConstantFn(resultCst.getValue()))
      return resultCst;

    // Remove the constant back to our operand list if it is the identity
    // constant for this operator (e.g. `x*1`) and there are other operands.
    if (identityConstantFn(resultCst.getValue()) && operands.size() != 1)
      operands.pop_back();
  }

  return operands.size() == 1 ? operands[0] : Attribute();
}

/// Analyze an operand to an add.  If it is a multiplication by a constant (e.g.
/// `(a*b*42)` then split it into the non-constant and the constant portions
/// (e.g. `a*b` and `42`).  Otherwise return the operand as the first value and
/// null as the second (standin for "multiplication by 1").
static std::pair<Attribute, Attribute> decomposeAddend(Attribute operand) {
  if (auto mul = dyn_castPE(PEO::Mul, operand))
    if (auto cst = mul.getOperands().back().dyn_cast<IntegerAttr>()) {
      auto nonCst = ParamExprAttr::get(PEO::Mul, mul.getOperands().drop_back());
      return {nonCst, cst};
    }
  return {operand, Attribute()};
}

static Attribute getOneOfType(Type type) {
  return IntegerAttr::get(type, APInt(type.getIntOrFloatBitWidth(), 1));
}

static Attribute simplifyAdd(SmallVector<Attribute, 4> &operands) {
  if (auto result = simplifyAssocOp(
          PEO::Add, operands, [](auto a, auto b) { return a + b; },
          /*identityCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // Canonicalize the add by splitting all addends into their variable and
  // constant factors.
  SmallVector<std::pair<Attribute, Attribute>> decomposedOperands;
  llvm::SmallDenseSet<Attribute> nonConstantParts;
  for (auto &op : operands) {
    decomposedOperands.push_back(decomposeAddend(op));

    // Keep track of non-constant parts we've already seen.  If we see multiple
    // uses of the same value, then we can fold them together with a multiply.
    // This handles things like `(a+b+a)` => `(a*2 + b)` and `(a*2 + b + a)` =>
    // `(a*3 + b)`.
    if (!nonConstantParts.insert(decomposedOperands.back().first).second) {
      // The thing we multiply will be the common expression.
      Attribute mulOperand = decomposedOperands.back().first;

      // Find the index of the first occurrence.
      size_t i = 0;
      while (decomposedOperands[i].first != mulOperand)
        ++i;
      // Remove both occurrences from the operand list.
      operands.erase(operands.begin() + (&op - &operands[0]));
      operands.erase(operands.begin() + i);

      auto type = mulOperand.getType();
      auto c1 = decomposedOperands[i].second,
           c2 = decomposedOperands.back().second;
      // Fill in missing constant multiplicands with 1.
      if (!c1)
        c1 = getOneOfType(type);
      if (!c2)
        c2 = getOneOfType(type);
      // Re-add the "a"*(c1+c2) expression to the operand list and
      // re-canonicalize.
      auto constant = ParamExprAttr::get(PEO::Add, c1, c2);
      auto mulCst = ParamExprAttr::get(PEO::Mul, mulOperand, constant);
      operands.push_back(mulCst);
      return ParamExprAttr::get(PEO::Add, operands);
    }
  }

  return {};
}

static Attribute simplifyMul(SmallVector<Attribute, 4> &operands) {
  if (auto result = simplifyAssocOp(
          PEO::Mul, operands, [](auto a, auto b) { return a * b; },
          /*identityCst*/ [](auto cst) { return cst.isOne(); },
          /*destructiveCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // We always build a sum-of-products representation, so if we see an addition
  // as a subexpr, we need to pull it out: (a+b)*c*d ==> (a*c*d + b*c*d).
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto subexpr = dyn_castPE(PEO::Add, operands[i])) {
      // Pull the `c*d` operands out - it is whatever operands remain after
      // removing the `(a+b)` term.
      SmallVector<Attribute> mulOperands(operands.begin(), operands.end());
      mulOperands.erase(mulOperands.begin() + i);

      // Build each add operand.
      SmallVector<Attribute> addOperands;
      for (auto addOperand : subexpr.getOperands()) {
        mulOperands.push_back(addOperand);
        addOperands.push_back(ParamExprAttr::get(PEO::Mul, mulOperands));
        mulOperands.pop_back();
      }
      // Canonicalize and form the add expression.
      return ParamExprAttr::get(PEO::Add, addOperands);
    }
  }

  return {};
}
static Attribute simplifyAnd(SmallVector<Attribute, 4> &operands) {
  return simplifyAssocOp(
      PEO::And, operands, [](auto a, auto b) { return a & b; },
      /*identityCst*/ [](auto cst) { return cst.isAllOnes(); },
      /*destructiveCst*/ [](auto cst) { return cst.isZero(); });
}

static Attribute simplifyOr(SmallVector<Attribute, 4> &operands) {
  return simplifyAssocOp(
      PEO::Or, operands, [](auto a, auto b) { return a | b; },
      /*identityCst*/ [](auto cst) { return cst.isZero(); },
      /*destructiveCst*/ [](auto cst) { return cst.isAllOnes(); });
}

static Attribute simplifyXor(SmallVector<Attribute, 4> &operands) {
  return simplifyAssocOp(
      PEO::Xor, operands, [](auto a, auto b) { return a ^ b; },
      /*identityCst*/ [](auto cst) { return cst.isZero(); });
}

static Attribute simplifyShl(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));

  if (auto rhs = operands[1].dyn_cast<IntegerAttr>()) {
    // Constant fold simple integers.
    if (auto lhs = operands[0].dyn_cast<IntegerAttr>())
      return IntegerAttr::get(lhs.getType(),
                              lhs.getValue().shl(rhs.getValue()));

    // Canonicalize `x << cst` => `x * (1<<cst)` to compose correctly with
    // add/mul canonicalization.
    auto rhsCst = APInt::getOneBitSet(rhs.getValue().getBitWidth(),
                                      rhs.getValue().getZExtValue());
    return ParamExprAttr::get(PEO::Mul, operands[0],
                              IntegerAttr::get(rhs.getType(), rhsCst));
  }
  return {};
}

static Attribute simplifyShrU(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // Implement support for identities like `x >> 0`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isZero())
      return operands[0];

  return foldBinaryOp(operands, [](auto a, auto b) { return a.lshr(b); });
}

static Attribute simplifyShrS(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // Implement support for identities like `x >> 0`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isZero())
      return operands[0];

  return foldBinaryOp(operands, [](auto a, auto b) { return a.ashr(b); });
}

static Attribute simplifyDivU(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // Implement support for identities like `x/1`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isOne())
      return operands[0];

  return foldBinaryOp(operands, [](auto a, auto b) { return a.udiv(b); });
}

static Attribute simplifyDivS(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // Implement support for identities like `x/1`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isOne())
      return operands[0];

  return foldBinaryOp(operands, [](auto a, auto b) { return a.sdiv(b); });
}

static Attribute simplifyModU(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // Implement support for identities like `x%1`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isOne())
      return IntegerAttr::get(rhs.getType(), 0);

  return foldBinaryOp(operands, [](auto a, auto b) { return a.urem(b); });
}

static Attribute simplifyModS(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // Implement support for identities like `x%1`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isOne())
      return IntegerAttr::get(rhs.getType(), 0);

  return foldBinaryOp(operands, [](auto a, auto b) { return a.srem(b); });
}

static Attribute simplifyCLog2(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  return foldUnaryOp(operands, [](auto a) {
    // Following the Verilog spec, clog2(0) is 0
    return APInt(a.getBitWidth(), a == 0 ? 0 : a.ceilLogBase2());
  });
}

static Attribute simplifyStrConcat(SmallVector<Attribute, 4> &operands) {
  // Combine all adjacent strings.
  SmallVector<Attribute> newOperands;
  SmallVector<StringAttr> stringsToCombine;
  auto combineAndPush = [&]() {
    if (stringsToCombine.empty())
      return;
    // Concatenate buffered strings, push to ops.
    SmallString<32> newString;
    for (auto part : stringsToCombine)
      newString.append(part.getValue());
    newOperands.push_back(
        StringAttr::get(stringsToCombine[0].getContext(), newString));
    stringsToCombine.clear();
  };

  for (Attribute op : operands) {
    if (auto strOp = op.dyn_cast<StringAttr>()) {
      // Queue up adjacent strings.
      stringsToCombine.push_back(strOp);
    } else {
      combineAndPush();
      newOperands.push_back(op);
    }
  }
  combineAndPush();

  assert(!newOperands.empty());
  if (newOperands.size() == 1)
    return newOperands[0];
  if (newOperands.size() < operands.size())
    return ParamExprAttr::get(PEO::StrConcat, newOperands);
  return {};
}

/// Build a parameter expression.  This automatically canonicalizes and
/// folds, so it may not necessarily return a ParamExprAttr.
Attribute ParamExprAttr::get(PEO opcode, ArrayRef<Attribute> operandsIn) {
  assert(!operandsIn.empty() && "Cannot have expr with no operands");
  // All operands must have the same type, which is the type of the result.
  auto type = operandsIn.front().getType();
  assert(llvm::all_of(operandsIn.drop_front(),
                      [&](auto op) { return op.getType() == type; }));

  SmallVector<Attribute, 4> operands(operandsIn.begin(), operandsIn.end());

  // Verify and canonicalize parameter expressions.
  Attribute result;
  switch (opcode) {
  case PEO::Add:
    result = simplifyAdd(operands);
    break;
  case PEO::Mul:
    result = simplifyMul(operands);
    break;
  case PEO::And:
    result = simplifyAnd(operands);
    break;
  case PEO::Or:
    result = simplifyOr(operands);
    break;
  case PEO::Xor:
    result = simplifyXor(operands);
    break;
  case PEO::Shl:
    result = simplifyShl(operands);
    break;
  case PEO::ShrU:
    result = simplifyShrU(operands);
    break;
  case PEO::ShrS:
    result = simplifyShrS(operands);
    break;
  case PEO::DivU:
    result = simplifyDivU(operands);
    break;
  case PEO::DivS:
    result = simplifyDivS(operands);
    break;
  case PEO::ModU:
    result = simplifyModU(operands);
    break;
  case PEO::ModS:
    result = simplifyModS(operands);
    break;
  case PEO::CLog2:
    result = simplifyCLog2(operands);
    break;
  case PEO::StrConcat:
    result = simplifyStrConcat(operands);
    break;
  }

  // If we folded to an operand, return it.
  if (result)
    return result;

  return Base::get(operands[0].getContext(), opcode, operands, type);
}

Attribute ParamExprAttr::parse(AsmParser &p, Type type) {
  // We require an opcode suffix like `#hw.param.expr.add`, we don't allow
  // parsing a plain `#hw.param.expr` on its own.
  p.emitError(p.getNameLoc(), "#hw.param.expr should have opcode suffix");
  return {};
}

/// Internal method used for .mlir file parsing when parsing the
/// "#hw.param.expr.mul" form of the attribute.
static Attribute parseParamExprWithOpcode(StringRef opcodeStr,
                                          DialectAsmParser &p, Type type) {
  SmallVector<Attribute> operands;
  if (p.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            operands.push_back({});
            return p.parseAttribute(operands.back(), type);
          }))
    return {};

  Optional<PEO> opcode = symbolizePEO(opcodeStr);
  if (!opcode.hasValue()) {
    p.emitError(p.getNameLoc(), "unknown parameter expr operator name");
    return {};
  }

  return ParamExprAttr::get(*opcode, operands);
}

void ParamExprAttr::print(AsmPrinter &p) const {
  p << "." << stringifyPEO(getOpcode()) << '<';
  llvm::interleaveComma(getOperands(), p.getStream(),
                        [&](Attribute op) { p.printAttributeWithoutType(op); });
  p << '>';
}

// Replaces any ParamDeclRefAttr within a parametric expression with its
// corresponding value from the map of provided parameters.
static FailureOr<Attribute>
replaceDeclRefInExpr(Location loc,
                     const std::map<std::string, Attribute> &parameters,
                     Attribute paramAttr) {
  if (paramAttr.dyn_cast<IntegerAttr>()) {
    // Nothing to do, constant value.
    return paramAttr;
  }
  if (auto paramRefAttr = paramAttr.dyn_cast<hw::ParamDeclRefAttr>()) {
    // Get the value from the provided parameters.
    auto it = parameters.find(paramRefAttr.getName().str());
    if (it == parameters.end())
      return emitError(loc)
             << "Could not find parameter " << paramRefAttr.getName().str()
             << " in the provided parameters for the expression!";
    return it->second;
  }
  if (auto paramExprAttr = paramAttr.dyn_cast<hw::ParamExprAttr>()) {
    // Recurse into all operands of the expression.
    llvm::SmallVector<Attribute, 4> replacedOperands;
    for (auto operand : paramExprAttr.getOperands()) {
      auto res = replaceDeclRefInExpr(loc, parameters, operand);
      if (failed(res))
        return {failure()};
      replacedOperands.push_back(res.getValue());
    }
    return {
        hw::ParamExprAttr::get(paramExprAttr.getOpcode(), replacedOperands)};
  }
  llvm_unreachable("Unhandled parametric attribute");
  return {};
}

FailureOr<Attribute> hw::evaluateParametricAttr(Location loc,
                                                ArrayAttr parameters,
                                                Attribute paramAttr) {
  // Create a map of the provided parameters for faster lookup.
  std::map<std::string, Attribute> parameterMap;
  for (auto param : parameters) {
    auto paramDecl = param.cast<ParamDeclAttr>();
    parameterMap[paramDecl.getName().str()] = paramDecl.getValue();
  }

  // First, replace any ParamDeclRefAttr in the expression with its
  // corresponding value in 'parameters'.
  auto paramAttrRes = replaceDeclRefInExpr(loc, parameterMap, paramAttr);
  if (failed(paramAttrRes))
    return {failure()};
  paramAttr = paramAttrRes.getValue();

  // Then, evaluate the parametric attribute.
  if (paramAttr.isa<IntegerAttr>() || paramAttr.isa<hw::ParamDeclRefAttr>())
    return paramAttr;
  if (auto paramExprAttr = paramAttr.dyn_cast<hw::ParamExprAttr>()) {
    // Since any ParamDeclRefAttr was replaced within the expression,
    // we re-evaluate the expression through the existing ParamExprAttr
    // canonicalizer.
    return ParamExprAttr::get(paramExprAttr.getOpcode(),
                              paramExprAttr.getOperands());
  }

  llvm_unreachable("Unhandled parametric attribute");
  return Attribute();
}

FailureOr<Type> hw::evaluateParametricType(Location loc, ArrayAttr parameters,
                                           Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<hw::IntType>([&](hw::IntType t) -> FailureOr<Type> {
        auto evaluatedWidth =
            evaluateParametricAttr(loc, parameters, t.getWidth());
        if (failed(evaluatedWidth))
          return {failure()};

        // If the width was evaluated to a constant, return an `IntegerType`
        if (auto intAttr = evaluatedWidth->dyn_cast<IntegerAttr>())
          return {IntegerType::get(type.getContext(),
                                   intAttr.getValue().getSExtValue())};

        // Otherwise parameter references are still involved
        return hw::IntType::get(evaluatedWidth.getValue());
      })
      .Case<hw::ArrayType>([&](hw::ArrayType arrayType) -> FailureOr<Type> {
        auto size =
            evaluateParametricAttr(loc, parameters, arrayType.getSizeAttr());
        if (failed(size))
          return failure();
        auto elementType =
            evaluateParametricType(loc, parameters, arrayType.getElementType());
        if (failed(elementType))
          return failure();

        // If the size was evaluated to a constant, use a 64-bit integer
        // attribute version of it
        if (auto intAttr = size->dyn_cast<IntegerAttr>())
          return hw::ArrayType::get(
              arrayType.getContext(), elementType.getValue(),
              IntegerAttr::get(IntegerType::get(type.getContext(), 64),
                               intAttr.getValue().getSExtValue()));

        // Otherwise parameter references are still involved
        return hw::ArrayType::get(arrayType.getContext(),
                                  elementType.getValue(), size.getValue());
      })
      .Default([&](auto) { return type; });
}

// Returns true if any part of this parametric attribute contains a reference
// to a parameter declaration.
static bool isParamAttrWithParamRef(Attribute expr) {
  return llvm::TypeSwitch<Attribute, bool>(expr)
      .Case([](ParamExprAttr attr) {
        return llvm::any_of(attr.getOperands(), isParamAttrWithParamRef);
      })
      .Case([](ParamDeclRefAttr) { return true; })
      .Default([](auto) { return false; });
}

bool hw::isParametricType(mlir::Type t) {
  return llvm::TypeSwitch<Type, bool>(t)
      .Case<hw::IntType>(
          [&](hw::IntType t) { return isParamAttrWithParamRef(t.getWidth()); })
      .Case<hw::ArrayType>([&](hw::ArrayType arrayType) {
        return isParametricType(arrayType.getElementType()) ||
               isParamAttrWithParamRef(arrayType.getSizeAttr());
      })
      .Default([](auto) { return false; });
}
