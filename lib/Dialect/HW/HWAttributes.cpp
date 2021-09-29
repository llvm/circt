//===- HWAttributes.cpp - Implement HW attributes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
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
  auto parseResult =
      generatedAttributeParser(getContext(), p, attrName, type, attr);
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
Attribute OutputFileAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                Type type) {
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

  return OutputFileAttr::get(context, filename,
                             BoolAttr::get(context, excludeFromFileList),
                             BoolAttr::get(context, includeReplicatedOps));
}

void OutputFileAttr::print(DialectAsmPrinter &p) const {
  p << "output_file<" << getFilename();
  if (getExcludeFromFilelist().getValue())
    p << ", excludeFromFileList";
  if (getIncludeReplicatedOps().getValue())
    p << ", includeReplicatedOps";
  p << ">";
}

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclAttr::parse(MLIRContext *context, DialectAsmParser &p,
                               Type type) {
  llvm::errs() << "Should never parse raw\n";
  abort();
}

void ParamDeclAttr::print(DialectAsmPrinter &p) const {
  p << "param.decl<" << getName() << ": " << getType();
  if (getValue())
    p << " = " << getValue();
  p << ">";
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclRefAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                  Type type) {
  StringAttr name;
  if (p.parseLess() || p.parseAttribute(name) || p.parseGreater())
    return Attribute();

  return ParamDeclRefAttr::get(context, name, type);
}

void ParamDeclRefAttr::print(DialectAsmPrinter &p) const {
  p << "param.decl.ref<" << getName() << ">";
}

//===----------------------------------------------------------------------===//
// ParamVerbatimAttr
//===----------------------------------------------------------------------===//

Attribute ParamVerbatimAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                   Type type) {
  StringAttr text;
  if (p.parseLess() || p.parseAttribute(text) || p.parseGreater())
    return Attribute();

  return ParamVerbatimAttr::get(context, text, type);
}

void ParamVerbatimAttr::print(DialectAsmPrinter &p) const {
  p << "param.verbatim<" << getValue() << ">";
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

/// If the specified attribute is a ParamExprAttr with the specified opcode,
/// return it.  Otherwise return null.
static ParamExprAttr dyn_castPE(PEO opcode, Attribute value) {
  if (auto expr = value.dyn_cast<ParamExprAttr>())
    if (expr.getOpcode() == opcode)
      return expr;
  return {};
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

  // Scan for any constants.
  size_t i = 0, e = operands.size();
  for (; i != e && !operands[i].isa<IntegerAttr>(); ++i)
    ;

  // No constants or constant at the end: no work to do.
  if (i >= e - 1)
    return {};

  // Take the constant out of the list.
  APInt cst = operands[i].cast<IntegerAttr>().getValue();
  operands[i] = operands.back();
  operands.pop_back();
  --e;

  // Scan for any more constants, merging them into this one.
  while (i != e) {
    if (auto cst2 = operands[i].dyn_cast<IntegerAttr>()) {
      cst = calculateFn(cst, cst2.getValue());
      operands[i] = operands.back();
      operands.pop_back();
      --e;
    } else {
      ++i;
    }
  }

  // All operands were constants, then that is our answer.
  auto resultConstant = IntegerAttr::get(type, cst);
  if (operands.empty())
    return resultConstant;

  // If the resulting constant is the destructive constant (e.g. `x*0`), then
  // return it.
  if (destructiveConstantFn && destructiveConstantFn(cst))
    return resultConstant;

  // Add the constant back to our operand list unless it is the identity
  // constant for this operator (e.g. `x*1`).
  if (!identityConstantFn(cst))
    operands.push_back(resultConstant);

  return operands.size() == 1 ? operands[0] : Attribute();
}

/// Analyze an operand to an add.  If it is a multiplication by a constant (e.g.
/// `(a*b*42)` then split it into the non-constant and the constant portions
/// (e.g. `a*b` and `42`).  Otherwise return the operand as the first value and
/// null as the second (standin for "multiplication 1").
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
  // TODO: Implement support for identities like `x >> 0`.
  return foldBinaryOp(operands, [](auto a, auto b) { return a.lshr(b); });
}

static Attribute simplifyShrS(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // TODO: Implement support for identities like `x >> 0`.
  return foldBinaryOp(operands, [](auto a, auto b) { return a.ashr(b); });
}

static Attribute simplifyDivU(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // TODO: Implement support for identities like `x/1`.
  return foldBinaryOp(operands, [](auto a, auto b) { return a.udiv(b); });
}

static Attribute simplifyDivS(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // TODO: Implement support for identities like `x/1`.
  return foldBinaryOp(operands, [](auto a, auto b) { return a.sdiv(b); });
}

static Attribute simplifyModU(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // TODO: Implement support for identities like `x%1`.
  return foldBinaryOp(operands, [](auto a, auto b) { return a.urem(b); });
}

static Attribute simplifyModS(SmallVector<Attribute, 4> &operands) {
  assert(isHWIntegerType(operands[0].getType()));
  // TODO: Implement support for identities like `x%1`.
  return foldBinaryOp(operands, [](auto a, auto b) { return a.srem(b); });
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
  }

  // If we folded to an operand, return it.
  if (result)
    return result;

  return Base::get(operands[0].getContext(), opcode, operands, type);
}

Attribute ParamExprAttr::parse(MLIRContext *context, DialectAsmParser &p,
                               Type type) {
  // We require an opcode suffix like `#hw.param.expr.add`, we don't allow
  // parsing a plain `#hw.param.expr` on its own.
  p.emitError(p.getNameLoc(), "#hw.param.expr should have opcode suffix");
  return {};
}

/// Internal method used for .mlir file parsing when parsing the
/// "#hw.param.expr.mul" form of the attribute.
static Attribute parseParamExprWithOpcode(StringRef opcodeStr,
                                          DialectAsmParser &p, Type type) {
  // FIXME(LLVM Merge): use parseCommaSeparatedList
  SmallVector<Attribute> operands;
  operands.push_back({});
  if (p.parseLess() || p.parseAttribute(operands.back(), type))
    return {};

  while (succeeded(p.parseOptionalComma())) {
    operands.push_back({});
    if (p.parseAttribute(operands.back(), type))
      return {};
  }

  if (p.parseGreater())
    return {};

  Optional<PEO> opcode = symbolizePEO(opcodeStr);
  if (!opcode.hasValue()) {
    p.emitError(p.getNameLoc(), "unknown parameter expr operator name");
    return {};
  }

  return ParamExprAttr::get(*opcode, operands);
}

void ParamExprAttr::print(DialectAsmPrinter &p) const {
  p << "param.expr." << stringifyPEO(getOpcode()) << '<';
  llvm::interleaveComma(getOperands(), p.getStream(),
                        [&](Attribute op) { p.printAttributeWithoutType(op); });
  p << '>';
}
