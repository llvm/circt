//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RTGInstructionFormat.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using namespace circt::tblgen::rtg;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Convert operand name to getter method name (e.g., "rd" -> "getRd")
static std::string getGetterName(StringRef operandName) {
  std::string getterName = "get";
  if (!operandName.empty()) {
    getterName += llvm::toUpper(operandName[0]);
    getterName += operandName.substr(1);
  }
  return getterName;
}

//===----------------------------------------------------------------------===//
// FormatParser Implementation
//===----------------------------------------------------------------------===//

SMLoc FormatParser::getLoc() {
  return SMLoc::getFromPointer(loc.getPointer() + pos);
}

void FormatParser::skipWhitespace() {
  while (pos < input.size() && (input[pos] == ' ' || input[pos] == '\t'))
    ++pos;
}

void FormatParser::consume(StringRef str, bool skipWhitespaceFirst) {
  auto loc = getLoc();
  if (!tryConsume(str, skipWhitespaceFirst))
    PrintFatalError(loc, "expected '" + str + "'");
}

bool FormatParser::tryConsume(StringRef str, bool skipWhitespaceFirst) {
  auto savedPos = pos;

  // In case the user tries to consume whitespace.
  if (input.substr(pos).starts_with(str)) {
    pos += str.size();
    return true;
  }

  if (skipWhitespaceFirst)
    skipWhitespace();

  if (input.substr(pos).starts_with(str)) {
    pos += str.size();
    return true;
  }

  // Do not advance the position if we were not able to consume the string.
  pos = savedPos;
  return false;
}

bool FormatParser::tryParseUIntLiteral(uint64_t &value) {
  size_t start = pos;
  size_t savedPos = pos;

  while (pos < input.size() && isdigit(input[pos]))
    ++pos;

  if (pos == start) {
    // No digits found, restore position
    pos = savedPos;
    return false;
  }

  StringRef numStr = input.substr(start, pos - start);
  if (numStr.getAsInteger(10, value)) {
    // Invalid integer, restore position
    pos = savedPos;
    return false;
  }

  return true;
}

uint64_t FormatParser::parseUIntLiteral() {
  auto loc = getLoc();
  uint64_t value;

  if (!tryParseUIntLiteral(value))
    PrintFatalError(loc, "expected unsigned integer literal");

  return value;
}

StringRef FormatParser::parseIdentifier() {
  size_t nameEnd = pos;
  while (nameEnd < input.size() &&
         (isalnum(input[nameEnd]) || input[nameEnd] == '_'))
    ++nameEnd;

  StringRef name = input.substr(pos, nameEnd - pos);
  pos = nameEnd;

  return name;
}

FormatNode *FormatParser::parseOptionalMnemonic() {
  auto loc = getLoc();
  return tryConsume("mnemonic") ? ctx.create<MnemonicNode>(loc) : nullptr;
}

FormatNode *FormatParser::parseOptionalOperand() {
  auto loc = getLoc();
  if (!tryConsume("$"))
    return nullptr;

  StringRef name = parseIdentifier();

  int64_t highBit = -1;
  int64_t lowBit = -1;
  parseOptionalSlice(highBit, lowBit);

  return ctx.create<OperandNode>(loc, name, highBit, lowBit);
}

FormatNode *FormatParser::parseOptionalStringLiteral() {
  auto loc = getLoc();
  if (!tryConsume("`"))
    return nullptr;

  size_t endPos = input.find('`', pos);
  if (endPos == StringRef::npos)
    PrintFatalError(loc, "unterminated backtick literal");

  StringRef literal = input.substr(pos, endPos - pos);
  pos = endPos + 1; // Skip closing '`'

  return ctx.create<StringLiteralNode>(loc, literal);
}

FormatNode *FormatParser::parseOptionalBinaryLiteral() {
  auto loc = getLoc();
  if (!tryConsume("0b"))
    return nullptr;

  size_t digitStart = pos;
  while (pos < input.size() && (input[pos] == '0' || input[pos] == '1'))
    ++pos;

  if (pos == digitStart)
    return nullptr;

  // Check for invalid characters after the binary digits
  if (!isAtScopeEnd() && input[pos] != ' ')
    PrintFatalError(getLoc(), "invalid character ('" +
                                  input.substr(pos, pos + 1) +
                                  "') after binary literal");

  APInt value(pos - digitStart, input.substr(digitStart, pos - digitStart), 2);
  return ctx.create<BinaryLiteralNode>(loc, value);
}

FormatNode *FormatParser::parseOptionalDecimalLiteral() {
  auto loc = getLoc();

  // Try to parse the bit width (decimal number before 'd')
  uint64_t width;
  if (!tryParseUIntLiteral(width))
    return nullptr;

  if (!tryConsume("d", false))
    return nullptr;

  uint64_t value = parseUIntLiteral();

  if (!llvm::isUIntN(width, value))
    PrintFatalError(loc, "decimal literal value " + Twine(value) +
                             " does not fit in " + Twine(width) + " bits");

  return ctx.create<BinaryLiteralNode>(loc, APInt(width, value));
}

FormatNode *FormatParser::parseOptionalSignednessSpecifier() {
  auto loc = getLoc();
  bool isSigned = false;
  StringRef specifierName;

  if (tryConsume("signed(", false)) {
    isSigned = true;
    specifierName = "'signed'";
  } else if (tryConsume("unsigned(", false)) {
    isSigned = false;
    specifierName = "'unsigned'";
  } else {
    return nullptr;
  }

  FormatNode *operand = parseOptionalOperand();
  if (!operand)
    PrintFatalError(getLoc(),
                    "expected operand after " + specifierName + " specifier");

  auto *operandNode = cast<OperandNode>(operand);

  consume(")");

  auto *res = ctx.create<SignednessNode>(loc, isSigned, operandNode);
  operandNode->parent = res;
  return res;
}

bool FormatParser::parseOptionalSlice(int64_t &highBit, int64_t &lowBit) {
  auto loc = getLoc();
  if (!tryConsume("[", false))
    return false;

  highBit = parseUIntLiteral();
  lowBit = tryConsume(":") ? parseUIntLiteral() : highBit;

  consume("]");

  if (highBit < lowBit)
    PrintFatalError(loc, "high bit (" + Twine(highBit) +
                             ") must be >= low bit (" + Twine(lowBit) + ")");

  return true;
}

FormatNode *FormatParser::parseNextNode() {
  FormatNode *node = parseOptionalMnemonic();
  if (!node)
    node = parseOptionalSignednessSpecifier();
  if (!node)
    node = parseOptionalIfThenElse();
  if (!node)
    node = parseOptionalBinaryLiteral();
  if (!node)
    node = parseOptionalDecimalLiteral();
  if (!node)
    node = parseOptionalStringLiteral();
  if (!node)
    node = parseOptionalOperand();
  if (!node)
    PrintFatalError(getLoc(),
                    "unexpected character '" + Twine(input[pos]) + "'");

  // Elements in the format are separated by a space.
  if (!isAtScopeEnd())
    consume(" ");
  skipWhitespace();

  return node;
}

FormatNode *FormatParser::parseOptionalIfThenElse() {
  auto loc = getLoc();
  if (!tryConsume("if"))
    return nullptr;

  consume(" ");

  // Parse the condition operand (must be a 1-bit immediate)
  FormatNode *condNode = parseOptionalOperand();
  if (!condNode)
    PrintFatalError(getLoc(), "expected operand after 'if' keyword");

  auto *condition = cast<OperandNode>(condNode);

  consume(" ");

  SmallVector<FormatNode *> thenBranch;
  bool hasThen = tryConsume("then");
  if (hasThen) {
    consume("(");

    while (!tryConsume(")"))
      thenBranch.push_back(parseNextNode());
  }

  SmallVector<FormatNode *> elseBranch;
  bool hasElse = tryConsume("else");
  if (hasElse) {
    consume("(");

    while (!tryConsume(")"))
      elseBranch.push_back(parseNextNode());
  }

  if (!hasThen && !hasElse)
    PrintFatalError(getLoc(), "expected 'then' or 'else' after 'if' condition");

  auto *node = ctx.create<IfThenElseNode>(loc, condition, std::move(thenBranch),
                                          std::move(elseBranch));
  condition->parent = node;
  return node;
}

void FormatParser::parseAndAppendToContext(llvm::SMLoc loc, StringRef format) {
  // Reset parser state
  pos = 0;
  input = format;
  this->loc = loc;

  while (pos < input.size()) {
    auto *node = parseNextNode();
    ctx.addNode(node);
  }
}

bool FormatParser::isAtScopeEnd() {
  return pos >= input.size() || input[pos] == ')';
}

//===----------------------------------------------------------------------===//
// Resolve Operands
//===----------------------------------------------------------------------===//

// Resolve the operand types and fill in the 'kinds' field of the
// OperandNode.
static void resolveOperand(ASTContext &ctx, OperandNode *operandNode) {
  const NamedTypeConstraint *namedOperand =
      findOperandByName(ctx.getOp(), operandNode->getName());
  if (!namedOperand)
    PrintFatalError(operandNode->getLoc(),
                    "unknown operand '$" + operandNode->getName() + "'");

  // Populate the kinds field (only if not already populated)
  if (!operandNode->kinds.empty())
    return;

  const llvm::Record &typeDef = namedOperand->constraint.getDef();
  classifyOperandType(typeDef, operandNode->kinds);

  if (operandNode->kinds.empty())
    PrintFatalError(operandNode->getLoc(),
                    "failed to classify ISA type for operand '" +
                        operandNode->getName() + "' with type '" +
                        typeDef.getName() + "'");
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

void Verifier::verify(MnemonicNode *node) {
  if (isBinary)
    PrintFatalError(node->getLoc(),
                    "binary format cannot contain " + node->getDesc());
}

void Verifier::verify(StringLiteralNode *node) {
  if (isBinary)
    PrintFatalError(node->getLoc(),
                    "binary format cannot contain " + node->getDesc());
}

void Verifier::verify(IfThenElseNode *node) {
  if (node->isThenBranchEmpty() && node->isElseBranchEmpty())
    PrintFatalError(
        node->getLoc(),
        "if-then-else must have at least one of 'then' or 'else' branch");

  OperandNode *condition = node->getCondition();
  verify(condition);

  if (condition->kinds.size() != 1 || !condition->kinds.contains<Immediate>() ||
      std::get<Immediate>(condition->kinds[0]).bitWidth != 1)
    PrintFatalError(condition->getLoc(), "if-then-else condition operand '$" +
                                             condition->getName() +
                                             "' must be a 1-bit immediate");

  for (auto *child : node->getThenBranch())
    dispatch(child);
  for (auto *child : node->getElseBranch())
    dispatch(child);
}

void Verifier::verify(BinaryLiteralNode *node) {
  if (isBinary)
    return;

  PrintFatalError(node->getLoc(),
                  "assembly format cannot contain binary literals");
}

void Verifier::verify(OperandNode *node) {
  resolveOperand(ctx, node);

  if (!isBinary && node->kinds.contains<Immediate, AnyImmediate>()) {
    if (!isa_and_nonnull<SignednessNode>(node->parent) &&
        !isa_and_nonnull<IfThenElseNode>(node->parent))
      PrintFatalError(node->getLoc(),
                      "immediate operand '$" + node->getName() +
                          "' must be wrapped in signed() or unsigned()");
  }
}

void Verifier::verify(SignednessNode *node) {
  if (isBinary)
    PrintFatalError(node->getLoc(),
                    "binary format cannot contain " + node->getDesc());

  OperandNode *operandNode = node->getOperand();
  verify(operandNode);

  // Check that wrapped operands are immediates
  if (operandNode->kinds.size() == 1) {
    OperandType kind = *operandNode->kinds.begin();
    if (std::holds_alternative<Label>(kind))
      PrintFatalError(operandNode->getLoc(),
                      "label operand '$" + operandNode->getName() +
                          "' cannot be wrapped in signed() or unsigned()");

    if (std::holds_alternative<Register>(kind))
      PrintFatalError(operandNode->getLoc(),
                      "register operand '$" + operandNode->getName() +
                          "' cannot be wrapped in signed() or unsigned()");
  }
}

void Verifier::dispatch(FormatNode *node) {
  TypeSwitch<FormatNode *>(node)
      .Case<StringLiteralNode, MnemonicNode, BinaryLiteralNode, OperandNode,
            SignednessNode, IfThenElseNode>([&](auto *node) { verify(node); })
      .Default([](auto *node) {
        PrintFatalError(node->getLoc(), "unhandled format node");
      });
}

void Verifier::verifyAsBinary(ASTContext &ctx) {
  Verifier verifier(ctx, true);

  for (auto *node : ctx.nodes())
    verifier.dispatch(node);
}

void Verifier::verifyAsAssembly(ASTContext &ctx) {
  Verifier verifier(ctx, false);

  for (auto *node : ctx.nodes())
    verifier.dispatch(node);
}

//===----------------------------------------------------------------------===//
// BinaryFormatGen Implementation
//===----------------------------------------------------------------------===//

static void
genIfThenElse(IfThenElseNode *node, const DenseSet<StringRef> &decls,
              const std::function<void(ArrayRef<FormatNode *>)> &callback,
              mlir::raw_indented_ostream &os) {
  auto name = node->getCondition()->getName();
  if (!decls.contains(name))
    os << "::llvm::APInt " << name
       << " = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor."
       << getGetterName(name) << "()).getValue();\n";

  os << "if (" << name;
  if (!node->isThenBranchEmpty() && !node->isElseBranchEmpty()) {
    os << ".isOne()) {\n";
    os.indent();
    callback(node->getThenBranch());
    os.unindent();
    os << "} else {\n";
    os.indent();
    callback(node->getElseBranch());
    os.unindent();
  } else if (!node->isThenBranchEmpty()) {
    os << ".isOne()) {\n";
    os.indent();
    callback(node->getThenBranch());
    os.unindent();
  } else if (!node->isElseBranchEmpty()) {
    os << ".isZero()) {\n";
    os.indent();
    callback(node->getElseBranch());
    os.unindent();
  }
  os << "}\n";
}

static void
genUnsupportedInstructionBinaryMethod(mlir::raw_indented_ostream &os,
                                      StringRef opClassName) {
  os << "void " << opClassName
     << "::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) "
        "{\n";
  os.indent();
  os << "assert(false && \"binary not supported\");\n";
  os.unindent();
  os << "}\n\n";
}

void BinaryFormatGen::gen(BinaryLiteralNode *node) {
  SmallVector<char> str;
  node->getValue().toStringUnsigned(str);
  os << "binary = binary.concat(llvm::APInt(" << node->getWidth() << ", " << str
     << "));\n";
}

void BinaryFormatGen::genDecl(OperandNode *node) {
  std::string getterName = getGetterName(node->getName());
  assert(!node->kinds.empty() && "must have been resolved");

  if (node->kinds.size() > 2)
    PrintFatalError(
        node->getLoc(),
        "more than 2 operand types per operand not supported in binary format");

  if (node->kinds.size() == 2 && !node->kinds.contains<Label>())
    PrintFatalError(node->getLoc(),
                    "one of the operand types must be a label in binary format "
                    "if there are 2 operand types possible");

  if (node->kinds.contains<Label>())
    os << "assert(!::llvm::isa<::circt::rtg::LabelAttr>(adaptor." << getterName
       << "()) && \"labels not supported in binary format\");\n";

  if (decls.insert(node->getName()).second) {
    os << "::llvm::APInt " << node->getName();
    if (node->kinds.contains<Register>())
      os << " = ::llvm::APInt("
         << std::get<Register>(node->kinds[0]).binaryEncodingWidth
         << ", ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor."
         << getterName << "()).getClassIndex());\n";
    else if (node->kinds.contains<Immediate, AnyImmediate>())
      os << " = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor."
         << getterName << "()).getValue();\n";
    else
      PrintFatalError(node->getLoc(),
                      "unsupported operand type in binary format");
  }
}

void BinaryFormatGen::genDecl(IfThenElseNode *node) {
  genDecl(node->getCondition());
  for (auto *child : node->getThenBranch())
    dispatchDecl(child);
  for (auto *child : node->getElseBranch())
    dispatchDecl(child);
}

void BinaryFormatGen::dispatchDecl(FormatNode *node) {
  TypeSwitch<FormatNode *>(node)
      .Case<OperandNode, IfThenElseNode>([&](auto *node) { genDecl(node); })
      .Default([](auto *node) { /*Nothing to do*/ });
}

void BinaryFormatGen::gen(OperandNode *node) {
  assert(decls.contains(node->getName()) &&
         "must have been declared in genDecl");

  if (node->hasBitSlice()) {
    os << "binary = binary.concat(" << node->getName() << ".extractBits("
       << node->getBitWidth() << ", " << node->getLowBit() << "));\n";
    return;
  }

  os << "binary = binary.concat(" << node->getName() << ");\n";
}

void BinaryFormatGen::gen(IfThenElseNode *node) {
  genIfThenElse(
      node, decls,
      [&](auto nodes) {
        for (auto *child : nodes)
          dispatch(child);
      },
      os);
}

void BinaryFormatGen::dispatch(FormatNode *node) {
  TypeSwitch<FormatNode *>(node)
      .Case<BinaryLiteralNode, OperandNode, IfThenElseNode>(
          [&](auto *node) { gen(node); })
      .Default([](auto *node) {
        PrintFatalError(node->getLoc(), "unhandled format node");
      });
}

static bool hasOperandThatIsLabelOnly(ASTContext &ctx) {
  return llvm::any_of(ctx.nodes(), [](const FormatNode *node) {
    if (auto *operandNode = dyn_cast<OperandNode>(node)) {
      return operandNode->kinds.size() == 1 &&
             operandNode->kinds.contains<Label>();
    }
    return false;
  });
}

void BinaryFormatGen::genInstructionMethod(ASTContext &ctx,
                                           StringRef opClassName) {
  // Make sure the format is valid for binary method generation.
  Verifier::verifyAsBinary(ctx);

  // Label operands cannot be supported in binary format.
  if (hasOperandThatIsLabelOnly(ctx))
    return genUnsupportedInstructionBinaryMethod(os, opClassName);

  os << "void " << opClassName
     << "::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) "
        "{\n";

  os.indent();

  for (auto *node : ctx.nodes())
    dispatchDecl(node);

  os << "\n";
  os << "::llvm::APInt binary = llvm::APInt::getZero(0);\n";

  for (auto *node : ctx.nodes())
    dispatch(node);

  os << "\n";
  os << "::llvm::SmallVector<char> str;\n";
  os << "binary.toStringUnsigned(str, 16);\n";
  os << "os << str;\n";

  os.unindent();
  os << "}\n\n";
}

//===----------------------------------------------------------------------===//
// AssemblyFormatGen Implementation
//===----------------------------------------------------------------------===//

/// Helper to print a label operand.
static void printLabel(mlir::raw_indented_ostream &os, StringRef getterName,
                       bool needsDynCast) {
  if (needsDynCast) {
    os << "if (auto label = ::llvm::dyn_cast<::circt::rtg::LabelAttr>(adaptor."
       << getterName << "())) {\n";
    os.indent();
    os << "os << label.getName();\n";
    os.unindent();
  } else {
    os << "os << ::llvm::cast<::circt::rtg::LabelAttr>(adaptor." << getterName
       << "()).getName();\n";
  }
}

/// Helper to print a register operand.
static void printRegister(mlir::raw_indented_ostream &os, StringRef getterName,
                          bool needsDynCast) {
  if (needsDynCast) {
    os << "if (auto reg = "
          "::llvm::dyn_cast<::circt::rtg::RegisterAttrInterface>(adaptor."
       << getterName << "())) {\n";
    os.indent();
    os << "os << reg.getRegisterAssembly();\n";
    os.unindent();
  } else {
    os << "os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor."
       << getterName << "()).getRegisterAssembly();\n";
  }
}

/// Helper to print an immediate operand with signedness.
static void printImmediate(mlir::raw_indented_ostream &os, StringRef getterName,
                           bool isSigned, bool needsDynCast) {
  if (needsDynCast) {
    os << "if (auto imm = "
          "::llvm::dyn_cast<::circt::rtg::ImmediateAttr>(adaptor."
       << getterName << "())) {\n";
    os.indent();
    os << "{\n";
    os.indent();
    os << "::llvm::SmallVector<char> strBuf;\n";
    os << "imm.getValue().toString" << (isSigned ? "Signed" : "Unsigned")
       << "(strBuf);\n";
    os << "os << strBuf;\n";
    os.unindent();
    os << "}\n";
    os.unindent();
  } else {
    os << "{\n";
    os.indent();
    os << "::llvm::SmallVector<char> strBuf;\n";
    os << "::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor." << getterName
       << "()).getValue().toString" << (isSigned ? "Signed" : "Unsigned")
       << "(strBuf);\n";
    os << "os << strBuf;\n";
    os.unindent();
    os << "}\n";
  }
}

void AssemblyFormatGen::gen(StringLiteralNode *node) {
  os << "os << \"" << node->getLiteral() << "\";\n";
}

void AssemblyFormatGen::gen(MnemonicNode *node) {
  os << "os << getOperationName().rsplit('.').second;\n";
}

void AssemblyFormatGen::gen(SignednessNode *node) {
  OperandNode *operandNode = node->getOperand();
  gen(operandNode);
}

void AssemblyFormatGen::gen(OperandNode *node) {
  std::string getterName = getGetterName(node->getName());

  bool needsDynCast = node->kinds.size() > 1;
  bool isFirst = true;

  if (node->kinds.contains<Label>()) {
    if (needsDynCast && !isFirst)
      os << "} else ";
    printLabel(os, getterName, needsDynCast);
    isFirst = false;
  }
  if (node->kinds.contains<Register>()) {
    if (needsDynCast && !isFirst)
      os << "} else ";
    printRegister(os, getterName, needsDynCast);
    isFirst = false;
  }
  if (node->kinds.contains<Immediate>()) {
    assert(node->parent != nullptr &&
           "verifiers must make sure immediate operands are wrapped in "
           "signedness node");
    bool isSigned = cast<SignednessNode>(node->parent)->isSigned();
    if (needsDynCast && !isFirst)
      os << "} else ";
    printImmediate(os, getterName, isSigned, needsDynCast);
    isFirst = false;
  }

  if (needsDynCast)
    os << "}\n";
}

void AssemblyFormatGen::gen(IfThenElseNode *node) {
  genIfThenElse(
      node, {},
      [&](auto nodes) {
        for (auto *child : nodes)
          dispatch(child);
      },
      os);
}

void AssemblyFormatGen::dispatch(FormatNode *node) {
  TypeSwitch<FormatNode *>(node)
      .Case<MnemonicNode, StringLiteralNode, SignednessNode, OperandNode,
            IfThenElseNode>([&](auto *node) { gen(node); });
}

void AssemblyFormatGen::genInstructionMethod(ASTContext &ctx,
                                             StringRef opClassName) {
  // Make sure the format is valid for assembly method generation.
  Verifier::verifyAsAssembly(ctx);

  os << "void " << opClassName
     << "::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor "
        "adaptor) {\n";

  os.indent();

  for (auto *node : ctx.nodes())
    dispatch(node);

  os.unindent();
  os << "}\n\n";
}

//===----------------------------------------------------------------------===//
// Code Generation Entry Point
//===----------------------------------------------------------------------===//

static void genInstructionMethod(FormatGen &gen, llvm::SMLoc loc, Operator &op,
                                 StringRef formatString) {
  StringRef opClassName = op.getCppClassName();
  ASTContext ctx(op);
  FormatParser parser(ctx);
  parser.parseAndAppendToContext(loc, formatString);
  gen.genInstructionMethod(ctx, opClassName);
}

void circt::tblgen::rtg::genInstructionPrintMethods(const llvm::Record *opDef,
                                                    raw_ostream &os) {
  Operator op(*opDef);
  auto *binaryFormatVal = opDef->getValue("isaBinaryFormat");
  auto *assemblyFormatVal = opDef->getValue("isaAssemblyFormat");
  StringRef binaryFormat = opDef->getValueAsString("isaBinaryFormat");
  StringRef assemblyFormat = opDef->getValueAsString("isaAssemblyFormat");

  mlir::raw_indented_ostream indentedOS(os);

  if (assemblyFormat.empty())
    PrintFatalError(assemblyFormatVal->getLoc(),
                    "ISA assembly format must not be empty");

  auto loc = llvm::SMLoc::getFromPointer(
      assemblyFormatVal->getLoc().getPointer() + 21);
  AssemblyFormatGen assemblyGen(indentedOS);
  genInstructionMethod(assemblyGen, loc, op, assemblyFormat);

  if (binaryFormat.empty()) {
    genUnsupportedInstructionBinaryMethod(indentedOS, op.getCppClassName());
    return;
  }

  loc =
      llvm::SMLoc::getFromPointer(binaryFormatVal->getLoc().getPointer() + 19);
  BinaryFormatGen binaryGen(indentedOS);
  genInstructionMethod(binaryGen, loc, op, binaryFormat);
}
