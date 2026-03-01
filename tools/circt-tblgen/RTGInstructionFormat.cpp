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
using namespace circt::tblgen;

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

// Find an operand by name in the operation's arguments
static const NamedTypeConstraint *findOperand(const Operator &op,
                                              StringRef operandName) {
  for (auto arg : op.getArgs()) {
    if (auto *operand = dyn_cast<NamedTypeConstraint *>(arg)) {
      if (operand->name == operandName)
        return operand;
    }
  }
  return nullptr;
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
  if (!tryConsume(str, skipWhitespaceFirst)) {
    std::string errorMsg = "expected '" + str.str() + "'";
    PrintFatalError(loc, errorMsg);
  }
}

bool FormatParser::tryConsume(StringRef str, bool skipWhitespaceFirst) {
  if (skipWhitespaceFirst)
    skipWhitespace();

  if (input.substr(pos).starts_with(str)) {
    pos += str.size();
    return true;
  }

  return false;
}

int64_t FormatParser::parseIntLiteral() {
  skipWhitespace();

  size_t start = pos;
  auto loc = getLoc();

  // Check for optional negative sign
  if (pos < input.size() && input[pos] == '-')
    ++pos;

  size_t digitStart = pos;
  while (pos < input.size() && isdigit(input[pos]))
    ++pos;

  if (pos == digitStart) {
    std::string errorMsg = "expected integer literal";
    PrintFatalError(loc, errorMsg);
  }

  StringRef numStr = input.substr(start, pos - start);
  int64_t value;
  if (numStr.getAsInteger(0, value)) {
    std::string errorMsg = "invalid integer literal '" + numStr.str() + "'";
    PrintFatalError(loc, errorMsg);
  }

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
  if (endPos == StringRef::npos) {
    std::string errorMsg = "unterminated backtick literal";
    PrintFatalError(loc, errorMsg);
  }

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
  if (pos < input.size() && input[pos] != ' ') {
    std::string errorMsg = "invalid character ('" +
                           input.substr(pos, pos + 1).str() +
                           "') after binary literal";
    PrintFatalError(getLoc(), errorMsg);
  }

  APInt value(pos - digitStart, input.substr(digitStart, pos - digitStart), 2);
  return ctx.create<BinaryLiteralNode>(loc, value);
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
  if (!operand) {
    std::string errorMsg =
        "expected operand after " + specifierName.str() + " specifier";
    PrintFatalError(getLoc(), errorMsg);
  }
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

  highBit = parseIntLiteral();
  lowBit = tryConsume(":") ? parseIntLiteral() : highBit;

  consume("]");

  if (highBit < 0 || lowBit < 0) {
    PrintFatalError(loc, "slice bits must be non-negative");
  }

  if (highBit < lowBit) {
    std::string errorMsg = "high bit (" + std::to_string(highBit) +
                           ") must be >= low bit (" + std::to_string(lowBit) +
                           ")";
    PrintFatalError(loc, errorMsg);
  }

  return true;
}

void FormatParser::parseAndAppendToContext(llvm::SMLoc loc, StringRef format) {
  // Reset parser state
  pos = 0;
  input = format;
  this->loc = loc;

  while (pos < input.size()) {
    FormatNode *node = parseOptionalMnemonic();
    if (!node)
      node = parseOptionalSignednessSpecifier();
    if (!node)
      node = parseOptionalBinaryLiteral();
    if (!node)
      node = parseOptionalStringLiteral();
    if (!node)
      node = parseOptionalOperand();
    if (!node)
      PrintFatalError(getLoc(),
                      "unexpected character '" + Twine(input[pos]) + "'");

    // Elements in the format are separated by a space.
    if (pos < input.size())
      consume(" ", false);
    skipWhitespace();

    ctx.addNode(node);
  }
}

//===----------------------------------------------------------------------===//
// Resolve Operands
//===----------------------------------------------------------------------===//

// Helper to classify a type and add it to the kinds set
// NOLINTNEXTLINE(misc-no-recursion)
static void classifyAndAddType(const llvm::Record &typeDef,
                               DenseSet<OperandType> &kinds) {
  StringRef typeName = typeDef.getName();

  if (typeName == "LabelType" || typeDef.isSubClassOf("LabelType")) {
    kinds.insert(OperandType::Label);
    return;
  }

  if (typeName == "ImmediateType" || typeName == "ImmediateOfWidth" ||
      typeDef.isSubClassOf("ImmediateType") ||
      typeDef.isSubClassOf("ImmediateOfWidth")) {
    kinds.insert(OperandType::Immediate);
    return;
  }

  if (typeDef.isSubClassOf("RegisterTypeBase")) {
    kinds.insert(OperandType::Register);
    return;
  }

  if (typeName == "AnyTypeOf" || typeDef.isSubClassOf("AnyTypeOf")) {
    auto allowedTypes = typeDef.getValueAsListOfDefs("allowedTypes");
    for (const llvm::Record *allowedType : allowedTypes)
      classifyAndAddType(*allowedType, kinds);
  }
}

// Resolve the operand types and fill in the 'kinds' field of the
// OperandNode.
static void resolveOperand(ASTContext &ctx, OperandNode *operandNode) {
  const NamedTypeConstraint *namedOperand =
      findOperand(ctx.getOp(), operandNode->getName());
  if (!namedOperand)
    PrintFatalError(operandNode->getLoc(),
                    "unknown operand '$" + operandNode->getName() + "'");

  // Populate the kinds field (only if not already populated)
  if (!operandNode->kinds.empty())
    return;

  const llvm::Record &typeDef = namedOperand->constraint.getDef();
  classifyAndAddType(typeDef, operandNode->kinds);

  if (operandNode->kinds.empty())
    PrintFatalError(operandNode->getLoc(),
                    "failed to classify ISA type for operand '" +
                        operandNode->getName() + "' with type '" +
                        typeDef.getName() + "'");

  if (operandNode->kinds.contains(OperandType::Register) &&
      typeDef.isSubClassOf("RegisterTypeBase"))
    operandNode->registerBinaryEncodingWidth =
        typeDef.getValueAsInt("binaryEncodingWidth");
}

//===----------------------------------------------------------------------===//
// Binary Format Verification
//===----------------------------------------------------------------------===//

void circt::tblgen::verifyBinaryFormat(ASTContext &ctx) {
  for (auto *node : ctx.nodes()) {
    if (isa<MnemonicNode, StringLiteralNode, SignednessNode>(node))
      PrintFatalError(node->getLoc(),
                      "binary format cannot contain " + node->getDesc());
    if (auto *operandNode = dyn_cast<OperandNode>(node))
      resolveOperand(ctx, operandNode);
  }
}

//===----------------------------------------------------------------------===//
// Assembly Format Verification
//===----------------------------------------------------------------------===//

static void verifyAssemblyFormat(ASTContext &ctx, BinaryLiteralNode *node) {
  PrintFatalError(node->getLoc(),
                  "assembly format cannot contain binary literals");
}

static void verifyAssemblyFormat(ASTContext &ctx, OperandNode *node) {
  resolveOperand(ctx, node);

  if (node->kinds.contains(OperandType::Immediate)) {
    if (!isa_and_nonnull<SignednessNode>(node->parent))
      PrintFatalError(node->getLoc(),
                      "immediate operand '$" + node->getName() +
                          "' must be wrapped in signed() or unsigned()");
  }
}

static void verifyAssemblyFormat(ASTContext &ctx, SignednessNode *node) {
  OperandNode *operandNode = node->getOperand();
  resolveOperand(ctx, operandNode);

  // Check that wrapped operands are immediates
  if (operandNode->kinds.size() == 1) {
    OperandType kind = *operandNode->kinds.begin();
    if (kind == OperandType::Label) {
      std::string errorMsg = "label operand '$" + operandNode->getName().str() +
                             "' cannot be wrapped in signed() or unsigned()";
      PrintFatalError(operandNode->getLoc(), errorMsg);
    } else if (kind == OperandType::Register) {
      std::string errorMsg = "register operand '$" +
                             operandNode->getName().str() +
                             "' cannot be wrapped in signed() or unsigned()";
      PrintFatalError(operandNode->getLoc(), errorMsg);
    }
  }
}

void circt::tblgen::verifyAssemblyFormat(ASTContext &ctx) {
  for (auto *node : ctx.nodes()) {
    TypeSwitch<FormatNode *>(node)
        .Case<BinaryLiteralNode, OperandNode, SignednessNode>(
            [&](auto *node) { ::verifyAssemblyFormat(ctx, node); })
        .Case<StringLiteralNode, MnemonicNode>(
            [&](auto *node) { /*Nothing to do*/ })
        .Default([](auto *node) {
          PrintFatalError(node->getLoc(), "unhandled format node");
        });
  }
}

//===----------------------------------------------------------------------===//
// BinaryFormatGen Implementation
//===----------------------------------------------------------------------===//

static void genUnsupportedInstructionBinaryMethod(llvm::raw_ostream &os,
                                                  StringRef opClassName) {
  os << "void " << opClassName
     << "::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) "
        "{\n";
  os << "  assert(false && \"binary not supported\");\n";
  os << "}\n\n";
}

void BinaryFormatGen::gen(BinaryLiteralNode *node) {
  SmallVector<char> str;
  node->getValue().toStringUnsigned(str);
  os << "  binary = binary.concat(llvm::APInt(" << node->getWidth() << ", "
     << str << "));\n";
}

void BinaryFormatGen::genDecl(OperandNode *node) {
  std::string getterName = getGetterName(node->getName());

  if (node->kinds.size() > 2)
    PrintFatalError(
        node->getLoc(),
        "more than 2 operand types per operand not supported in binary format");

  if (node->kinds.size() == 2 && !node->kinds.contains(OperandType::Label))
    PrintFatalError(node->getLoc(),
                    "one of the operand types must be a label in binary format "
                    "if there are 2 operand types possible");

  if (node->kinds.contains(OperandType::Label))
    os << "  assert(!isa<rtg::LabelAttr>(adaptor." << getterName
       << "()) && \"labels not supported in binary format\");\n";

  if (decls.insert(node->getName()).second) {
    os << "  llvm::APInt " << node->getName();
    if (node->kinds.contains(OperandType::Register))
      os << " = llvm::APInt(" << node->registerBinaryEncodingWidth
         << ", cast<rtg::RegisterAttrInterface>(adaptor." + getterName +
                "()).getClassIndex());\n";
    else if (node->kinds.contains(OperandType::Immediate))
      os << " = cast<rtg::ImmediateAttr>(adaptor." + getterName +
                "()).getValue();\n";
    else
      PrintFatalError(node->getLoc(),
                      "unsupported operand type in binary format");
  }
}

void BinaryFormatGen::gen(OperandNode *node) {
  assert(decls.contains(node->getName()) &&
         "must have been declared in genDecl");

  if (node->hasBitSlice()) {
    os << "  binary = binary.concat(" << node->getName() << ".extractBits("
       << node->getBitWidth() << ", " << node->getLowBit() << "));\n";
    return;
  }

  os << "  binary = binary.concat(" << node->getName() << ");\n";
}

static bool hasOperandThatIsLabelOnly(ASTContext &ctx) {
  return llvm::any_of(ctx.nodes(), [](const FormatNode *node) {
    if (auto *operandNode = dyn_cast<OperandNode>(node)) {
      return operandNode->kinds.size() == 1 &&
             operandNode->kinds.contains(OperandType::Label);
    }
    return false;
  });
}

void BinaryFormatGen::genInstructionMethod(ASTContext &ctx,
                                           StringRef opClassName) {
  // Make sure the format is valid for binary method generation.
  verifyBinaryFormat(ctx);

  // Label operands cannot be supported in binary format.
  if (hasOperandThatIsLabelOnly(ctx))
    return genUnsupportedInstructionBinaryMethod(os, opClassName);

  os << "void " << opClassName
     << "::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) "
        "{\n";

  for (auto *node : ctx.nodes()) {
    if (auto *operandNode = dyn_cast<OperandNode>(node))
      genDecl(operandNode);
  }

  os << "\n";
  os << "  llvm::APInt binary = llvm::APInt::getZero(0);\n";

  for (auto *node : ctx.nodes()) {
    TypeSwitch<FormatNode *>(node)
        .Case<BinaryLiteralNode, OperandNode>([&](auto *node) { gen(node); })
        .Default([](auto *node) {
          PrintFatalError(node->getLoc(), "unhandled format node");
        });
  }

  os << "\n";
  os << "  llvm::SmallVector<char> str;\n";
  os << "  binary.toStringUnsigned(str, 16);\n";
  os << "  os << str;\n";
  os << "}\n\n";
}

//===----------------------------------------------------------------------===//
// AssemblyFormatGen Implementation
//===----------------------------------------------------------------------===//

/// Helper to print a label operand.
static void printLabel(raw_ostream &os, StringRef getterName,
                       bool needsDynCast) {
  if (needsDynCast) {
    os << "if (auto label = dyn_cast<rtg::LabelAttr>(adaptor." << getterName
       << "())) {\n";
    os << "    os << label.getName();\n";
  } else {
    os << "os << cast<rtg::LabelAttr>(adaptor." << getterName
       << "()).getName();\n";
  }
}

/// Helper to print a register operand.
static void printRegister(raw_ostream &os, StringRef getterName,
                          bool needsDynCast) {
  if (needsDynCast) {
    os << "if (auto reg = dyn_cast<rtg::RegisterAttrInterface>(adaptor."
       << getterName << "())) {\n";
    os << "    os << reg.getRegisterAssembly();\n";
  } else {
    os << "os << cast<rtg::RegisterAttrInterface>(adaptor." << getterName
       << "()).getRegisterAssembly();\n";
  }
}

/// Helper to print an immediate operand with signedness.
static void printImmediate(raw_ostream &os, StringRef getterName, bool isSigned,
                           bool needsDynCast) {
  if (needsDynCast) {
    os << "if (auto imm = dyn_cast<rtg::ImmediateAttr>(adaptor." << getterName
       << "())) {\n";
    os << "    {\n";
    os << "      SmallVector<char> strBuf;\n";
    os << "      imm.getValue().toString" << (isSigned ? "Signed" : "Unsigned")
       << "(strBuf);\n";
    os << "      os << strBuf;\n";
    os << "    }\n";
  } else {
    os << "{\n";
    os << "    SmallVector<char> strBuf;\n";
    os << "    cast<rtg::ImmediateAttr>(adaptor." << getterName
       << "()).getValue().toString" << (isSigned ? "Signed" : "Unsigned")
       << "(strBuf);\n";
    os << "    os << strBuf;\n";
    os << "  }\n";
  }
}

void AssemblyFormatGen::gen(StringLiteralNode *node) {
  os << "  os << \"" << node->getLiteral() << "\";\n";
}

void AssemblyFormatGen::gen(MnemonicNode *node) {
  os << "  os << getOperationName().rsplit('.').second;\n";
}

void AssemblyFormatGen::gen(SignednessNode *node) {
  OperandNode *operandNode = node->getOperand();
  gen(operandNode);
}

void AssemblyFormatGen::gen(OperandNode *node) {
  std::string getterName = getGetterName(node->getName());

  bool needsDynCast = node->kinds.size() > 1;
  bool isFirst = true;

  os << "  ";

  if (node->kinds.contains(OperandType::Label)) {
    if (needsDynCast && !isFirst)
      os << "  } else ";
    printLabel(os, getterName, needsDynCast);
    isFirst = false;
  }
  if (node->kinds.contains(OperandType::Register)) {
    if (needsDynCast && !isFirst)
      os << "  } else ";
    printRegister(os, getterName, needsDynCast);
    isFirst = false;
  }
  if (node->kinds.contains(OperandType::Immediate)) {
    assert(node->parent != nullptr &&
           "verifiers must make sure immediate operands are wrapped in "
           "signedness node");
    bool isSigned = cast<SignednessNode>(node->parent)->isSigned();
    if (needsDynCast && !isFirst)
      os << "  } else ";
    printImmediate(os, getterName, isSigned, needsDynCast);
    isFirst = false;
  }

  if (needsDynCast)
    os << "  }\n";
}

void AssemblyFormatGen::genInstructionMethod(ASTContext &ctx,
                                             StringRef opClassName) {
  // Make sure the format is valid for assembly method generation.
  verifyAssemblyFormat(ctx);

  os << "void " << opClassName
     << "::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor "
        "adaptor) {\n";

  for (auto *node : ctx.nodes()) {
    TypeSwitch<FormatNode *>(node)
        .Case<MnemonicNode, StringLiteralNode, SignednessNode, OperandNode>(
            [&](auto *node) { gen(node); });
  }

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

void circt::tblgen::genInstructionPrintMethods(const llvm::Record *opDef,
                                               raw_ostream &os) {
  Operator op(*opDef);
  auto *binaryFormatVal = opDef->getValue("isaBinaryFormat");
  auto *assemblyFormatVal = opDef->getValue("isaAssemblyFormat");
  StringRef binaryFormat = opDef->getValueAsString("isaBinaryFormat");
  StringRef assemblyFormat = opDef->getValueAsString("isaAssemblyFormat");

  if (assemblyFormat.empty())
    PrintFatalError(assemblyFormatVal->getLoc(),
                    "ISA assembly format must not be empty");

  auto loc = llvm::SMLoc::getFromPointer(
      assemblyFormatVal->getLoc().getPointer() + 21);
  AssemblyFormatGen assemblyGen(os);
  genInstructionMethod(assemblyGen, loc, op, assemblyFormat);

  if (binaryFormat.empty()) {
    genUnsupportedInstructionBinaryMethod(os, op.getCppClassName());
    return;
  }

  loc =
      llvm::SMLoc::getFromPointer(binaryFormatVal->getLoc().getPointer() + 19);
  BinaryFormatGen binaryGen(os);
  genInstructionMethod(binaryGen, loc, op, binaryFormat);
}
