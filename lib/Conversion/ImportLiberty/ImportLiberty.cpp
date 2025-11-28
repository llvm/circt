//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Liberty file import functionality.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ImportLiberty.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "import-liberty"

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::comb;

namespace {

/// Liberty token types for lexical analysis
enum class LibertyTokenKind {
  // Literals
  Identifier,
  String,
  Number,

  // Punctuation
  LBrace, // {
  RBrace, // }
  LParen, // (
  RParen, // )
  Colon,  // :
  Semi,   // ;
  Comma,  // ,

  // Special
  EndOfFile,
  Error
};

StringRef stringifyTokenKind(LibertyTokenKind kind) {
  switch (kind) {
  case LibertyTokenKind::Identifier:
    return "identifier";
  case LibertyTokenKind::String:
    return "string";
  case LibertyTokenKind::Number:
    return "number";
  case LibertyTokenKind::LBrace:
    return "'{'";
  case LibertyTokenKind::RBrace:
    return "'}'";
  case LibertyTokenKind::LParen:
    return "'('";
  case LibertyTokenKind::RParen:
    return "')'";
  case LibertyTokenKind::Colon:
    return "':'";
  case LibertyTokenKind::Semi:
    return "';'";
  case LibertyTokenKind::Comma:
    return "','";

  case LibertyTokenKind::EndOfFile:
    return "end of file";
  case LibertyTokenKind::Error:
    return "error";
  }
  return "unknown";
}

struct LibertyToken {
  LibertyTokenKind kind;
  StringRef spelling;
  SMLoc location;

  LibertyToken(LibertyTokenKind kind, StringRef spelling, SMLoc location)
      : kind(kind), spelling(spelling), location(location) {}

  bool is(LibertyTokenKind k) const { return kind == k; }
};

class LibertyLexer {
public:
  LibertyLexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
      : sourceMgr(sourceMgr), context(context),
        curBuffer(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
        curPtr(curBuffer.begin()) {}

  LibertyToken nextToken();
  LibertyToken peekToken();

  SMLoc getCurrentLoc() const { return SMLoc::getFromPointer(curPtr); }

  Location translateLocation(llvm::SMLoc loc) const {
    unsigned mainFileID = sourceMgr.getMainFileID();
    auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
    return FileLineColLoc::get(
        StringAttr::get(
            context,
            sourceMgr.getMemoryBuffer(mainFileID)->getBufferIdentifier()),
        lineAndColumn.first, lineAndColumn.second);
  }

  bool isAtEnd() const { return curPtr >= curBuffer.end(); }

private:
  const llvm::SourceMgr &sourceMgr;
  MLIRContext *context;
  StringRef curBuffer;
  const char *curPtr;

  void skipWhitespaceAndComments();
  LibertyToken lexIdentifier();
  LibertyToken lexString();
  LibertyToken lexNumber();
  LibertyToken makeToken(LibertyTokenKind kind, const char *start);
};

// Helper class to parse boolean expressions from Liberty function attributes
class ExpressionParser {
public:
  ExpressionParser(LibertyLexer &lexer, OpBuilder &builder, StringRef expr,
                   const llvm::StringMap<Value> &values, SMLoc baseLoc)
      : lexer(lexer), builder(builder), values(values), baseLoc(baseLoc),
        exprStart(expr.begin()) {
    tokenize(expr);
  }

  ParseResult parse(Value &result) { return parseOrExpr(result); }

private:
  enum class TokenKind { ID, AND, OR, XOR, NOT, LPAREN, RPAREN, END };

  struct Token {
    TokenKind kind;
    StringRef spelling;
    SMLoc loc;
  };

  LibertyLexer &lexer;
  OpBuilder &builder;
  const llvm::StringMap<Value> &values;
  SMLoc baseLoc;
  const char *exprStart;
  SmallVector<Token> tokens;
  size_t pos = 0;

  SMLoc getLoc(const char *ptr) {
    size_t offset = ptr - exprStart;
    return SMLoc::getFromPointer(baseLoc.getPointer() + 1 + offset);
  }

  void tokenize(StringRef expr) {
    const char *ptr = expr.begin();
    const char *end = expr.end();

    while (ptr < end) {
      // Skip whitespace
      if (isspace(*ptr)) {
        ++ptr;
        continue;
      }

      // Calculate location relative to baseLoc
      // baseLoc points to the opening quote of the string attribute
      // ptr points into the StringAttr value
      // We assume the StringAttr value content matches the source content
      // (escapes etc.)
      SMLoc loc = getLoc(ptr);

      // Identifier
      if (isalnum(*ptr) || *ptr == '_') {
        const char *start = ptr;
        while (ptr < end && (isalnum(*ptr) || *ptr == '_'))
          ++ptr;
        tokens.push_back({TokenKind::ID, StringRef(start, ptr - start), loc});
        continue;
      }

      // Operators and punctuation
      switch (*ptr) {
      case '*':
      case '&':
        tokens.push_back({TokenKind::AND, StringRef(ptr, 1), loc});
        break;
      case '+':
      case '|':
        tokens.push_back({TokenKind::OR, StringRef(ptr, 1), loc});
        break;
      case '^':
        tokens.push_back({TokenKind::XOR, StringRef(ptr, 1), loc});
        break;
      case '!':
      case '\'':
        tokens.push_back({TokenKind::NOT, StringRef(ptr, 1), loc});
        break;
      case '(':
        tokens.push_back({TokenKind::LPAREN, StringRef(ptr, 1), loc});
        break;
      case ')':
        tokens.push_back({TokenKind::RPAREN, StringRef(ptr, 1), loc});
        break;
      }
      ++ptr;
    }

    tokens.push_back({TokenKind::END, "", getLoc(ptr)});
  }

  Token peek() const { return tokens[pos]; }
  Token consume() { return tokens[pos++]; }

  Value trueVal = nullptr;

  Value createNot(Value val, SMLoc loc) {
    if (!trueVal)
      trueVal = hw::ConstantOp::create(builder, lexer.translateLocation(loc),
                                       builder.getI1Type(), 1);
    return comb::XorOp::create(builder, lexer.translateLocation(loc), val,
                               trueVal);
  }

  // Parse: OrExpr -> XorExpr { ('+'|'|') XorExpr }
  ParseResult parseOrExpr(Value &result) {
    Value lhs;
    if (parseXorExpr(lhs))
      return failure();
    while (peek().kind == TokenKind::OR) {
      auto loc = consume().loc;
      Value rhs;
      if (parseXorExpr(rhs))
        return failure();
      lhs = comb::OrOp::create(builder, lexer.translateLocation(loc), lhs, rhs);
    }
    result = lhs;
    return success();
  }

  // Parse: XorExpr -> AndExpr { '^' AndExpr }
  ParseResult parseXorExpr(Value &result) {
    Value lhs;
    if (parseAndExpr(lhs))
      return failure();
    while (peek().kind == TokenKind::XOR) {
      auto loc = consume().loc;
      Value rhs;
      if (parseAndExpr(rhs))
        return failure();
      lhs =
          comb::XorOp::create(builder, lexer.translateLocation(loc), lhs, rhs);
    }
    result = lhs;
    return success();
  }

  // Parse: AndExpr -> UnaryExpr { ('*'|'&') UnaryExpr }
  ParseResult parseAndExpr(Value &result) {
    Value lhs;
    if (parseUnaryExpr(lhs))
      return failure();
    while (peek().kind == TokenKind::AND) {
      auto loc = consume().loc;
      Value rhs;
      if (parseUnaryExpr(rhs))
        return failure();
      lhs =
          comb::AndOp::create(builder, lexer.translateLocation(loc), lhs, rhs);
    }
    result = lhs;
    return success();
  }

  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) const {
    return mlir::emitError(lexer.translateLocation(loc), message);
  }

  InFlightDiagnostic emitWarning(llvm::SMLoc loc, const Twine &message) const {
    return mlir::emitWarning(lexer.translateLocation(loc), message);
  }

  // Parse: UnaryExpr -> ('!'|'\'') UnaryExpr | '(' OrExpr ')' ['\''] | ID
  // ['\'']
  ParseResult parseUnaryExpr(Value &result) {
    // Prefix NOT
    if (peek().kind == TokenKind::NOT) {
      auto loc = consume().loc;
      Value val;
      if (parseUnaryExpr(val))
        return failure();
      result = createNot(val, loc);
      return success();
    }

    // Parenthesized expression
    if (peek().kind == TokenKind::LPAREN) {
      consume();
      Value val;
      if (parseOrExpr(val))
        return failure();
      if (peek().kind != TokenKind::RPAREN)
        return failure();
      consume();
      // Postfix NOT
      if (peek().kind == TokenKind::NOT) {
        auto loc = consume().loc;
        val = createNot(val, loc);
      }
      result = val;
      return success();
    }

    // Identifier
    if (peek().kind == TokenKind::ID) {
      auto tok = consume();
      StringRef name = tok.spelling;
      auto it = values.find(name);
      if (it == values.end())
        return emitError(tok.loc, "variable not found");
      Value val = it->second;
      // Postfix NOT
      if (peek().kind == TokenKind::NOT) {
        auto loc = consume().loc;
        val = createNot(val, loc);
      }
      result = val;
      return success();
    }

    return emitError(peek().loc, "expected expression");
  }
};

// Parsed result of a Liberty group
struct LibertyGroup {
  StringRef name;
  SMLoc loc;
  SmallVector<Attribute> args;
  SmallVector<std::tuple<StringRef, Attribute, SMLoc>> attrs;
  SmallVector<std::unique_ptr<LibertyGroup>> subGroups;

  // Helper to find an attribute by name
  std::pair<Attribute, SMLoc> getAttribute(StringRef name) const {
    for (const auto &attr : attrs)
      if (std::get<0>(attr) == name)
        return {std::get<1>(attr), std::get<2>(attr)};
    return {};
  }

  void eraseSubGroup(llvm::function_ref<bool(const LibertyGroup &)> pred) {
    subGroups.erase(
        std::remove_if(subGroups.begin(), subGroups.end(),
                       [pred](const auto &group) { return pred(*group); }),
        subGroups.end());
  }
};

class LibertyParser {
public:
  LibertyParser(const llvm::SourceMgr &sourceMgr, MLIRContext *context,
                ModuleOp module)
      : lexer(sourceMgr, context), module(module),
        builder(module.getBodyRegion()) {}

  ParseResult parse();

private:
  LibertyLexer lexer;
  ModuleOp module;
  OpBuilder builder;

  // Specific group parsers
  ParseResult parseLibrary();
  ParseResult parseGroupBody(LibertyGroup &group);
  ParseResult parseStatement(LibertyGroup &parent);

  // Lowering methods
  ParseResult lowerCell(const LibertyGroup &group);

  //===--------------------------------------------------------------------===//
  // Parser for Subgroup of "cell" group.
  //===--------------------------------------------------------------------===//

  Attribute convertGroupToAttr(const LibertyGroup &group);

  // Attribute parsing
  ParseResult parseAttribute(Attribute &result);
  static ParseResult parseAttribute(Attribute &result, OpBuilder &builder,
                                    StringRef attr);

  // Expression parsing
  // Expression parsing
  ParseResult parseExpression(Value &result, StringRef expr,
                              const llvm::StringMap<Value> &values, SMLoc loc);

  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) const {
    return mlir::emitError(lexer.translateLocation(loc), message);
  }

  InFlightDiagnostic emitWarning(llvm::SMLoc loc, const Twine &message) const {
    return mlir::emitWarning(lexer.translateLocation(loc), message);
  }

  ParseResult consume(LibertyTokenKind kind, const Twine &msg) {
    if (lexer.nextToken().is(kind))
      return success();
    return emitError(lexer.getCurrentLoc(), msg);
  }

  ParseResult consumeUntil(LibertyTokenKind kind) {
    while (!lexer.peekToken().is(kind) &&
           lexer.peekToken().kind != LibertyTokenKind::EndOfFile)
      lexer.nextToken();
    if (lexer.peekToken().is(kind))
      return success();
    return emitError(lexer.getCurrentLoc(),
                     " expected " + stringifyTokenKind(kind));
  }

  ParseResult consume(LibertyTokenKind kind) {
    if (lexer.nextToken().is(kind))
      return success();
    return emitError(lexer.getCurrentLoc(),
                     " expected " + stringifyTokenKind(kind));
  }

  ParseResult expect(LibertyTokenKind kind) {
    if (!lexer.peekToken().is(kind))
      return emitError(lexer.getCurrentLoc(),
                       " expected " + stringifyTokenKind(kind));

    return success();
  }

  StringRef getTokenSpelling(const LibertyToken &token) {
    StringRef str = token.spelling;
    if (token.kind == LibertyTokenKind::String)
      str = str.drop_front().drop_back();
    return str;
  }
};

//===----------------------------------------------------------------------===//
// LibertyLexer Implementation
//===----------------------------------------------------------------------===//

void LibertyLexer::skipWhitespaceAndComments() {
  while (curPtr < curBuffer.end()) {
    if (isspace(*curPtr)) {
      ++curPtr;
      continue;
    }

    // Comments
    if (*curPtr == '/' && curPtr + 1 < curBuffer.end()) {
      if (*(curPtr + 1) == '*') { // /* ... */
        curPtr += 2;
        while (curPtr + 1 < curBuffer.end() &&
               (*curPtr != '*' || *(curPtr + 1) != '/'))
          ++curPtr;
        if (curPtr + 1 < curBuffer.end())
          curPtr += 2;
        continue;
      }
      if (*(curPtr + 1) == '/') { // // ...
        while (curPtr < curBuffer.end() && *curPtr != '\n')
          ++curPtr;
        continue;
      }
    }

    // Backslash newline continuation
    if (*curPtr == '\\' && curPtr + 1 < curBuffer.end() &&
        *(curPtr + 1) == '\n') {
      curPtr += 2;
      continue;
    }

    break;
  }
}

LibertyToken LibertyLexer::lexIdentifier() {
  const char *start = curPtr;
  while (curPtr < curBuffer.end() &&
         (isalnum(*curPtr) || *curPtr == '_' || *curPtr == '.'))
    ++curPtr;
  return makeToken(LibertyTokenKind::Identifier, start);
}

LibertyToken LibertyLexer::lexString() {
  const char *start = curPtr;
  ++curPtr; // skip opening quote
  while (curPtr < curBuffer.end() && *curPtr != '"') {
    if (*curPtr == '\\' && curPtr + 1 < curBuffer.end())
      ++curPtr; // skip escaped char
    ++curPtr;
  }
  if (curPtr < curBuffer.end())
    ++curPtr; // skip closing quote
  return makeToken(LibertyTokenKind::String, start);
}

LibertyToken LibertyLexer::lexNumber() {
  const char *start = curPtr;
  bool seenDot = false;
  while (curPtr < curBuffer.end()) {
    if (isdigit(*curPtr)) {
      ++curPtr;
    } else if (*curPtr == '.' && !seenDot) {
      seenDot = true;
      ++curPtr;
    } else {
      break;
    }
  }
  return makeToken(LibertyTokenKind::Number, start);
}

LibertyToken LibertyLexer::makeToken(LibertyTokenKind kind, const char *start) {
  return LibertyToken(kind, StringRef(start, curPtr - start),
                      SMLoc::getFromPointer(start));
}

LibertyToken LibertyLexer::nextToken() {
  skipWhitespaceAndComments();

  if (curPtr >= curBuffer.end())
    return makeToken(LibertyTokenKind::EndOfFile, curPtr);

  const char *start = curPtr;
  char c = *curPtr;

  if (isalpha(c) || c == '_')
    return lexIdentifier();

  if (isdigit(c) ||
      (c == '.' && curPtr + 1 < curBuffer.end() && isdigit(*(curPtr + 1))))
    return lexNumber();

  if (c == '"')
    return lexString();

  ++curPtr;
  switch (c) {
  case '{':
    return makeToken(LibertyTokenKind::LBrace, start);
  case '}':
    return makeToken(LibertyTokenKind::RBrace, start);
  case '(':
    return makeToken(LibertyTokenKind::LParen, start);
  case ')':
    return makeToken(LibertyTokenKind::RParen, start);
  case ':':
    return makeToken(LibertyTokenKind::Colon, start);
  case ';':
    return makeToken(LibertyTokenKind::Semi, start);
  case ',':
    return makeToken(LibertyTokenKind::Comma, start);

  default:
    return makeToken(LibertyTokenKind::Error, start);
  }
}

LibertyToken LibertyLexer::peekToken() {
  const char *savedPtr = curPtr;
  LibertyToken token = nextToken();
  curPtr = savedPtr;
  return token;
}

//===----------------------------------------------------------------------===//
// LibertyParser Implementation
//===----------------------------------------------------------------------===//

ParseResult LibertyParser::parse() {
  while (lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.peekToken();
    // Skip any stray tokens that aren't valid group starts
    if (token.kind != LibertyTokenKind::Identifier ||
        token.spelling != "library") {
      return emitError(token.location, "expected liberty");
    }
    lexer.nextToken();
    if (parseLibrary())
      return failure();
  }
  return success();
}

ParseResult LibertyParser::parseLibrary() {
  LibertyGroup libertyLib;
  if (parseGroupBody(libertyLib))
    return failure();

  for (auto &stmt : libertyLib.subGroups) {
    if (stmt->name == "cell") {
      if (lowerCell(*stmt))
        return failure();
      continue;
    }
  }

  libertyLib.eraseSubGroup(
      [](const LibertyGroup &group) { return group.name == "cell"; });
  auto attr = convertGroupToAttr(libertyLib);
  module->setAttr("synth.liberty.library", attr);
  return success();
}

// Parse a template like: lu_table_template (delay_template_6x6) { variable_1:
// total_output_net_capacitance; variable_2: input_net_transition; }
Attribute LibertyParser::convertGroupToAttr(const LibertyGroup &group) {
  SmallVector<NamedAttribute> attrs;
  if (!group.args.empty())
    attrs.push_back(
        builder.getNamedAttr("args", builder.getArrayAttr(group.args)));

  for (const auto &attr : group.attrs)
    attrs.push_back(builder.getNamedAttr(std::get<0>(attr), std::get<1>(attr)));

  llvm::StringMap<SmallVector<Attribute>> subGroups;
  for (const auto &sub : group.subGroups)
    subGroups[sub->name].push_back(convertGroupToAttr(*sub));

  for (auto &it : subGroups)
    attrs.push_back(
        builder.getNamedAttr(it.getKey(), builder.getArrayAttr(it.getValue())));

  return builder.getDictionaryAttr(attrs);
}

ParseResult LibertyParser::lowerCell(const LibertyGroup &group) {
  if (group.args.empty())
    return emitError(group.loc, "cell missing name");

  StringRef cellName;
  if (auto strAttr = dyn_cast<StringAttr>(group.args[0]))
    cellName = strAttr.getValue();
  else
    return emitError(group.loc, "cell name must be a string");

  SmallVector<hw::PortInfo> ports;
  SmallVector<const LibertyGroup *> pinGroups;

  // First pass: gather ports
  for (const auto &sub : group.subGroups) {
    if (sub->name != "pin")
      continue;

    pinGroups.push_back(sub.get());
    if (sub->args.empty())
      return emitError(sub->loc, "pin missing name");

    StringRef pinName;
    if (auto strAttr = dyn_cast<StringAttr>(sub->args[0]))
      pinName = strAttr.getValue();
    else
      return emitError(sub->loc, "pin name must be a string");

    std::optional<hw::ModulePort::Direction> dir;
    SmallVector<NamedAttribute> pinAttrs;

    for (const auto &attr : sub->attrs) {
      if (std::get<0>(attr) == "direction") {
        if (auto val = dyn_cast<StringAttr>(std::get<1>(attr))) {
          if (val.getValue() == "input")
            dir = hw::ModulePort::Direction::Input;
          else if (val.getValue() == "output")
            dir = hw::ModulePort::Direction::Output;
          else if (val.getValue() == "inout")
            dir = hw::ModulePort::Direction::InOut;
          else
            return emitError(sub->loc,
                             "pin direction must be input, output, or inout");
        }
        continue;
      }
      pinAttrs.push_back(
          builder.getNamedAttr(std::get<0>(attr), std::get<1>(attr)));
    }

    if (!dir)
      return emitError(sub->loc, "pin direction must be specified");

    llvm::StringMap<SmallVector<Attribute>> subGroups;
    for (const auto &child : sub->subGroups) {
      // TODO: Properly handle timing subgroups etc.
      subGroups[child->name].push_back(convertGroupToAttr(*child));
    }

    for (auto &it : subGroups)
      pinAttrs.push_back(builder.getNamedAttr(
          it.getKey(), builder.getArrayAttr(it.getValue())));

    auto libertyAttrs = builder.getDictionaryAttr(pinAttrs);
    auto attrs = builder.getDictionaryAttr(
        builder.getNamedAttr("synth.liberty.pin", libertyAttrs));

    hw::PortInfo port;
    port.name = builder.getStringAttr(pinName);
    port.type = builder.getI1Type();
    port.dir = *dir;
    port.attrs = attrs;
    ports.push_back(port);
  }

  // Fix up argNum for inputs
  int inputIdx = 0;
  for (auto &p : ports) {
    if (p.dir == hw::ModulePort::Direction::Input)
      p.argNum = inputIdx++;
    else
      p.argNum = 0;
  }

  auto loc = lexer.translateLocation(group.loc);
  auto moduleOp = hw::HWModuleOp::create(
      builder, loc, builder.getStringAttr(cellName), ports);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());

  llvm::StringMap<Value> portValues;
  for (const auto &port : ports)
    if (port.dir == hw::ModulePort::Direction::Input)
      portValues[port.name.getValue()] =
          moduleOp.getBodyBlock()->getArgument(port.argNum);

  SmallVector<Value> outputs;
  for (const auto &port : ports) {
    if (port.dir != hw::ModulePort::Direction::Output)
      continue;

    auto *it = llvm::find_if(pinGroups, [&](const LibertyGroup *g) {
      assert(g->name == "pin" && "expected pin group");
      // First arg is the pin name
      return cast<StringAttr>(g->args[0]).getValue() == port.name.getValue();
    });

    if (!it)
      continue;
    const LibertyGroup *pg = *it;
    auto attrPair = pg->getAttribute("function");
    if (!attrPair.first)
      return emitError(pg->loc, "expected function attribute");
    Value val;
    if (parseExpression(val, cast<StringAttr>(attrPair.first).getValue(),
                        portValues, attrPair.second))
      return failure();
    outputs.push_back(val);
  }

  auto *block = moduleOp.getBodyBlock();
  block->getTerminator()->setOperands(outputs);
  return success();
}

ParseResult LibertyParser::parseGroupBody(LibertyGroup &group) {
  // Parse args: ( arg1, arg2 )
  if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
    lexer.nextToken(); // (
    while (lexer.peekToken().kind != LibertyTokenKind::RParen) {
      Attribute arg;
      if (parseAttribute(arg))
        return failure();
      group.args.push_back(arg);
      if (lexer.peekToken().kind == LibertyTokenKind::Comma)
        lexer.nextToken();
    }
    if (consume(LibertyTokenKind::RParen, "expected ')'"))
      return failure();
  }

  // Parse body: { ... }
  if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
    lexer.nextToken(); // {
    while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
           lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
      if (parseStatement(group))
        return failure();
    }
    if (consume(LibertyTokenKind::RBrace, "expected '}'"))
      return failure();
  } else {
    // Optional semicolon if no body
    if (lexer.peekToken().kind == LibertyTokenKind::Semi)
      lexer.nextToken();
  }
  return success();
}

// Parse group, attribute, or define statements
ParseResult LibertyParser::parseStatement(LibertyGroup &parent) {
  auto nameTok = lexer.nextToken();
  if (nameTok.kind != LibertyTokenKind::Identifier)
    return emitError(nameTok.location, "expected identifier");
  StringRef name = nameTok.spelling;

  // Attribute statement.
  if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
    lexer.nextToken(); // :
    Attribute val;
    auto loc = lexer.peekToken().location;
    if (parseAttribute(val))
      return failure();
    parent.attrs.emplace_back(name, val, loc);
    return consume(LibertyTokenKind::Semi, "expected ';'");
  }

  // Group statement.
  if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
    auto subGroup = std::make_unique<LibertyGroup>();
    subGroup->name = name;
    subGroup->loc = nameTok.location;
    if (parseGroupBody(*subGroup))
      return failure();
    parent.subGroups.push_back(std::move(subGroup));
    return success();
  }

  // TODO: Support define.

  return emitError(nameTok.location, "expected ':' or '('");
}

ParseResult LibertyParser::parseAttribute(Attribute &result, OpBuilder &builder,
                                          StringRef attr) {
  double val;
  if (!attr.getAsDouble(val)) {
    result = builder.getF64FloatAttr(val);
  } else {
    // Keep as string if not a valid number
    result = builder.getStringAttr(attr);
  }
  return success();
}
// Parse an attribute value, which can be:
// - A string: "value"
// - A number: 1.23
// - An identifier: value
ParseResult LibertyParser::parseAttribute(Attribute &result) {
  auto token = lexer.peekToken();
  // Number token
  if (token.is(LibertyTokenKind::Number)) {
    lexer.nextToken();
    StringRef numStr = token.spelling;
    double val;
    if (!numStr.getAsDouble(val)) {
      result = builder.getF64FloatAttr(val);
      return success();
    }
    return emitError(token.location, "expected number value");
  }

  // Identifier token
  if (token.is(LibertyTokenKind::Identifier)) {
    lexer.nextToken();
    result = builder.getStringAttr(token.spelling);
    return success();
  }

  if (token.is(LibertyTokenKind::String)) {
    lexer.nextToken();
    StringRef str = getTokenSpelling(token);
    result = builder.getStringAttr(str);
    return success();
  }

  // TODO: Add array for timing attributes
  return emitError(token.location, "expected attribute value");
}

// Parse boolean expressions from Liberty function attributes
// Supports: *, +, ^, !, (), and identifiers
ParseResult LibertyParser::parseExpression(Value &result, StringRef expr,
                                           const llvm::StringMap<Value> &values,
                                           SMLoc loc) {
  ExpressionParser parser(lexer, builder, expr, values, loc);
  return parser.parse(result);
}

} // namespace

namespace circt::liberty {
void registerImportLibertyTranslation() {
  TranslateToMLIRRegistration reg(
      "import-liberty", "Import Liberty file",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        ModuleOp module = ModuleOp::create(UnknownLoc::get(context));
        LibertyParser parser(sourceMgr, context, module);
        // Load required dialects
        context->loadDialect<hw::HWDialect>();
        context->loadDialect<comb::CombDialect>();
        if (failed(parser.parse()))
          return OwningOpRef<ModuleOp>();
        return OwningOpRef<ModuleOp>(module);
      },
      [](DialectRegistry &registry) {
        registry.insert<HWDialect>();
        registry.insert<comb::CombDialect>();
      });
}
} // namespace circt::liberty
