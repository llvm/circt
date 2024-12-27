//===- VCD.h - Support for VCD parser/printer -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides support for VCD parser/printer.
//           (Lexer)                 (Parser)                  (Parser)
// VCD input -----> Sequence of Tokens ----> Sequence of Command ->
// VCDFile struct
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_VCD_H
#define CIRCT_SUPPORT_VCD_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"

#include <functional>
#include <variant>

using namespace circt;
using namespace mlir;
namespace circt {
namespace vcd {

struct VCDFile {
  struct Header {
    StringAttr timestamp;
    StringAttr versionNumber;
    uint64_t timescale;
  } header;

  struct Node {};
  struct Scope : Node {
    StringAttr name;
    // SV 21.7.2.1
    enum ScopeType {
      module,
      begin,
    } kind;
    SmallVector<std::unique_ptr<Node>> children;
    Scope(StringAttr name, ScopeType kind) : name(name), kind(kind) {}
  };

  struct Decl {
    enum Kind {
      scope,
      variable,
    } kind;
  };

  struct Variable : Node {
    enum Kind {
      wire,
      reg,
      integer,
      real,
      time,
    } kind;
    StringAttr name;
  };

  struct VariableDefinition {
    std::unique_ptr<Scope> scope;
  } definition;

  struct ValueChange {
    // Lazying loading.
    StringRef remainingData;
  } valueChange;
};

class VCDToken {
public:
  enum Kind {
    // Markers
    eof, // End of file

    // Basic tokens
    identifier,   // Signal identifiers
    command,      // $... commands
    time_value,   // #... time values
    scalar_value, // 0,1,x,z etc
    vector_value, // b... or B...
    real_value,   // r... or R...

    // Commands
    kw_date,
    kw_version,
    kw_timescale,
    kw_scope,
    kw_var,
    kw_upscope,
    kw_enddefinitions,
    kw_dumpvars,
    kw_end,

    // Punctuation
    l_paren, // (
    r_paren, // )

    // Special
    string, // "..."
  };

  VCDToken() = default;
  VCDToken(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  StringRef getSpelling() const { return spelling; }
  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }
  bool isCommand() const { return spelling.starts_with("$"); }

  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  Kind kind;
  StringRef spelling;
  llvm::SMLoc loc;
};

struct VCDCommand {
  VCDToken token;
  SmallVector<VCDToken> args;

  VCDCommand() = default;
  VCDCommand(VCDToken token, SmallVector<VCDToken> args)
      : token(token), args(args) {}
};

class VCDLexer {
public:
  VCDLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context);

  void lexToken() { curToken = lexTokenImpl(); }
  const VCDToken &getToken() const { return curToken; }
  mlir::Location translateLocation(llvm::SMLoc loc);

private:
  VCDToken lexTokenImpl();
  VCDToken formToken(VCDToken::Kind kind, const char *tokStart) {
    return VCDToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  VCDToken emitError(const char *loc, const Twine &message);

  // Lexer implementation methods
  VCDToken lexIdentifier(const char *tokStart);
  VCDToken lexCommand(const char *tokStart);
  VCDToken lexTimeValue(const char *tokStart);
  VCDToken lexString(const char *tokStart);
  VCDToken lexValue(const char *tokStart);
  void skipWhitespace();

  const llvm::SourceMgr &sourceMgr;
  const mlir::StringAttr bufferNameIdentifier;
  StringRef curBuffer;
  const char *curPtr;
  VCDToken curToken;
};

VCDToken VCDLexer::lexTokenImpl() {
  skipWhitespace();

  const char *tokStart = curPtr;
  switch (*curPtr++) {
  default:
    // Handle identifiers
    if (llvm::isAlpha(curPtr[-1]))
      return lexIdentifier(tokStart);
    return emitError(tokStart, "unexpected character");

  case 0:
    // Handle EOF
    if (curPtr - 1 == curBuffer.end())
      return formToken(VCDToken::eof, tokStart);
    [[fallthrough]];

  case ' ':
  case '\t':
  case '\n':
  case '\r':
    skipWhitespace();
    return lexTokenImpl();

  case '$':
    return lexCommand(tokStart);

  case '#':
    return lexTimeValue(tokStart);

  case '"':
    return lexString(tokStart);

  case '(':
    return formToken(VCDToken::l_paren, tokStart);
  case ')':
    return formToken(VCDToken::r_paren, tokStart);

  // Value changes
  case '0':
  case '1':
  case 'x':
  case 'X':
  case 'z':
  case 'Z':
  case 'b':
  case 'B':
  case 'r':
  case 'R':
    return lexValue(tokStart);
  }
}

VCDToken VCDLexer::lexCommand(const char *tokStart) {
  // Read until whitespace
  while (!llvm::isSpace(*curPtr) && *curPtr != 0)
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);

  // Match known commands
  VCDToken::Kind kind =
      llvm::StringSwitch<VCDToken::Kind>(spelling)
          .Case("$date", VCDToken::kw_date)
          .Case("$version", VCDToken::kw_version)
          .Case("$timescale", VCDToken::kw_timescale)
          .Case("$scope", VCDToken::kw_scope)
          .Case("$var", VCDToken::kw_var)
          .Case("$upscope", VCDToken::kw_upscope)
          .Case("$enddefinitions", VCDToken::kw_enddefinitions)
          .Case("$dumpvars", VCDToken::kw_dumpvars)
          .Case("$end", VCDToken::kw_end)
          .Default(VCDToken::command);

  if (kind == VCDToken::command)
    return emitError(tokStart, "unknown command");

  return VCDToken(kind, spelling);
}

VCDToken VCDLexer::lexTimeValue(const char *tokStart) {
  while (llvm::isDigit(*curPtr))
    ++curPtr;
  return formToken(VCDToken::time_value, tokStart);
}

VCDToken VCDLexer::lexValue(const char *tokStart) {
  char firstChar = tokStart[0];

  // Handle scalar values
  if (firstChar == '0' || firstChar == '1' || firstChar == 'x' ||
      firstChar == 'X' || firstChar == 'z' || firstChar == 'Z') {
    return formToken(VCDToken::scalar_value, tokStart);
  }

  // Handle vector values (b... or B...)
  if (firstChar == 'b' || firstChar == 'B') {
    while (*curPtr == '0' || *curPtr == '1' || *curPtr == 'x' ||
           *curPtr == 'X' || *curPtr == 'z' || *curPtr == 'Z')
      ++curPtr;
    return formToken(VCDToken::vector_value, tokStart);
  }

  // Handle real values (r... or R...)
  if (firstChar == 'r' || firstChar == 'R') {
    while (llvm::isDigit(*curPtr) || *curPtr == '.' || *curPtr == '-')
      ++curPtr;
    return formToken(VCDToken::real_value, tokStart);
  }

  return emitError(tokStart, "invalid value format");
}

VCDToken VCDLexer::lexIdentifier(const char *tokStart) {
  while (llvm::isAlnum(*curPtr) || *curPtr == '_' || *curPtr == '$')
    ++curPtr;
  return formToken(VCDToken::identifier, tokStart);
}

VCDToken VCDLexer::lexString(const char *tokStart) {
  while (*curPtr != '"' && *curPtr != 0) {
    if (*curPtr == '\\' && *(curPtr + 1) == '"')
      curPtr += 2;
    else
      ++curPtr;
  }

  if (*curPtr == 0)
    return emitError(tokStart, "unterminated string");

  ++curPtr; // Consume closing quote
  return formToken(VCDToken::string, tokStart);
}

void VCDLexer::skipWhitespace() {
  while (llvm::isSpace(*curPtr))
    ++curPtr;
}

struct VCDParser {
  bool lazyLoadValueChange = false;
  ParseResult parseCommand(VCDCommand &command) {
    const auto &token = lexer.getToken();
    // if is command
    if (!token.isCommand())
      return failure();

    command.token = token;
    consumeToken();

    while (!lexer.getToken().is(VCDToken::kw_end)) {
      auto arg = lexer.getToken();
      command.args.push_back(arg);
      consumeToken();
    }
    return success();
  }

  ParseResult parseCommands(SmallVectorImpl<VCDCommand> &commands) {
    while (!lexer.getToken().is(VCDToken::eof)) {
      VCDCommand command;
      if (parseCommand(command).failed())
        return failure();
      commands.push_back(command);
    }
    return success();
  }

  FailureOr<VCDFile> convertCommandsToFile(ArrayRef<VCDCommand> commands) {}

  FailureOr<VCDFile> convertCommandsToFile(ArrayRef<VCDCommand> commands) {
    VCDFile file;

    struct ParserImpl {
      VCDParser &parser;
      ArrayRef<VCDCommand> commands;
      ParserImpl(VCDParser &parser, ArrayRef<VCDCommand> commands)
          : parser(parser), commands(commands) {}
      VCDCommand getCommand() { return commands.front(); }
      void consumeCommand() { commands = commands.drop_front(); }

      bool isDone() { return commands.empty(); }
      std::unique_ptr<VCDFile::Scope> parseScope() {
        auto command = getCommand();
        assert(command.token.getKind() == VCDToken::kw_scope);
        consumeCommand();
        auto scope = std::make_unique<VCDFile::Scope>(name);
      }
    } parser(*this, commands);
    // 1. Parse header section
    while (!parser.isDone() &&
           !parser.getCommand().token.is(VCDToken::kw_scope)) {
      const auto &front = parser.getCommand();
      switch (front.token.getKind()) {
      case VCDToken::kw_date:
        file.header.timestamp =
            StringAttr::get(getContext(), front.args[0].getSpelling());
        break;
      case VCDToken::kw_version:
        file.header.versionNumber =
            StringAttr::get(getContext(), front.args[0].getSpelling());
        break;
      case VCDToken::kw_timescale:
        // FIXME: Parse timescale
        file.header.timescale = 0;
        break;
      default:
        return failure();
      }
      parser.consumeCommand();
    }

    // There is no scope command, weird but we are done.
    if (parser.isDone())
      return success();

    // 2. Parse variable definition section
    // 3. Parse value change section

    return file;
  }
  LogicalResult parseVariableDefinition();
  LogicalResult parseValueChange();
  mlir::MLIRContext *context;
  mlir::MLIRContext *getContext() { return context; }

  InFlightDiagnostic emitError() {
    // TODO: Add location converter
    return mlir::emitError(mlir::UnknownLoc::get(getContext()));
  }

  void consumeToken() { lexer.lexToken(); }

  mlir::ParseResult parseExpectedId(StringRef expected) {
    const auto &token = lexer.getToken();
    if (token.is(VCDToken::identifier) && token.getSpelling() == expected) {
      consumeToken();
      return success();
    }
    return failure();
  }

  mlir::ParseResult parseExpectedKeyword(VCDToken::Kind expected) {
    const auto &token = lexer.getToken();
    if (token.getKind() == expected) {
      consumeToken();
      return success();
    }
    return failure();
  }

  mlir::ParseResult parseId(StringRef &id) {
    const auto &token = lexer.getToken();
    if (!token.is(VCDToken::identifier))
      return emitError() << "expected id";

    id = token.getSpelling();
    consumeToken();
    return success();
  }

  const VCDToken &getToken() const { return lexer.getToken(); }
  bool consumeKeywordIf(StringRef keyword) {
    const auto &token = lexer.getToken();
    if (token.is(VCDToken::command) && token.getSpelling() == keyword) {
      consumeToken();
      return true;
    }
    return false;
  }

  // These are pass-through but defined for understandability.
  LogicalResult convertValueChange();
  LogicalResult convertScope();
  const VCDToken &getNextToken() {
    consumeToken();
    return getToken();
  }

  SmallVector<hw::HWModuleOp> moduleStack;
  igraph::InstanceGraph &instanceGraph;
  VCDLexer lexer;

  friend raw_ostream &operator<<(raw_ostream &os, const VCDToken &token) {
    if (token.is(VCDToken::eof))
      return os << 0;
    if (token.is(VCDToken::command))
      os << '$';
    return os << token.getSpelling();
  }

  ParseResult parse(VCDFile &file);
};

struct VCDEmitter {};

} // namespace vcd
} // namespace circt

#endif // CIRCT_SUPPORT_VCD_H
