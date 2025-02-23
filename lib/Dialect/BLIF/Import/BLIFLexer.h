//===- BLIFLexer.h - .blif lexer and token definitions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the a Lexer and Token interface for .blif files.
//
//===----------------------------------------------------------------------===//

#ifndef BLIFTOMLIR_BLIFLEXER_H
#define BLIFTOMLIR_BLIFLEXER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
class Location;
} // namespace mlir

namespace circt {
namespace blif {

/// This represents a specific token for .fir files.
class BLIFToken {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#define TOK_KEYWORD_DOT(SPELLING) kw_##SPELLING,
#include "BLIFTokenKinds.def"
  };

  BLIFToken(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_wire).
  bool isKeyword() const;

  bool isModelHeaderKeyword() const {
    return isAny(kw_inputs, kw_outputs, kw_clocks, kw_input, kw_output,
                 kw_clock);
  }

  /// Given a token containing a string literal, return its value, including
  /// removing the quote characters and unescaping the contents of the string.
  /// The lexer has already verified that this token is valid.
  std::string getStringValue() const;
  static std::string getStringValue(StringRef spelling);

  /// Given a token containing a verbatim string, return its value, including
  /// removing the quote characters and unescaping the quotes of the string. The
  /// lexer has already verified that this token is valid.
  std::string getVerbatimStringValue() const;
  static std::string getVerbatimStringValue(StringRef spelling);

  // Location processing.
  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

class BLIFLexerCursor;

/// This implements a lexer for .fir files.
class BLIFLexer {
public:
  BLIFLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context);

  const llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }

  /// Move to the next valid token.
  void lexToken() { curToken = lexTokenImpl(); }

  const BLIFToken &getToken() const { return curToken; }

  mlir::Location translateLocation(llvm::SMLoc loc);

  /// Get an opaque pointer into the lexer state that can be restored later.
  BLIFLexerCursor getCursor() const;

private:
  BLIFToken lexTokenImpl();

  // Helpers.
  BLIFToken formToken(BLIFToken::Kind kind, const char *tokStart) {
    return BLIFToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  BLIFToken emitError(const char *loc, const Twine &message);

  // Lexer implementation methods.
  BLIFToken lexFileInfo(const char *tokStart);
  BLIFToken lexIdentifier(const char *tokStart);
  BLIFToken lexNumber(const char *tokStart);
  BLIFToken lexNumberOrCover(const char *tokStart);
  void skipComment();
  BLIFToken lexString(const char *tokStart, bool isVerbatim);
  BLIFToken lexCommand(const char *tokStart);

  const llvm::SourceMgr &sourceMgr;
  const mlir::StringAttr bufferNameIdentifier;

  StringRef curBuffer;
  const char *curPtr;

  /// This is the next token that hasn't been consumed yet.
  BLIFToken curToken;

  BLIFLexer(const BLIFLexer &) = delete;
  void operator=(const BLIFLexer &) = delete;
  friend class BLIFLexerCursor;
};

/// This is the state captured for a lexer cursor.
class BLIFLexerCursor {
public:
  BLIFLexerCursor(const BLIFLexer &lexer)
      : state(lexer.curPtr), curToken(lexer.getToken()) {}

  void restore(BLIFLexer &lexer) {
    lexer.curPtr = state;
    lexer.curToken = curToken;
  }

private:
  const char *state;
  BLIFToken curToken;
};

inline BLIFLexerCursor BLIFLexer::getCursor() const {
  return BLIFLexerCursor(*this);
}

} // namespace blif
} // namespace circt

#endif // BLIFTOMLIR_BLIFLEXER_H
