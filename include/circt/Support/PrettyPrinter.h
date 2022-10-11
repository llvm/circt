//===- PrettyPrinter.h - Pretty printing ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a pretty-printer.
// "PrettyPrinting", Derek C. Oppen, 1980.
// https://dx.doi.org/10.1145/357114.357115
//
// This was selected as it is linear in number of tokens O(n) and requires
// memory O(linewidth).
//
// See PrettyPrinter.cpp for more information.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PRETTYPRINTER_H
#define CIRCT_SUPPORT_PRETTYPRINTER_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <deque>
#include <vector>

namespace circt {
namespace pretty {

// TODO: revisit integer types, don't use sint/uint, consider packing.
using sint = int32_t;
using uint = uint32_t;

//===----------------------------------------------------------------------===//
// Tokens
//===----------------------------------------------------------------------===//

enum class Breaks { Consistent, Inconsistent };

class Token {
public:
  enum class Kind { String, Break, Begin, End };

private:
  const Kind kind;

protected:
  Token(Kind k) : kind(k) {}

public:
  Kind getKind() const { return kind; }
  virtual ~Token() = default;
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Token::Kind DerivedKind>
struct TokenBase : public Token {
  TokenBase() : Token(DerivedKind) {}
  static bool classof(const Token *t) { return t->getKind() == DerivedKind; }
  ~TokenBase() override = default;
};

struct StringToken : public TokenBase<StringToken, Token::Kind::String> {
  llvm::StringRef text;
  StringToken(llvm::StringRef text) : text(text) {}
};

struct BreakToken : public TokenBase<BreakToken, Token::Kind::Break> {
  uint spaces; // how many spaces when not broken
  sint offset; // amount to adjust indentation level by if breaks here
  // ... extras
  BreakToken(uint spaces = 1, sint offset = 0)
      : spaces(spaces), offset(offset) {}
};

struct BeginToken : public TokenBase<BeginToken, Token::Kind::Begin> {
  sint offset;
  Breaks breaks;
  BeginToken(ssize_t offset = 2, Breaks breaks = Breaks::Inconsistent)
      : offset(offset), breaks(breaks) {}
};

struct EndToken : public TokenBase<EndToken, Token::Kind::End> {};

//===----------------------------------------------------------------------===//
// PrettyPrinter
//===----------------------------------------------------------------------===//

class PrettyPrinter {
public:
  PrettyPrinter(llvm::raw_ostream &os, uint margin)
      : space(margin), margin(margin), os(os) {}

  /// Add token for printing.  In Oppen, this is "scan".
  void add(Token &t);

  void eof() {
    if (!scanStack.empty()) {
      checkStack(0);
      advanceLeft();
    }
  }

private:
  /// Format token with tracked size.
  struct FormattedToken {
    Token &token; /// underlying token
    sint size;    /// calculate size when positive.
  };

  enum class PrintBreaks { Consistent, Inconsistent, Fits };

  /// Printing information for active scope, stored in printStack.
  struct PrintEntry {
    uint offset;
    PrintBreaks breaks;
  };

  /// Print out tokens we know sizes for, and drop from token buffer.
  void advanceLeft();

  /// Break encountered, set sizes of begin/breaks in scanStack we now know.
  void checkStack(uint depth);

  /// Check if there's enough tokens to hit width, if so print.
  /// If scan size is wider than line, it's infinity.
  void checkStream();

  /// Print a token, maintaining printStack for context.
  void print(FormattedToken f);

  /// Characters left on this line.
  sint space;

  /// Sizes: printed, enqueued
  sint leftTotal;
  sint rightTotal;

  /// Unprinted tokens, combination of 'token' and 'size' in Oppen.
  std::deque<FormattedToken> tokens;
  /// index of first token, for resolving scanStack entries.
  uint tokenOffset = 0;

  /// Stack of begin/break tokens, adjust by tokenOffset to index into tokens.
  std::deque<uint> scanStack;

  /// Stack of printing contexts (indentation + breaking behavior).
  std::vector<PrintEntry> printStack;

  /// Current indentation level
  // TODO: implement this!
  // uint indent;

  /// Whitespace to print before next, tracked to avoid trailing whitespace.
  // TODO: implement this!
  // uint pendingIndentation;

  // sizeInfinity
  // printStack

  /// Target line width.
  uint margin;

  /// Output stream.
  llvm::raw_ostream &os;
};

} // end namespace pretty
} // end namespace circt

#endif // CIRCT_SUPPORT_PRETTYPRINTER_H
