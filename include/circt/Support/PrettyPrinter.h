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

//===----------------------------------------------------------------------===//
// Tokens
//===----------------------------------------------------------------------===//

enum class Breaks { Consistent, Inconsistent };

class Token {
public:
  enum class Kind { String, Break, Begin, End };
  struct StringInfo {
    const char *str;
    uint32_t len;
  };
  struct BreakInfo {
    uint32_t spaces; // how many spaces when not broken
    int32_t offset;  // amount to adjust indentation level by if breaks here
  };
  struct BeginInfo {
    int32_t offset;
    Breaks breaks;
  };
  struct EndInfo {
    // Nothing
  };

private:
  union {
    StringInfo stringInfo;
    BreakInfo breakInfo;
    BeginInfo beginInfo;
    EndInfo endInfo;
  } data;
  const Kind kind;

protected:
  template <Kind k>
  auto &getInfoImpl() {
    if constexpr (k == Kind::String)
      return data.stringInfo;
    if constexpr (k == Kind::Break)
      return data.breakInfo;
    if constexpr (k == Kind::Begin)
      return data.beginInfo;
    if constexpr (k == Kind::End)
      return data.endInfo;
    llvm_unreachable("unhandled token kind");
  }

  Token(Kind k) : kind(k) {}

public:
  auto getKind() const { return kind; }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Token::Kind DerivedKind>
struct TokenBase : public Token {
  static bool classof(const Token *t) { return t->getKind() == DerivedKind; }

protected:
  TokenBase() : Token(DerivedKind) {}
  auto &getInfoMut() { return getInfoImpl<DerivedKind>(); }

public:
  const auto &getInfo() { return getInfoMut(); }
};

struct StringToken : public TokenBase<StringToken, Token::Kind::String> {
  StringToken(llvm::StringRef text) {
    assert(text.size() == (uint32_t)text.size());
    getInfoMut() = {text.data(), (uint32_t)text.size()};
  }
  StringRef text() { return StringRef(getInfo().str, getInfo().len); }
};

struct BreakToken : public TokenBase<BreakToken, Token::Kind::Break> {
  BreakToken(uint32_t spaces = 1, int32_t offset = 0) {
    getInfoMut() = {spaces, offset};
  }
  auto spaces() { return getInfo().spaces; }
  auto offset() { return getInfo().offset; }
};

struct BeginToken : public TokenBase<BeginToken, Token::Kind::Begin> {
  BeginToken(int32_t offset = 2, Breaks breaks = Breaks::Inconsistent) {
    getInfoMut() = {offset, breaks};
  }
  auto offset() { return getInfo().offset; }
  auto breaks() { return getInfo().breaks; }
};

struct EndToken : public TokenBase<EndToken, Token::Kind::End> {};

//===----------------------------------------------------------------------===//
// PrettyPrinter
//===----------------------------------------------------------------------===//

class PrettyPrinter {
public:
  /// Listener to Token storage events.
  struct Listener {
    virtual ~Listener();
    /// No tokens referencing external memory are present.
    virtual void clear(){};
  };

  // TODO: allow setting starting indentation level!
  PrettyPrinter(llvm::raw_ostream &os, uint32_t margin,
                Listener *listener = nullptr)
      : space(margin), margin(margin), os(os), listener(listener) {
    assert(margin < kInfinity / 2);
  }

  /// Add token for printing.  In Oppen, this is "scan".
  void add(Token t);

  /// Add a range of tokens.
  template <typename R>
  void addTokens(R &&tokens) {
    for (Token &t : tokens)
      add(t);
  }

  void eof();

  void setListener(Listener *newListener) { listener = newListener; };
  auto *getListener() const { return listener; }

  static constexpr uint32_t kInfinity = 0xFFFFFU;

private:
  /// Format token with tracked size.
  struct FormattedToken {
    Token token;  /// underlying token
    int32_t size; /// calculate size when positive.
  };

  enum class PrintBreaks { Consistent, Inconsistent, Fits };

  /// Printing information for active scope, stored in printStack.
  struct PrintEntry {
    uint32_t offset;
    PrintBreaks breaks;
  };

  /// Print out tokens we know sizes for, and drop from token buffer.
  void advanceLeft();

  /// Break encountered, set sizes of begin/breaks in scanStack we now know.
  void checkStack();

  /// Check if there's enough tokens to hit width, if so print.
  /// If scan size is wider than line, it's infinity.
  void checkStream();

  /// Print a token, maintaining printStack for context.
  void print(FormattedToken f);

  /// Clear token buffer, scanStack must be empty.
  void clear();

  /// Characters left on this line.
  int32_t space;

  /// Sizes: printed, enqueued
  int32_t leftTotal;
  int32_t rightTotal;

  /// Unprinted tokens, combination of 'token' and 'size' in Oppen.
  std::deque<FormattedToken> tokens;
  /// index of first token, for resolving scanStack entries.
  uint32_t tokenOffset = 0;

  /// Stack of begin/break tokens, adjust by tokenOffset to index into tokens.
  std::deque<uint32_t> scanStack;

  /// Stack of printing contexts (indentation + breaking behavior).
  std::vector<PrintEntry> printStack;

  /// Current indentation level
  // TODO: implement this!
  // uint32_t indent;

  /// Whitespace to print before next, tracked to avoid trailing whitespace.
  // TODO: implement this!
  // uint32_t pendingIndentation;

  // sizeInfinity
  // printStack

  /// Target line width.
  uint32_t margin;

  /// Output stream.
  llvm::raw_ostream &os;

  Listener *listener = nullptr;
};

} // end namespace pretty
} // end namespace circt

#endif // CIRCT_SUPPORT_PRETTYPRINTER_H
