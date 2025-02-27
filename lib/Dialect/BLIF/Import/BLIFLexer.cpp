//===- BLIFLexer.cpp - .blif file lexer implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .blif file lexer.
//
//===----------------------------------------------------------------------===//

#include "BLIFLexer.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace blif;
using llvm::SMLoc;
using llvm::SMRange;
using llvm::SourceMgr;

#define isdigit(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS
#define isalpha(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS

//===----------------------------------------------------------------------===//
// BLIFToken
//===----------------------------------------------------------------------===//

SMLoc BLIFToken::getLoc() const {
  return SMLoc::getFromPointer(spelling.data());
}

SMLoc BLIFToken::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange BLIFToken::getLocRange() const {
  return SMRange(getLoc(), getEndLoc());
}

/// Return true if this is one of the keyword token kinds (e.g. kw_wire).
bool BLIFToken::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "BLIFTokenKinds.def"
  }
}

/// Given a token containing a string literal, return its value, including
/// removing the quote characters and unescaping the contents of the string. The
/// lexer has already verified that this token is valid.
std::string BLIFToken::getStringValue(StringRef spelling) {
  // Start by dropping the quotes.
  StringRef bytes = spelling.drop_front().drop_back();

  std::string result;
  result.reserve(bytes.size());
  for (size_t i = 0, e = bytes.size(); i != e;) {
    auto c = bytes[i++];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c1 = bytes[i++];
    switch (c1) {
    case '\\':
    case '"':
    case '\'':
      result.push_back(c1);
      continue;
    case 'b':
      result.push_back('\b');
      continue;
    case 'n':
      result.push_back('\n');
      continue;
    case 't':
      result.push_back('\t');
      continue;
    case 'f':
      result.push_back('\f');
      continue;
    case 'r':
      result.push_back('\r');
      continue;
      // TODO: Handle the rest of the escapes (octal and unicode).
    default:
      break;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c2 = bytes[i++];

    assert(llvm::isHexDigit(c1) && llvm::isHexDigit(c2) && "invalid escape");
    result.push_back((llvm::hexDigitValue(c1) << 4) | llvm::hexDigitValue(c2));
  }

  return result;
}

std::string BLIFToken::getVerbatimStringValue(StringRef spelling) {
  // Start by dropping the quotes.
  StringRef bytes = spelling.drop_front().drop_back();

  std::string result;
  result.reserve(bytes.size());
  for (size_t i = 0, e = bytes.size(); i != e;) {
    auto c = bytes[i++];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c1 = bytes[i++];
    if (c1 != '\'') {
      result.push_back(c);
    }
    result.push_back(c1);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// BLIFLexer
//===----------------------------------------------------------------------===//

static StringAttr getMainBufferNameIdentifier(const llvm::SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

BLIFLexer::BLIFLexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
      curBuffer(
          sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
      curPtr(curBuffer.begin()),
      // Prime the BLIFst token.
      curToken(lexTokenImpl()) {}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location BLIFLexer::translateLocation(llvm::SMLoc loc) {
  assert(loc.isValid());
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
}

/// Emit an error message and return a BLIFToken::error token.
BLIFToken BLIFLexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(translateLocation(SMLoc::getFromPointer(loc)), message);
  return formToken(BLIFToken::error, loc);
}

//===----------------------------------------------------------------------===//
// Lexer Implementation Methods
//===----------------------------------------------------------------------===//

BLIFToken BLIFLexer::lexTokenImpl() {
  while (true) {
    const char *tokStart = curPtr;
    switch (*curPtr++) {
    default:
      // Handle identifiers.
      if (llvm::isAlpha(curPtr[-1]))
        return lexIdentifier(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that llvm::MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(BLIFToken::eof, tokStart);

      [[fallthrough]]; // Treat as whitespace.

    case '\\':
      // Handle line continuations.
      if (*curPtr == '\r') {
        ++curPtr;
      }
      if (*curPtr == '\n') {
        ++curPtr;
        continue;
      }

    case '\n':
      //        return formToken(BLIFToken::newline, tokStart);

    case ' ':
    case '\t':
    case '\r':
      // Handle whitespace.
      continue;

    case '.':
      return lexCommand(tokStart);

    case '#':
      skipComment();
      continue;

    case '-':
    case '0':
    case '1':
      return lexNumberOrCover(tokStart);

    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexNumber(tokStart);
    }
  }
}

/// Lex a period or a keyword that starts with a period.
///
///   Command :== '.' [a-zA-Z_]+
///
BLIFToken BLIFLexer::lexCommand(const char *tokStart) {

  // Match the rest of the command regex: [a-zA-Z_]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_')
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);

  // See if the identifier is a keyword.
  BLIFToken::Kind kind = llvm::StringSwitch<BLIFToken::Kind>(spelling)
#define TOK_KEYWORD_DOT(SPELLING) .Case("." #SPELLING, BLIFToken::kw_##SPELLING)
#include "BLIFTokenKinds.def"
                             .Default(BLIFToken::error);
  if (kind != BLIFToken::error) {
    ++curPtr;
    return formToken(kind, tokStart);
  }

  // Otherwise, this is a period.
  return emitError(tokStart, "unexpected character after period");
}

/// Lex an identifier.
///
///   LegalStartChar ::= [a-zA-Z]
///   LegalIdChar    ::= LegalStartChar | [0-9] | '$' | '_'
///
///   Id ::= LegalStartChar (LegalIdChar)*
///
BLIFToken BLIFLexer::lexIdentifier(const char *tokStart) {

  // Match the rest of the identifier regex: [0-9a-zA-Z$_]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$')
    ++curPtr;

  return formToken(BLIFToken::identifier, tokStart);
}

/// Skip a comment line, starting with a '#' and going to end of line.
void BLIFLexer::skipComment() {
  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Lex a number literal.
///
///   UnsignedInt ::= '0' | PosInt
///   PosInt ::= [1-9] ([0-9])*
///
BLIFToken BLIFLexer::lexNumber(const char *tokStart) {
  assert(llvm::isDigit(curPtr[-1]) || curPtr[-1] == '-');

  // There needs to be at least one digit.
  if (!llvm::isDigit(*curPtr) && !llvm::isDigit(curPtr[-1]))
    return emitError(tokStart, "unexpected character after sign");

  while (llvm::isDigit(*curPtr))
    ++curPtr;

  return formToken(BLIFToken::integer, tokStart);
}

/// Lex a number literal or a cover literal
///
///
///   Cover ::= [0-9\-]*
///
BLIFToken BLIFLexer::lexNumberOrCover(const char *tokStart) {
  while (llvm::isDigit(*curPtr) || *curPtr == '-')
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);
  if (spelling.contains('2') || spelling.contains('3') ||
      spelling.contains('4') || spelling.contains('5') ||
      spelling.contains('6') || spelling.contains('7') ||
      spelling.contains('8') || spelling.contains('9') ||
      !spelling.contains('-')) {
    curPtr = tokStart + 1;
    return lexNumber(tokStart);
  }
  return formToken(BLIFToken::integer, tokStart);
}
