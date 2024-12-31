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
std::string BLIFToken::getStringValue() const {
  assert(getKind() == string);
  return getStringValue(getSpelling());
}

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

/// Given a token containing a verbatim string, return its value, including
/// removing the quote characters and unescaping the quotes of the string. The
/// lexer has already verified that this token is valid.
std::string BLIFToken::getVerbatimStringValue() const {
  assert(getKind() == verbatim_string);
  return getVerbatimStringValue(getSpelling());
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
        return lexIdentifierOrKeyword(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that llvm::MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(BLIFToken::eof, tokStart);

      [[fallthrough]]; // Treat as whitespace.

    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;

    case '.':
      return lexPeriodOrKeyword(tokStart);
    case ',':
      return formToken(BLIFToken::comma, tokStart);
    case ':':
      return formToken(BLIFToken::colon, tokStart);
    case '(':
      return formToken(BLIFToken::l_paren, tokStart);
    case ')':
      return formToken(BLIFToken::r_paren, tokStart);
    case '{':
      if (*curPtr == '|')
        return ++curPtr, formToken(BLIFToken::l_brace_bar, tokStart);
      return formToken(BLIFToken::l_brace, tokStart);
    case '}':
      return formToken(BLIFToken::r_brace, tokStart);
    case '[':
      return formToken(BLIFToken::l_square, tokStart);
    case ']':
      return formToken(BLIFToken::r_square, tokStart);
    case '<':
      if (*curPtr == '=')
        return ++curPtr, formToken(BLIFToken::less_equal, tokStart);
      return formToken(BLIFToken::less, tokStart);
    case '>':
      return formToken(BLIFToken::greater, tokStart);
    case '=':
      if (*curPtr == '>')
        return ++curPtr, formToken(BLIFToken::equal_greater, tokStart);
      return formToken(BLIFToken::equal, tokStart);
    case '?':
      return formToken(BLIFToken::question, tokStart);
    case '@':
      if (*curPtr == '[')
        return lexFileInfo(tokStart);
      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");
    case '|':
      if (*curPtr == '}')
        return ++curPtr, formToken(BLIFToken::r_brace_bar, tokStart);
      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case '#':
      skipComment();
      continue;

    case '"':
      return lexString(tokStart, /*isVerbatim=*/false);
    case '\'':
      return lexString(tokStart, /*isVerbatim=*/true);

    case '-':
    case '+':
    case '0':
    case '1':
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

/// Lex a file info specifier.
///
///   FileInfo ::= '@[' ('\]'|.)* ']'
///
BLIFToken BLIFLexer::lexFileInfo(const char *tokStart) {
  while (1) {
    switch (*curPtr++) {
    case ']': // This is the end of the fileinfo literal.
      return formToken(BLIFToken::fileinfo, tokStart);
    case '\\':
      // Ignore escaped ']'
      if (*curPtr == ']')
        ++curPtr;
      break;
    case 0:
      // This could be the end of file in the middle of the fileinfo.  If so
      // emit an error.
      if (curPtr - 1 != curBuffer.end())
        break;
      [[fallthrough]];
    case '\n': // Vertical whitespace isn't allowed in a fileinfo.
    case '\v':
    case '\f':
      return emitError(tokStart, "unterminated file info specifier");
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Lex a period or a keyword that starts with a period.
///
///   Period ::= '.'
///
BLIFToken BLIFLexer::lexPeriodOrKeyword(const char *tokStart) {

  // Match the rest of the identifier regex: [a-zA-Z_$-]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '-')
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);

  // See if the identifier is a keyword.  By default, it is a period.
  BLIFToken::Kind kind = llvm::StringSwitch<BLIFToken::Kind>(spelling)
#define TOK_KEYWORD_DOT(SPELLING) .Case("." #SPELLING, BLIFToken::kw_##SPELLING)
#include "BLIFTokenKinds.def"
                             .Default(BLIFToken::period);
  if (kind != BLIFToken::period) {
    ++curPtr;
    return formToken(kind, tokStart);
  }

  // Otherwise, this is a period.
  return formToken(BLIFToken::period, tokStart);
}

/// Lex an identifier or keyword that starts with a letter.
///
///   LegalStartChar ::= [a-zA-Z_]
///   LegalIdChar    ::= LegalStartChar | [0-9] | '$'
///
///   Id ::= LegalStartChar (LegalIdChar)*
///   LiteralId ::= [a-zA-Z0-9$_]+
///
BLIFToken BLIFLexer::lexIdentifierOrKeyword(const char *tokStart) {
  // Remember that this is a literalID
  bool isLiteralId = *tokStart == '`';

  // Match the rest of the identifier regex: [0-9a-zA-Z_$-]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '-')
    ++curPtr;

  // Consume the trailing '`' in a literal identifier.
  if (isLiteralId) {
    if (*curPtr != '`')
      return emitError(tokStart, "unterminated literal identifier");
    ++curPtr;
  }

  StringRef spelling(tokStart, curPtr - tokStart);

  // Check to see if this is a 'primop', which is an identifier juxtaposed with
  // a '(' character.
  if (*curPtr == '(') {
    BLIFToken::Kind kind = llvm::StringSwitch<BLIFToken::Kind>(spelling)
#define TOK_LPKEYWORD(SPELLING) .Case(#SPELLING, BLIFToken::lp_##SPELLING)
#include "BLIFTokenKinds.def"
                               .Default(BLIFToken::identifier);
    if (kind != BLIFToken::identifier) {
      ++curPtr;
      return formToken(kind, tokStart);
    }
  }

  // See if the identifier is a keyword.  By default, it is an identifier.
  BLIFToken::Kind kind = llvm::StringSwitch<BLIFToken::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, BLIFToken::kw_##SPELLING)
#include "BLIFTokenKinds.def"
                             .Default(BLIFToken::identifier);

  // If this has the backticks of a literal identifier and it fell through the
  // above switch, indicating that it was not found to e a keyword, then change
  // its kind from identifier to literal identifier.
  if (isLiteralId && kind == BLIFToken::identifier)
    kind = BLIFToken::literal_identifier;

  return BLIFToken(kind, spelling);
}

/// Skip a comment line, starting with a ';' and going to end of line.
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

/// StringLit         ::= '"' UnquotedString? '"'
/// VerbatimStringLit ::= '\'' UnquotedString? '\''
/// UnquotedString    ::= ( '\\\'' | '\\"' | ~[\r\n] )+?
///
BLIFToken BLIFLexer::lexString(const char *tokStart, bool isVerbatim) {
  while (1) {
    switch (*curPtr++) {
    case '"': // This is the end of the string literal.
      if (isVerbatim)
        break;
      return formToken(BLIFToken::string, tokStart);
    case '\'': // This is the end of the raw string.
      if (!isVerbatim)
        break;
      return formToken(BLIFToken::verbatim_string, tokStart);
    case '\\':
      // Ignore escaped '\'' or '"'
      if (*curPtr == '\'' || *curPtr == '"' || *curPtr == '\\')
        ++curPtr;
      else if (*curPtr == 'u' || *curPtr == 'U')
        return emitError(tokStart, "unicode escape not supported in string");
      break;
    case 0:
      // This could be the end of file in the middle of the string.  If so
      // emit an error.
      if (curPtr - 1 != curBuffer.end())
        break;
      [[fallthrough]];
    case '\n': // Vertical whitespace isn't allowed in a string.
    case '\r':
    case '\v':
    case '\f':
      return emitError(tokStart, "unterminated string");
    default:
      if (curPtr[-1] & ~0x7F)
        return emitError(tokStart, "string characters must be 7-bit ASCII");
      // Skip over other characters.
      break;
    }
  }
}

/// Lex a number literal.
///
///   UnsignedInt ::= '0' | PosInt
///   PosInt ::= [1-9] ([0-9])*
///   DoubleLit ::=
///       ( '+' | '-' )? Digit+ '.' Digit+ ( 'E' ( '+' | '-' )? Digit+ )?
///   TripleLit ::=
///       Digit+ '.' Digit+ '.' Digit+
///   Radix-specified Integer ::=
///       ( '-' )? '0' ( 'b' | 'o' | 'd' | 'h' ) LegalDigit*
///
BLIFToken BLIFLexer::lexNumber(const char *tokStart) {
  assert(llvm::isDigit(curPtr[-1]) || curPtr[-1] == '+' || curPtr[-1] == '-');

  // There needs to be at least one digit.
  if (!llvm::isDigit(*curPtr) && !llvm::isDigit(curPtr[-1]))
    return emitError(tokStart, "unexpected character after sign");

  // If we encounter a "b", "o", "d", or "h", this is a radix-specified integer
  // literal.  This is only supported for BLIFRTL 2.4.0 or later.  This is
  // always lexed, but rejected during parsing if the version is too old.
  const char *oldPtr = curPtr;
  if (curPtr[-1] == '-' && *curPtr == '0')
    ++curPtr;
  if (curPtr[-1] == '0') {
    switch (*curPtr) {
    case 'b':
      ++curPtr;
      while (*curPtr >= '0' && *curPtr <= '1')
        ++curPtr;
      return formToken(BLIFToken::radix_specified_integer, tokStart);
    case 'o':
      ++curPtr;
      while (*curPtr >= '0' && *curPtr <= '7')
        ++curPtr;
      return formToken(BLIFToken::radix_specified_integer, tokStart);
    case 'd':
      ++curPtr;
      while (llvm::isDigit(*curPtr))
        ++curPtr;
      return formToken(BLIFToken::radix_specified_integer, tokStart);
    case 'h':
      ++curPtr;
      while (llvm::isHexDigit(*curPtr))
        ++curPtr;
      return formToken(BLIFToken::radix_specified_integer, tokStart);
    default:
      curPtr = oldPtr;
      break;
    }
  }

  while (llvm::isDigit(*curPtr))
    ++curPtr;

  // If we encounter a '.' followed by a digit, then this is a floating point
  // literal, otherwise this is an integer or negative integer.
  if (*curPtr != '.' || !llvm::isDigit(curPtr[1])) {
    if (*tokStart == '-' || *tokStart == '+')
      return formToken(BLIFToken::signed_integer, tokStart);
    return formToken(BLIFToken::integer, tokStart);
  }

  // Lex a floating point literal.
  curPtr += 2;
  while (llvm::isDigit(*curPtr))
    ++curPtr;

  bool hasE = false;
  if (*curPtr == 'E') {
    hasE = true;
    ++curPtr;
    if (*curPtr == '+' || *curPtr == '-')
      ++curPtr;
    while (llvm::isDigit(*curPtr))
      ++curPtr;
  }

  // If we encounter a '.' followed by a digit, again, and there was no
  // exponent, then this is a version literal.  Otherwise it is a floating point
  // literal.
  if (*curPtr != '.' || !llvm::isDigit(curPtr[1]) || hasE)
    return formToken(BLIFToken::floatingpoint, tokStart);

  // Lex a version literal.
  curPtr += 2;
  while (llvm::isDigit(*curPtr))
    ++curPtr;
  return formToken(BLIFToken::version, tokStart);
}
