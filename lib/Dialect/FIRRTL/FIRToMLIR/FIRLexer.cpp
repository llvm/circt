//===- FIRLexer.cpp - .fir file lexer implementation ----------------------===//
//
// This implements a .fir file lexer.
//
//===----------------------------------------------------------------------===//

#include "FIRLexer.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace spt;
using namespace firrtl;
using namespace mlir;
using llvm::SMLoc;
using llvm::SMRange;
using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// FIRToken
//===----------------------------------------------------------------------===//

SMLoc FIRToken::getLoc() const {
  return SMLoc::getFromPointer(spelling.data());
}

SMLoc FIRToken::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange FIRToken::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

/// Return true if this is one of the keyword token kinds (e.g. kw_wire).
bool FIRToken::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "FIRTokenKinds.def"
  }
}

//===----------------------------------------------------------------------===//
// FIRLexer
//===----------------------------------------------------------------------===//

FIRLexer::FIRLexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location FIRLexer::translateLocation(llvm::SMLoc loc) {
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  auto *buffer = sourceMgr.getMemoryBuffer(mainFileID);

  return FileLineColLoc::get(buffer->getBufferIdentifier(), lineAndColumn.first,
                             lineAndColumn.second, context);
}

/// Emit an error message and return a FIRToken::error token.
FIRToken FIRLexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(translateLocation(SMLoc::getFromPointer(loc)), message);
  return formToken(FIRToken::error, loc);
}

/// Return the indentation level of the specified token.
unsigned FIRLexer::getIndentation(const FIRToken &tok) const {
  // Count the number of horizontal whitespace characters before the token.
  auto *bufStart = curBuffer.begin();

  auto isHorizontalWS = [](char c) -> bool { return c == ' ' || c == '\t'; };
  auto isVerticalWS = [](char c) -> bool {
    return c == '\n' || c == '\r' || c == '\f' || c == '\v';
  };

  unsigned indent = 0;
  const auto *ptr = (const char *)tok.getSpelling().data();
  while (ptr != bufStart && !isVerticalWS(ptr[-1]) && isHorizontalWS(ptr[-1]))
    --ptr, ++indent;

  return indent;
}

//===----------------------------------------------------------------------===//
// Lexer Implementation Methods
//===----------------------------------------------------------------------===//

FIRToken FIRLexer::lexToken() {
  while (true) {
    const char *tokStart = curPtr;
    switch (*curPtr++) {
    default:
      // Handle identifiers.
      if (isalpha(curPtr[-1]))
        return lexIdentifierOrKeyword(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that llvm::MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(FIRToken::eof, tokStart);

      LLVM_FALLTHROUGH; // Treat as whitespace.

    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;

    case '_':
      // Handle identifiers.
      return lexIdentifierOrKeyword(tokStart);

    case '.':
      return formToken(FIRToken::period, tokStart);
    case ',':
      return formToken(FIRToken::comma, tokStart);
    case ':':
      return formToken(FIRToken::colon, tokStart);
    case '(':
      return formToken(FIRToken::l_paren, tokStart);
    case ')':
      return formToken(FIRToken::r_paren, tokStart);
    case '{':
      return formToken(FIRToken::l_brace, tokStart);
    case '}':
      return formToken(FIRToken::r_brace, tokStart);
    case '[':
      return formToken(FIRToken::l_square, tokStart);
    case ']':
      return formToken(FIRToken::r_square, tokStart);
    case '<':
      if (*curPtr == '-') {
        ++curPtr;
        return formToken(FIRToken::less_minus, tokStart);
      }
      if (*curPtr == '=') {
        ++curPtr;
        return formToken(FIRToken::less_equal, tokStart);
      }
      return formToken(FIRToken::less, tokStart);
    case '>':
      return formToken(FIRToken::greater, tokStart);
    case '=':
      return formToken(FIRToken::equal, tokStart);
    case '?':
      return formToken(FIRToken::question, tokStart);
    case '@':
      if (*curPtr == '[')
        return lexFileInfo(tokStart);
      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case ';':
      skipComment();
      continue;

    case '"':
      return lexString(tokStart);

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
FIRToken FIRLexer::lexFileInfo(const char *tokStart) {
  while (1) {
    switch (*curPtr++) {
    case ']': // This is the end of the fileinfo literal.
      return formToken(FIRToken::fileinfo, tokStart);
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
      LLVM_FALLTHROUGH;
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

/// Lex an identifier or keyword that starts with a letter.
///
///   LegalStartChar ::= [a-zA-Z_]
///   LegalIdChar    ::= LegalStartChar | [0-9] | '$'
//
///   Id ::= LegalStartChar (LegalIdChar)*
///
FIRToken FIRLexer::lexIdentifierOrKeyword(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_$]*
  while (isalpha(*curPtr) || isdigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$')
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);

  // Check to see if this is a 'primop', which is an identifier juxtaposed with
  // a '(' character.
  if (*curPtr == '(') {
    FIRToken::Kind kind = llvm::StringSwitch<FIRToken::Kind>(spelling)
#define TOK_LPKEYWORD(SPELLING, NUMEXP, NUMCST)                                \
  .Case(#SPELLING, FIRToken::lp_##SPELLING)
#include "FIRTokenKinds.def"
                              .Default(FIRToken::identifier);
    if (kind != FIRToken::identifier) {
      ++curPtr;
      return FIRToken(kind, tokStart);
    }
  }

  // Check to see if this identifier is a keyword.
  FIRToken::Kind kind = llvm::StringSwitch<FIRToken::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, FIRToken::kw_##SPELLING)
#include "FIRTokenKinds.def"
                            .Default(FIRToken::identifier);

  return FIRToken(kind, spelling);
}

/// Skip a comment line, starting with a ';' and going to end of line.
void FIRLexer::skipComment() {
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
      LLVM_FALLTHROUGH;
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// StringLit      ::= '"' UnquotedString? '"'
/// UnquotedString ::= ( '\\\'' | '\\"' | ~[\r\n] )+?
///
FIRToken FIRLexer::lexString(const char *tokStart) {
  while (1) {
    switch (*curPtr++) {
    case '"': // This is the end of the string literal.
      return formToken(FIRToken::string, tokStart);
    case '\\':
      // Ignore escaped '"'
      if (*curPtr == '"')
        ++curPtr;
      break;
    case 0:
      // This could be the end of file in the middle of the string.  If so
      // emit an error.
      if (curPtr - 1 != curBuffer.end())
        break;
      LLVM_FALLTHROUGH;
    case '\n': // Vertical whitespace isn't allowed in a string.
    case '\v':
    case '\f':
      return emitError(tokStart, "unterminated string");
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
FIRToken FIRLexer::lexNumber(const char *tokStart) {
  assert(isdigit(curPtr[-1]));

  while (isdigit(*curPtr))
    ++curPtr;

  return formToken(FIRToken::integer, tokStart);
}
