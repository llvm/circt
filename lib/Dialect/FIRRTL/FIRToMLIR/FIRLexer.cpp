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
Location FIRLexer::getEncodedSourceLocation(llvm::SMLoc loc) {
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  auto *buffer = sourceMgr.getMemoryBuffer(mainFileID);

  return FileLineColLoc::get(buffer->getBufferIdentifier(), lineAndColumn.first,
                             lineAndColumn.second, context);
}

/// Emit an error message and return a FIRToken::error token.
FIRToken FIRLexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(getEncodedSourceLocation(SMLoc::getFromPointer(loc)),
                  message);
  return formToken(FIRToken::error, loc);
}

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
      return formToken(FIRToken::less, tokStart);
    case '>':
      return formToken(FIRToken::greater, tokStart);
    case '=':
      return formToken(FIRToken::equal, tokStart);
    case '?':
      return formToken(FIRToken::question, tokStart);
    case ';':
      skipComment();
      continue;

      //    case '"':
      //      return lexString(tokStart);

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

  // Check to see if this identifier is a keyword.
  StringRef spelling(tokStart, curPtr - tokStart);

  FIRToken::Kind kind = llvm::StringSwitch<FIRToken::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, FIRToken::kw_##SPELLING)
#include "FIRTokenKinds.def"
                            .Default(FIRToken::identifier);

  return FIRToken(kind, spelling);
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
