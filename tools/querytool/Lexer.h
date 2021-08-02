#include <cstddef>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace querytool {
namespace lexer {

enum class TokenType {
  INVALID,
  END,
  LPAREN_TOKEN,
  RPAREN_TOKEN,
  EQUALS_TOKEN,
  GLOB_TOKEN,
  DOUBLE_GLOB_TOKEN,
  LITERAL_TOKEN,
  REGEX_TOKEN,
  COLON_TOKEN,
  DOUBLE_COLON_TOKEN,
  AND_TOKEN,
  OR_TOKEN
};

class Token {
public:
  Token() : type (TokenType::INVALID), start (0), end (0) { }

  TokenType getType() {
    return type;
  }

  llvm::StringRef getStringFromSpan(llvm::StringRef source) {
    if (end < source.size()) {
      return source.slice(start, end);
    }

    return llvm::StringRef("");
  }

private:
  Token(TokenType type, size_t start, size_t end) : type (type), start (start), end (end) { }

  TokenType type;
  size_t start;
  size_t end;

  friend class Lexer;
};

class LexerState {
private:
  LexerState() : strPos (0), vecPos (0) { }

  size_t strPos;
  size_t vecPos;

  friend class Lexer;
};

class Lexer {
public:
  Lexer(llvm::StringRef source) : source (source), previousTokens (std::vector<Token>()), state (LexerState()) { }

  Token next();

  llvm::StringRef getSource() {
    return source;
  }

  LexerState pushState() {
    return state;
  }

  void popState(LexerState state) {
    this->state = state;
  }

private:
  llvm::StringRef source;
  std::vector<Token> previousTokens;
  LexerState state;
};

} /* namespace lexer */
} /* namespace querytool */

