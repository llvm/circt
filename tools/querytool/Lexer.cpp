#include "Lexer.h"

namespace querytool {
namespace lexer {

Token Lexer::next() {
  if (state.vecPos < previousTokens.size()) {
    auto token = previousTokens[state.vecPos++];
    state.strPos = token.end;
    return token;
  }

  size_t start = state.strPos;
  size_t end = 0;
  size_t len = source.size();
  auto type = TokenType::INVALID;
  bool cont = true;

  if (start == len) {
    return Token(TokenType::END, start, start);
  }

  for (size_t i = start; cont && i < len; ++i) {
    char c = source[i];

    switch (type) {
      case TokenType::INVALID:
        if (start != i) {
          cont = false;
          end = i;
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
          start = i + 1;
        } else if (c == '(') {
          type = TokenType::LPAREN_TOKEN;
        } else if (c == ')') {
          type = TokenType::RPAREN_TOKEN;
        } else if (c == '=') {
          type = TokenType::EQUALS_TOKEN;
        } else if (c == '*') {
          type = TokenType::GLOB_TOKEN;
        } else if (c == ':') {
          type = TokenType::COLON_TOKEN;
        } else if (c == '&') {
          type = TokenType::AND_TOKEN;
        } else if (c == '|') {
          type = TokenType::OR_TOKEN;
        } else if (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || c == '_') {
          type = TokenType::LITERAL_TOKEN;
        } else if (c == '/') {
          type = TokenType::REGEX_TOKEN;
        }
        break;

      case TokenType::LITERAL_TOKEN:
        if (!(('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || c == '_')) {
          end = i;
          cont = false;
        }
        break;

      case TokenType::REGEX_TOKEN:
        if (i - 1 != start && source[i - 1] == '/') {
          end = i;
          cont = false;
        }
        break;

      case TokenType::GLOB_TOKEN:
        if (c == '*') {
          type = TokenType::DOUBLE_GLOB_TOKEN;
        } else {
          end = i;
          cont = false;
        }
        break;

      case TokenType::COLON_TOKEN:
        if (c == ':') {
          type = TokenType::DOUBLE_COLON_TOKEN;
        } else {
          end = i;
          cont = false;
        }
        break;

      case TokenType::LPAREN_TOKEN:
      case TokenType::RPAREN_TOKEN:
      case TokenType::EQUALS_TOKEN:
      case TokenType::DOUBLE_GLOB_TOKEN:
      case TokenType::DOUBLE_COLON_TOKEN:
      case TokenType::AND_TOKEN:
      case TokenType::OR_TOKEN:
      case TokenType::END:
        end = i;
        cont = false;
        break;
    }
  }
  if (end <= start) {
    end = len;
  }

  auto token = Token(type, start, end);
  previousTokens.push_back(token);
  state.strPos = end;
  ++state.vecPos;
  return token;
}

} /* namespace lexer */
} /* namespace querytool */
