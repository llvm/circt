#include "Parser.h"
#include "Lexer.h"

namespace querytool {
namespace parser {

using namespace querytool::lexer;

template<class T>
std::vector<T> infix(T (*fn)(Lexer &, bool &), TokenType op, Lexer &lexer, bool &errored) {
  std::vector<T> values;

  auto first = fn(lexer, errored);
  if (errored) {
    return values;
  }

  values.push_back(first);
  while (true) {
    Token token;
    auto state = lexer.pushState();
    if ((token = lexer.next()).getType() != op) {
      lexer.popState(state);
      return values;
    }

    auto next = fn(lexer, errored);
    if (errored) {
      lexer.popState(state);
      return values;
    }

    values.push_back(next);
  }
}

Filter *parseOr(Lexer &lexer, bool &errored);

Filter *parseValue(Lexer &lexer, bool &errored) {
  auto state = lexer.pushState();

  Token token;
  switch ((token = lexer.next()).getType()) {
    case TokenType::GLOB_TOKEN:
      return new NameFilter(new GlobFilterType());

    case TokenType::DOUBLE_GLOB_TOKEN:
      return new NameFilter(new RecursiveGlobFilterType());

    case TokenType::LITERAL_TOKEN: {
      auto literal = token.getStringFromSpan(lexer.getSource()).str();

      auto state2 = lexer.pushState();
      if (lexer.next().getType() == TokenType::EQUALS_TOKEN) {
        Token token;
        switch ((token = lexer.next()).getType()) {
          case TokenType::GLOB_TOKEN:
            return new AttributeFilter(literal, new GlobFilterType());

          case TokenType::LITERAL_TOKEN: {
            auto value = token.getStringFromSpan(lexer.getSource()).str();
            return new AttributeFilter(literal, new LiteralFilterType(value));
          }

          case TokenType::REGEX_TOKEN: {
            auto value = token.getStringFromSpan(lexer.getSource());
            auto regex = value.slice(1, value.size() - 1).str();
            return new AttributeFilter(literal, new RegexFilterType(regex));
          }

          default:
            errored = true;
            lexer.popState(state);
            return nullptr;
        }
      } else {
        lexer.popState(state2);
        return new NameFilter(new LiteralFilterType(literal));
      }
    }

    case TokenType::REGEX_TOKEN: {
      auto value = token.getStringFromSpan(lexer.getSource());
      auto regex = value.slice(1, value.size() - 1).str();
      return new NameFilter(new RegexFilterType(regex));
    }

    case TokenType::LPAREN_TOKEN: {
      auto *value = parseOr(lexer, errored);
      if (errored || (token = lexer.next()).getType() != TokenType::RPAREN_TOKEN) {
        errored = true;
        lexer.popState(state);
        return nullptr;
      }

      return value;
    }

    default:
      errored = true;
      lexer.popState(state);
      return nullptr;
  }
}

Filter *parseInstances(Lexer &lexer, bool &errored) {
  auto filters = infix<Filter *>(parseValue, TokenType::DOUBLE_COLON_TOKEN, lexer, errored);

  if (errored) {
    return nullptr;
  }

  auto *result = filters.back();
  filters.pop_back();
  while (!filters.empty()) {
    result = new InstanceFilter(filters.back(), result);
    filters.pop_back();
  }

  return result;
}

Filter *parseAnd(Lexer &lexer, bool &errored) {
  auto filters = infix<Filter *>(parseInstances, TokenType::AND_TOKEN, lexer, errored);

  if (errored) {
    return nullptr;
  }

  if (filters.size() == 1) {
    return filters[0];
  }

  return new AndFilter(filters);
}

Filter *parseOr(Lexer &lexer, bool &errored) {
  auto filters = infix<Filter *>(parseAnd, TokenType::OR_TOKEN, lexer, errored);

  if (errored) {
    return nullptr;
  }

  if (filters.size() == 1) {
    return filters[0];
  }

  return new OrFilter(filters);
}

Filter *parse(llvm::StringRef source, bool &errored) {
  Lexer lexer(source);
  errored = false;
  return parseOr(lexer, errored);
}

} /* namespace parser */
} /* namespace querytool */
