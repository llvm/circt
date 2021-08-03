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
      break;
    }

    auto next = fn(lexer, errored);
    if (errored) {
      return values;
    }

    values.push_back(next);
  }

  return values;
}


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
      return new NameFilter(new LiteralFilterType(literal));
    }

    case TokenType::REGEX_TOKEN: {
      auto regex = token.getStringFromSpan(lexer.getSource()).str();
      return new NameFilter(new RegexFilterType(regex));
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
