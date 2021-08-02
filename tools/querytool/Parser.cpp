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


Filter parseValue(Lexer &lexer, bool &errored) {
  auto state = lexer.pushState();

  Token token;
  switch ((token = lexer.next()).getType()) {
    case TokenType::GLOB_TOKEN:
      return NameFilter(GlobFilterType());

    case TokenType::DOUBLE_GLOB_TOKEN:
      return NameFilter(RecursiveGlobFilterType());

    case TokenType::LITERAL_TOKEN: {
      auto literal = token.getStringFromSpan(lexer.getSource()).str();
      return NameFilter(LiteralFilterType(literal));
    }

    case TokenType::REGEX_TOKEN: {
      auto regex = token.getStringFromSpan(lexer.getSource()).str();
      return NameFilter(RegexFilterType(regex));
    }

    default:
      errored = true;
      lexer.popState(state);
      return Filter();
  }
}

Filter parseInstances(Lexer &lexer, bool &errored) {
  auto filters = infix<Filter>(parseValue, TokenType::AND_TOKEN, lexer, errored);

  if (errored) {
    return Filter();
  }

  auto result = filters.back();
  filters.pop_back();
  while (!filters.empty()) {
    result = InstanceFilter(filters.back(), result);
    filters.pop_back();
  }

  return result;
}

Filter parseAnd(Lexer &lexer, bool &errored) {
  auto filters = infix<Filter>(parseInstances, TokenType::AND_TOKEN, lexer, errored);

  if (errored) {
    return Filter();
  }

  if (filters.size() == 1) {
    return filters[0];
  }

  return AndFilter(filters);
}

Filter parseOr(Lexer &lexer, bool &errored) {
  auto filters = infix<Filter>(parseAnd, TokenType::OR_TOKEN, lexer, errored);

  if (errored) {
    return Filter();
  }

  if (filters.size() == 1) {
    return filters[0];
  }

  return OrFilter(filters);
}

Filter parse(llvm::StringRef source, bool &errored) {
  Lexer lexer(source);
  errored = false;
  return parseOr(lexer, errored);
}

} /* namespace parser */
} /* namespace querytool */
