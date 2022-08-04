//===- EmissionPattern.h - Emission Pattern Base and Utility --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares the emission pattern base and utility classes.
//
//===----------------------------------------------------------------------===//

#ifndef EMISSION_PATTERN_H
#define EMISSION_PATTERN_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include <any>
namespace circt {
namespace ExportSystemC {

class EmissionPrinter;

template <typename T>
class Flag {
public:
  Flag<T>(StringRef name, T defaultValue) {
    this->name = name;
    this->defaultValue = defaultValue;
  }

  StringRef getName() const { return name; }
  T getDefault() const { return defaultValue; }

private:
  StringRef name;
  T defaultValue;
};

class EmissionConfig {
public:
  EmissionConfig() {}

  template <typename T>
  void set(StringRef flag, T value) {
    flags[flag] = value;
  }

  template <typename T>
  T get(Flag<T> flag) {
    if (!flags.count(flag.getName()))
      return flag.getDefault();
    return std::any_cast<T>(flags[flag.getName()]);
  }

private:
  DenseMap<StringRef, std::any> flags;
};

// source: https://en.cppreference.com/w/cpp/language/operator_precedence
// lower number == higher precedence
enum class Precedence {
  LIT = 0,
  VAR = 0,
  SCOPE_RESOLUTION = 1,
  POSTFIX_INC = 2,
  POSTFIX_DEC = 2,
  FUNCTIONAL_CAST = 2,
  FUNCTION_CALL = 2,
  SUBSCRIPT = 2,
  MEMBER_ACCESS = 2,
  PREFIX_INC = 3,
  PREFIX_DEC = 3,
  NOT = 3,
  CAST = 3,
  DEREFERENCE = 3,
  ADDRESS_OF = 3,
  SIZEOF = 3,
  NEW = 3,
  DELETE = 3,
  POINTER_TO_MEMBER = 4,
  MUL = 5,
  DIV = 5,
  MOD = 5,
  ADD = 6,
  SUB = 6,
  SHL = 7,
  SHR = 7,
  RELATIONAL = 9,
  EQUALITY = 10,
  BITWISE_AND = 11,
  BITWISE_XOR = 12,
  BITWISE_OR = 13,
  LOGICAL_AND = 14,
  LOGICAL_OR = 15,
  TERNARY = 16,
  THROW = 16,
  ASSIGN = 16,
  COMMA = 17
};

class EmissionResult {
public:
  EmissionResult();
  EmissionResult(mlir::StringRef expression, Precedence precedence);

  bool failed() { return isFailure; }
  Precedence getExpressionPrecedence() { return precedence; }
  std::string &getExpressionString() { return expression; }

private:
  bool isFailure;
  Precedence precedence;
  std::string expression;
};

struct EmissionPattern {
  EmissionPattern() {}

  virtual ~EmissionPattern() = default;

  virtual bool match(mlir::Operation *op, EmissionConfig &config) = 0;
  virtual EmissionResult getExpression(mlir::Value value,
                                       EmissionConfig &config,
                                       EmissionPrinter &p) = 0;
  virtual mlir::LogicalResult emitStatement(mlir::Operation *op,
                                            EmissionConfig &config,
                                            EmissionPrinter &p) = 0;

  /// This method provides a convenient interface for creating and initializing
  /// derived rewrite patterns of the given type `T`.
  template <typename E>
  static std::unique_ptr<E> create() {
    std::unique_ptr<E> pattern = std::make_unique<E>();
    return pattern;
  }
};

class EmissionPatternSet {
public:
  EmissionPatternSet(std::vector<std::unique_ptr<EmissionPattern>> &patterns);

  template <typename... Es>
  void add() {
    (void)std::initializer_list<int>{0, (addImpl<Es>(), 0)...};
  }

private:
  template <typename E>
  std::enable_if_t<std::is_base_of<EmissionPattern, E>::value> addImpl() {
    std::unique_ptr<E> pattern = EmissionPattern::create<E>();
    patterns.emplace_back(std::move(pattern));
  }

private:
  std::vector<std::unique_ptr<EmissionPattern>> &patterns;
};

} // namespace ExportSystemC
} // namespace circt

#endif // EMISSION_PATTERN_H
