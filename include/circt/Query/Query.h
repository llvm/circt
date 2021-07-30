#include <regex>
#include <string>
#include <vector>

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVVisitors.h"

namespace circt {
namespace query {

enum class FilterType {
  UNSET,
  REGEX,
  LITERAL,
  GLOB,
  RECURSIVE_GLOB
};

enum class PortType {
  NONE    = 1,
  INPUT   = 2,
  OUTPUT  = 4,
};

bool operator &(PortType a, PortType b);
PortType operator |(PortType a, PortType b);

class Range {
public:
  Range(size_t start, size_t end) : start (start), end (end) { }
  bool contains(size_t n) { return start <= n && n <= end; }

  size_t start;
  size_t end;
};

class Ranges {
public:
  Ranges() : ranges (std::vector<Range>()) { }
  Ranges(std::vector<Range> ranges) : ranges (ranges) { }

  bool contains(size_t n) {
    if (ranges.empty()) {
      return true;
    }

    for (auto range : ranges) {
      if (range.contains(n)) {
        return true;
      }
    }

    return false;
  }
private:
  std::vector<Range> ranges;
};

class ValueTypeType {
public:
  ValueTypeType(StringRef dialect, StringRef opName) : dialect (dialect), opName (opName) { }

  bool operationIsOfType(Operation *op) {
    bool nameMatches = true, dialectMatches = true;
    if (!dialect.empty()) {
      auto name = op->getDialect()->getNamespace();
      dialectMatches = name == dialect;
    }

    if (!opName.empty()) {
      auto name = op->getName().stripDialect();
      nameMatches = name == opName;
    }

    return nameMatches && dialectMatches;
  }

private:
  StringRef dialect;
  StringRef opName;
};

class ValueType {
public:
  ValueType() : types (std::vector<ValueTypeType>()), port (PortType::NONE), widths (Ranges()) {
    types.push_back(ValueTypeType(StringRef("hw"), StringRef("module")));
    types.push_back(ValueTypeType(StringRef("hw"), StringRef("instance")));
  }

  ValueType(std::vector<ValueTypeType> types, PortType port, Ranges widths) : types (types), port (port), widths (widths) { }

  bool operationIsOfType(Operation *op) {
    for (auto &type : types) {
      if (type.operationIsOfType(op)) {
        return true;
      }
    }

    return false;
  }

  PortType getPort() {
    return port;
  }

  bool containsWidth(size_t width) {
    return widths.contains(width);
  }

private:
  std::vector<ValueTypeType> types;
  PortType port;
  Ranges widths;
};

class Filter;

class FilterNode {
public:
  FilterNode() : tag (FilterType::UNSET), type (ValueType()), regex (std::regex()), literal (std::string()) { }
  FilterNode(const FilterNode &other);
  FilterNode &operator =(const FilterNode &other);

  static FilterNode newGlob();
  static FilterNode newGlob(ValueType type);
  static FilterNode newRecursiveGlob();
  static FilterNode newLiteral(std::string &literal);
  static FilterNode newLiteral(std::string &literal, ValueType type);
  static FilterNode newRegex(std::string &regex);
  static FilterNode newRegex(std::string &regex, ValueType type);

private:
  FilterType tag;
  ValueType type;
  std::regex regex;
  std::string literal;

  friend class Filter;
  friend std::vector<mlir::Operation *> filterAsVector(Filter &filter, Operation *root);
};

class Filter {
public:
  Filter() : nodes (std::vector<FilterNode>()) { }
  Filter(std::vector<FilterNode> nodes) : nodes (nodes) { }

  size_t size() { return nodes.size(); }

private:
  std::vector<FilterNode> nodes;

  friend Filter parseFilter(std::string &filter);
  friend std::vector<mlir::Operation *> filterAsVector(Filter &filter, Operation *root);
};

// TODO: filterAsIterator()
std::vector<Operation *> filterAsVector(Filter &filter, Operation *root);
std::vector<Operation *> filterAsVector(Filter &filter, std::vector<Operation *> results);

} /* namespace query */
} /* namespace circt */

