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

enum class ValueTypeType {
  MODULE    = 1,
  WIRE      = 2,
  REGISTER  = 4,
};

bool operator &(ValueTypeType a, ValueTypeType b);
ValueTypeType operator |(ValueTypeType a, ValueTypeType b);

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

class ValueType {
public:
  ValueType() : type (ValueTypeType::MODULE), port (PortType::NONE), widths (Ranges()) { }
  ValueType(ValueTypeType type, PortType port, Ranges widths) : type (type), port (port), widths (widths) { }

  ValueTypeType getType() {
    return type;
  }

  PortType getPort() {
    return port;
  }

  bool containsWidth(size_t width) {
    return widths.contains(width);
  }

private:
  ValueTypeType type;
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
  friend void matchAndAppend(FilterNode &node, Operation *module, std::vector<std::pair<Operation *, size_t>> &opStack, size_t i, std::string &name, bool &match);
};

class Filter {
public:
  Filter() : nodes (std::vector<FilterNode>()) { }
  Filter(std::string &filter);

  size_t size() { return nodes.size(); }

  static Filter newFilter(size_t count, FilterNode nodes[]);

private:
  std::vector<FilterNode> nodes;

  friend Filter parseFilter(std::string &filter);
  friend std::vector<mlir::Operation *> filterAsVector(Filter &filter, Operation *root);
  friend std::vector<mlir::Operation *> filterAsVector(Filter &filter, ModuleOp &module);
};

// TODO: filterAsIterator()
std::vector<mlir::Operation *> filterAsVector(Filter &filter, ModuleOp &module);
std::vector<mlir::Operation *> filterAsVector(Filter &filter, Operation *root);

} /* namespace query */
} /* namespace circt */

