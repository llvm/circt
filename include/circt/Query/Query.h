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

class Filter;

class FilterNode {
public:
  ~FilterNode();

private:
  FilterNode() : tag (FilterType::UNSET) { }

  FilterType tag;
  union {
    std::regex regex;
    std::string literal;
  };

  friend class Filter;
  friend std::vector<mlir::Operation *> filterAsVector(Filter &filter, Operation *module);
};

class Filter {
public:
  Filter(std::string &filter);

private:
  Filter() : nodes (std::vector<FilterNode>()) { }
  std::vector<FilterNode> nodes;

  friend Filter parseFilter(std::string &filter);
  friend std::vector<mlir::Operation *> filterAsVector(Filter &filter, Operation *module);
};

// TODO: filterAsIterator()
std::vector<mlir::Operation *> filterAsVector(Filter &filter);

} /* namespace query */
} /* namespace circt */

