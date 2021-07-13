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
  FilterNode() : tag (FilterType::UNSET), regex (std::regex()), literal (std::string()) { }
  FilterNode(const FilterNode &other);
  FilterNode &operator =(const FilterNode &other);

  static FilterNode newGlob();
  static FilterNode newRecursiveGlob();
  static FilterNode newLiteral(std::string &literal);
  static FilterNode newRegex(std::string &regex);

private:
  FilterType tag;
  std::regex regex;
  std::string literal;

  friend class Filter;
  friend std::vector<mlir::Operation *> filterAsVector(Filter &filter, ModuleOp &module);
};

class Filter {
public:
  Filter(std::string &filter);

  static Filter newFilter(size_t count, FilterNode nodes[]);

private:
  Filter() : nodes (std::vector<FilterNode>()) { }
  std::vector<FilterNode> nodes;

  friend Filter parseFilter(std::string &filter);
  friend std::vector<mlir::Operation *> filterAsVector(Filter &filter, ModuleOp &module);
};

// TODO: filterAsIterator()
std::vector<mlir::Operation *> filterAsVector(Filter &filter, ModuleOp &module);

} /* namespace query */
} /* namespace circt */

