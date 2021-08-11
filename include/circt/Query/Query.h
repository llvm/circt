#include <iostream>
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

class FilterType {
public:
  virtual ~FilterType() { }

  virtual bool valueMatches(std::string &value) { return false; }
  virtual bool addSelf() { return false; }
  virtual FilterType *clone() { return nullptr; }
};

class GlobFilterType : public FilterType {
public:
  GlobFilterType() { }

  bool valueMatches(std::string &value) override { return true; }
  FilterType *clone() override { return new GlobFilterType; }
};

class RecursiveGlobFilterType : public FilterType {
public:
  RecursiveGlobFilterType() { }

  bool valueMatches(std::string &value) override { return true; }
  bool addSelf() override { return true; }
  FilterType *clone() override { return new RecursiveGlobFilterType; }
};

class LiteralFilterType : public FilterType {
public:
  LiteralFilterType(std::string &literal) : literal (literal) { }

  bool valueMatches(std::string &value) override { return value == literal; }
  FilterType *clone() override { return new LiteralFilterType(literal); }

private:
  std::string literal;
};

class RegexFilterType : public FilterType {
public:
  RegexFilterType(std::string &regex) : regex (std::regex(regex)) { }
  RegexFilterType(std::regex &regex) : regex (regex) { }

  bool valueMatches(std::string &value) override { return std::regex_match(value, regex); }
  FilterType *clone() override { return new RegexFilterType(regex); }

private:
  std::regex regex;
};

class Filter {
public:
  virtual ~Filter() {
    delete type;
  }

  virtual bool matches(Operation *op) { return false; }
  virtual bool addSelf() { return type->addSelf(); }
  virtual Filter *nextFilter() { return nullptr; };
  virtual Filter *clone() { return nullptr; }
  virtual std::vector<Operation *> nextOperations(Operation *op) { std::vector<Operation *> ops; ops.push_back(op); return ops; }

  FilterType *getType() { return type; }

  std::vector<Operation *> filter(Operation *root);
  std::vector<Operation *> filter(std::vector<Operation *> &results);

protected:
  Filter(FilterType *type) : type (type) { }

  FilterType *type;
};

class AttributeFilter : public Filter {
public:
  AttributeFilter(std::string &key, FilterType *type) : Filter(type), key (key) { }

  bool matches(Operation *op) override;
  Filter *clone() override;

private:
  std::string key;
};

class NameFilter : public Filter {
public:
  NameFilter(FilterType *type) : Filter(type) { }

  bool matches(Operation *op) override;
  Filter *clone() override;
};

class OpFilter : public Filter {
public:
  OpFilter(FilterType *type) : Filter(type) { }

  bool matches(Operation *op) override;
  Filter *clone() override;
};

class AndFilter : public Filter {
public:
  AndFilter(std::vector<Filter *> &filters) : Filter(new FilterType), filters (filters) { }

  ~AndFilter() {
    for (auto *f : filters) {
      delete f;
    }
  }

  bool matches(Operation *op) override;
  Filter *clone() override;

private:
  std::vector<Filter *> filters;
};

class OrFilter : public Filter {
public:
  OrFilter(std::vector<Filter *> &filters) : Filter(new FilterType), filters (filters) { }

  ~OrFilter() {
    for (auto *f : filters) {
      delete f;
    }
  }

  bool matches(Operation *op) override;
  Filter *clone() override;

private:
  std::vector<Filter *> filters;
};

class InstanceFilter : public Filter {
public:
  InstanceFilter(Filter *filter, Filter *child) : Filter(new FilterType), filter (filter), child (child) { }
  ~InstanceFilter() {
    delete filter;
    delete child;
  }

  bool matches(Operation *op) override;
  bool addSelf() override;
  Filter *clone() override;

protected:
  Filter *nextFilter() override;

private:
  Filter *filter;
  Filter *child;
};

class UsageFilter : public Filter {
public:
  UsageFilter(Filter *filter) : Filter(new FilterType), filter (filter) { }

  bool matches(Operation *op) override { return filter->matches(op); }
  std::vector<Operation *> nextOperations(Operation *op) override;

private:
  Filter *filter;
};

std::vector<std::pair<Operation *, std::vector<Attribute>>> dumpAttributes(std::vector<Operation *> results, std::vector<std::string> filters);

} /* namespace query */
} /* namespace circt */

