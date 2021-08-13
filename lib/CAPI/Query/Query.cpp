//===- Query.cpp - C Interface to Query -----------------------------------===//
//
//  Implements a C Interface for the Query API.
//
//===----------------------------------------------------------------------===//

#include <stdarg.h>
#include <stdlib.h>
#include <string>

#include "circt-c/Query.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace circt;
using namespace query;

CirctQueryFilterType circtQueryNewGlobFilterType() {
  return { new GlobFilterType };
}

CirctQueryFilterType circtQueryNewRecursiveGlobFilterType() {
  return { new RecursiveGlobFilterType };
}

CirctQueryFilterType circtQueryNewLiteralFilterType(char *literal) {
  std::string s(literal);
  return { new LiteralFilterType(s) };
}

CirctQueryFilterType circtQueryNewRegexFilterType(char *regex) {
  std::string s(regex);
  return { new RegexFilterType(s) };
}

void circtQueryDeleteFilterType(CirctQueryFilterType type) {
  delete (FilterType *) type.ptr;
}

CirctQueryFilterData circtQueryNewFilterData(MlirOperation root) {
  return { new FilterData(unwrap(root)) };
}

void circtQueryDeleteFilterData(CirctQueryFilterData data) {
  delete (FilterData *) data.ptr;
}

CirctQueryFilter circtQueryNewAttributeFilter(char *key, CirctQueryFilterType type) {
  std::string s(key);
  return { new AttributeFilter(s, (FilterType *) type.ptr) };
}

CirctQueryFilter circtQueryNewNameFilter(CirctQueryFilterType type) {
  return { new NameFilter((FilterType *) type.ptr) };
}

CirctQueryFilter circtQueryNewOperatorFilter(CirctQueryFilterType type) {
  return { new OpFilter((FilterType *) type.ptr) };
}

CirctQueryFilter circtQueryNewAndFilter(size_t count, CirctQueryFilter *filters) {
  std::vector<Filter *> fs;
  for (size_t i = 0; i < count; i++) {
    fs.push_back((Filter *) filters[i].ptr);
  }

  return { new AndFilter(fs) };
}

CirctQueryFilter circtQueryNewOrFilter(size_t count, CirctQueryFilter *filters) {
  std::vector<Filter *> fs;
  for (size_t i = 0; i < count; i++) {
    fs.push_back((Filter *) filters[i].ptr);
  }

  return { new OrFilter(fs) };
}

CirctQueryFilter circtQueryNewInstanceFilter(CirctQueryFilter filter, CirctQueryFilter child) {
  return { new InstanceFilter((Filter *) filter.ptr, (Filter *) child.ptr) };
}

CirctQueryFilter circtQueryNewUsageFilter(CirctQueryFilter filter) {
  return { new UsageFilter((Filter *) filter.ptr) };
}

CirctQueryFilter circtQueryCloneFilter(CirctQueryFilter filter) {
  return { ((Filter *) filter.ptr)->clone() };
}

void circtQueryDeleteFilter(CirctQueryFilter filter) {
  delete (Filter *) filter.ptr;
}

CirctQueryFilterResult circtQueryFilterFromRoot(CirctQueryFilter filter, MlirOperation root, CirctQueryFilterData data) {
  return { new std::vector<Operation *>(((Filter *) filter.ptr)->filter(unwrap(root), *(FilterData *) data.ptr)) };
}

CirctQueryFilterResult circtQueryFilterFromResult(CirctQueryFilter filter, CirctQueryFilterResult result, CirctQueryFilterData data) {
  return { new std::vector<Operation *>(((Filter *) filter.ptr)->filter(*(std::vector<Operation *> *) result.ptr, *(FilterData *) data.ptr)) };
}

MlirOperation circtQueryGetFromFilterResult(CirctQueryFilterResult result, size_t index) {
  auto &vec = *(std::vector<Operation *> *) result.ptr;
  if (index < vec.size()) {
    return wrap(vec[index]);
  }

  return { nullptr };
}

void circtQueryDeleteFilterResult(CirctQueryFilterResult result) {
  delete (std::vector<Operation *> *) result.ptr;
}

CirctQueryAttributeDump circtQueryDumpAttributes(CirctQueryFilterResult result, size_t count, char **filter) {
  auto &results = *(std::vector<Operation *> *) result.ptr;
  std::vector<std::string> filters;

  for (size_t i = 0; i < count; ++i) {
    filters.push_back(std::string(filter[i]));
  }

  auto dump = dumpAttributes(results, filters);
  return {new std::vector(dump)};
}

CirctQueryOperationAttributesPair circtQueryGetFromAttributeDump(CirctQueryAttributeDump dump, size_t i) {
  auto &vec = *(std::vector<std::pair<Operation *, std::vector<Attribute>>> *) dump.ptr;
  if (i < vec.size()) {
    return {
      wrap(vec[i].first),
      {&vec[i].second}
    };
  }

  return {{nullptr}, {nullptr}};
}

bool circtQueryIsOperationAttributePairNull(CirctQueryOperationAttributesPair pair) {
  return pair.op.ptr == nullptr || pair.list.ptr == nullptr;
}

MlirAttribute circtQueryGetFromOperationAttributePair(CirctQueryOperationAttributesPair pair, size_t i) {
  auto &vec = *(std::vector<Attribute> *) pair.list.ptr;
  if (i < vec.size()) {
    return wrap(vec[i]);
  }

  return {nullptr};
}

void circtQueryDeleteAttributeDump(CirctQueryAttributeDump dump) {
  delete (std::vector<std::pair<Operation *, std::vector<Attribute>>> *) dump.ptr;
}
