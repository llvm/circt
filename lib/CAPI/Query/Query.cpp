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
  return {new GlobFilterType};
}

CirctQueryFilterType circtQueryNewRecursiveGlobFilterType() {
  return {new RecursiveGlobFilterType};
}

CirctQueryFilterType circtQueryNewLiteralFilterType(char *literal) {
  std::string s(literal);
  return {new LiteralFilterType(s)};
}

CirctQueryFilterType circtQueryNewRegexFilterType(char *regex) {
  std::string s(regex);
  return {new RegexFilterType(s)};
}

void circtQueryDeleteFilterType(CirctQueryFilterType type) {
  delete (FilterType *)type.ptr;
}

CirctQueryFilterData circtQueryNewFilterData(MlirOperation root) {
  return {new FilterData(unwrap(root))};
}

void circtQueryDeleteFilterData(CirctQueryFilterData data) {
  delete (FilterData *)data.ptr;
}

CirctQueryFilter circtQueryNewAttributeFilter(char *key,
                                              CirctQueryFilterType type) {
  std::string s(key);
  return {new AttributeFilter(s, (FilterType *)type.ptr)};
}

CirctQueryFilter circtQueryNewNameFilter(CirctQueryFilterType type) {
  return {new NameFilter((FilterType *)type.ptr)};
}

CirctQueryFilter circtQueryNewOperatorFilter(CirctQueryFilterType type) {
  return {new OpFilter((FilterType *)type.ptr)};
}

CirctQueryFilter circtQueryNewAndFilter(size_t count,
                                        CirctQueryFilter *filters) {
  std::vector<Filter *> fs;
  for (size_t i = 0; i < count; i++) {
    fs.push_back((Filter *)filters[i].ptr);
  }

  return {new AndFilter(fs)};
}

CirctQueryFilter circtQueryNewOrFilter(size_t count,
                                       CirctQueryFilter *filters) {
  std::vector<Filter *> fs;
  for (size_t i = 0; i < count; i++) {
    fs.push_back((Filter *)filters[i].ptr);
  }

  return {new OrFilter(fs)};
}

CirctQueryFilter circtQueryNewInstanceFilter(CirctQueryFilter filter,
                                             CirctQueryFilter child) {
  return {new InstanceFilter((Filter *)filter.ptr, (Filter *)child.ptr)};
}

CirctQueryFilter circtQueryNewUsageFilter(CirctQueryFilter filter) {
  return {new UsageFilter((Filter *)filter.ptr)};
}

CirctQueryFilter circtQueryCloneFilter(CirctQueryFilter filter) {
  return {((Filter *)filter.ptr)->clone()};
}

void circtQueryDeleteFilter(CirctQueryFilter filter) {
  delete (Filter *)filter.ptr;
}

CirctQueryFilterResult circtQueryFilterFromRoot(CirctQueryFilter filter,
                                                MlirOperation root,
                                                CirctQueryFilterData data) {
  return {new std::vector<Operation *>(
      ((Filter *)filter.ptr)->filter(unwrap(root), *(FilterData *)data.ptr))};
}

CirctQueryFilterResult circtQueryFilterFromResult(CirctQueryFilter filter,
                                                  CirctQueryFilterResult result,
                                                  CirctQueryFilterData data) {
  return {new std::vector<Operation *>(
      ((Filter *)filter.ptr)
          ->filter(*(std::vector<Operation *> *)result.ptr,
                   *(FilterData *)data.ptr))};
}

MlirOperation circtQueryGetFromFilterResult(CirctQueryFilterResult result,
                                            size_t index) {
  auto &vec = *(std::vector<Operation *> *)result.ptr;
  if (index < vec.size()) {
    return wrap(vec[index]);
  }

  return {nullptr};
}

void circtQueryDeleteFilterResult(CirctQueryFilterResult result) {
  delete (std::vector<Operation *> *)result.ptr;
}

CirctQueryAttributeDump circtQueryDumpAttributes(CirctQueryFilterResult result,
                                                 size_t count, char **filter) {
  auto &results = *(std::vector<Operation *> *)result.ptr;
  llvm::StringRef filters[count];

  for (size_t i = 0; i < count; ++i) {
    filters[i] = llvm::StringRef(filter[i]);
  }
  auto array = llvm::ArrayRef<llvm::StringRef>(filters, count);

  auto dump = dumpAttributes(results, array);
  return {new std::vector(dump)};
}

CirctQueryOperationAttributesPair
circtQueryGetFromAttributeDumpByOp(CirctQueryAttributeDump dump,
                                   MlirOperation op) {
  auto &map = *(op_attr_map *)dump.ptr;
  for (auto &elem : map) {
    if (elem.first == unwrap(op)) {
      return {.op = op, .map = {&elem.second}};
    }
  }

  return {{nullptr}, {nullptr}};
}

CirctQueryOperationAttributesPair
circtQueryGetFromAttributeDumpByIndex(CirctQueryAttributeDump dump, size_t i) {
  auto &map = *(op_attr_map *)dump.ptr;
  if (i < map.size()) {
    return {.op = wrap(map[i].first), .map = {&map[i].second}};
  }

  return {{nullptr}, {nullptr}};
}

bool circtQueryIsOperationAttributePairNull(
    CirctQueryOperationAttributesPair pair) {
  return pair.op.ptr == nullptr || pair.map.ptr == nullptr;
}

CirctQueryIdentifierAttributePair circtQueryGetFromOperationAttributePairByKey(
    CirctQueryOperationAttributesPair pair, char *key) {
  auto &map = *(attr_map *)pair.map.ptr;
  for (auto &elem : map) {
    if (elem.first == key) {
      return {.ident = wrap(elem.first), .attr = wrap(elem.second)};
    }
  }

  return {{nullptr, 0}, {nullptr}};
}

CirctQueryIdentifierAttributePair
circtQueryGetFromOperationAttributePairByIndex(
    CirctQueryOperationAttributesPair pair, size_t i) {
  auto &map = *(attr_map *)pair.map.ptr;
  if (i < map.size()) {
    return {.ident = wrap(map[i].first), .attr = wrap(map[i].second)};
  }

  return {{nullptr, 0}, {nullptr}};
}

bool circtQueryIsIdentifierAttributePairNull(
    CirctQueryIdentifierAttributePair pair) {
  return pair.ident.data == nullptr || pair.attr.ptr == nullptr;
}

void circtQueryDeleteAttributeDump(CirctQueryAttributeDump dump) {
  delete (op_attr_map *)dump.ptr;
}
