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

void circtQueryDeleteFilter(CirctQueryFilter filter) {
  delete (Filter *) filter.ptr;
}

