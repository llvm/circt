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

CirctQueryWidthRange CirctQueryNewWidthRange(size_t start, size_t end) {
  return nullptr;
}

void CirctQueryDeleteValueType(CirctQueryValueType type) {
}

void CirctQueryDeleteWidthRange(CirctQueryWidthRange range) {
}

CirctQueryFilterNode CirctQueryNewGlobFilter() {
  return nullptr;
}

CirctQueryFilterNode CirctQueryNewGlobFilterWithType(CirctQueryValueType type) {
  return nullptr;
}

CirctQueryFilterNode CirctQueryNewRecursiveGlobFilter() {
  return nullptr;
}

CirctQueryFilterNode CirctQueryNewLiteralFilter(char *literal) {
  return nullptr;
}

CirctQueryFilterNode CirctQueryNewLiteralFilterWithType(char *literal, CirctQueryValueType type) {
  return nullptr;
}

CirctQueryFilterNode CirctQueryNewRegexFilter(char *regex) {
  return nullptr;
}

CirctQueryFilterNode CirctQueryNewRegexFilterWithType(char *regex, CirctQueryValueType type) {
  return nullptr;
}

void CirctQueryDeleteFilterNode(CirctQueryFilterNode node) {
}

CirctQueryFilter CirctQueryNewFilterArray(size_t count, CirctQueryFilterNode *nodes) {
  return nullptr;
}

CirctQueryFilter CirctQueryNewFilter(size_t count, ...) {
  return nullptr;
}

CirctQueryFilterResult CirctQueryFilterFromRoot(CirctQueryFilter filter, MlirOperation root) {
  return nullptr;
}

size_t CirctQueryFilterResultSize(CirctQueryFilterResult result) {
  return result->size();
}

MlirOperation CirctQueryGetFromFilterResult(CirctQueryFilterResult result, size_t i) {
  return wrap((*result)[i]);
}

void CirctQueryDeleteFilterResult(CirctQueryFilterResult result) {
  delete result;
}

void CirctQueryDeleteFilter(CirctQueryFilter filter) {
}
