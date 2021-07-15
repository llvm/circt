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

CirctQueryFilterNode CirctQueryNewGlobFilter() {
  CirctQueryFilterNode node = new FilterNode;
  *node = FilterNode::newGlob();
  return node;
}

CirctQueryFilterNode CirctQueryNewRecursiveGlobFilter() {
  CirctQueryFilterNode node = new FilterNode;
  *node = FilterNode::newRecursiveGlob();
  return node;
}

CirctQueryFilterNode CirctQueryNewLiteralFilter(char *literal) {
  CirctQueryFilterNode node = new FilterNode;
  auto s = std::string(literal);
  *node = FilterNode::newLiteral(s);
  return node;
}

CirctQueryFilterNode CirctQueryNewRegexFilter(char *regex) {
  CirctQueryFilterNode node = new FilterNode;
  auto s = std::string(regex);
  *node = FilterNode::newRegex(s);
  return node;
}

void CirctQueryDeleteFilterNode(CirctQueryFilterNode node) {
  delete node;
}

CirctQueryFilter CirctQueryNewFilterArray(size_t count, CirctQueryFilterNode *nodes) {
  FilterNode rawNodes[count];

  for (size_t i = 0; i < count; i++) {
    rawNodes[i] = *nodes[i];
  }

  CirctQueryFilter filter = new Filter;
  *filter = Filter::newFilter(count, rawNodes);
  return filter;
}

CirctQueryFilter CirctQueryNewFilter(size_t count, ...) {
  va_list va;
  va_start(va, count);
  FilterNode nodes[count];

  for (size_t i = 0; i < count; i++) {
    auto *node = va_arg(va, CirctQueryFilterNode);
    nodes[i] = *node;
  }

  CirctQueryFilter filter = new Filter;
  *filter = Filter::newFilter(count, nodes);
  va_end(va);
  return filter;
}

void CirctQueryDeleteFilter(CirctQueryFilter filter) {
  delete filter;
}
