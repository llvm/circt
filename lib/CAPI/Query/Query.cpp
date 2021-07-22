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
  return new Range(start, end);
}

CirctQueryValueType CirctQueryNewValueType(CirctQueryValueTypeType typeType, CirctQueryPortType port, size_t count, ...) {
  va_list va;
  va_start(va, count);
  auto widths = std::vector<Range>();
  for (size_t i = 0; i < count; i++) {
    widths.push_back(*va_arg(va, CirctQueryWidthRange));
  }
  va_end(va);

  return new ValueType((ValueTypeType) typeType, (PortType) port, Ranges(widths));
}

CirctQueryValueType CirctQueryNewValueTypeArray(CirctQueryValueTypeType typeType, CirctQueryPortType port, size_t count, CirctQueryWidthRange ranges[]) {
  auto widths = std::vector<Range>();
  for (size_t i = 0; i < count; i++) {
    widths.push_back(*ranges[i]);
  }

  return new ValueType((ValueTypeType) typeType, (PortType) port, Ranges(widths));
}

void CirctQueryDeleteValueType(CirctQueryValueType type) {
  delete type;
}

void CirctQueryDeleteWidthRange(CirctQueryWidthRange range) {
  delete range;
}

CirctQueryFilterNode CirctQueryNewGlobFilter() {
  CirctQueryFilterNode node = new FilterNode;
  *node = FilterNode::newGlob();
  return node;
}

CirctQueryFilterNode CirctQueryNewGlobFilterWithType(CirctQueryValueType type) {
  CirctQueryFilterNode node = new FilterNode;
  *node = FilterNode::newGlob(*type);
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

CirctQueryFilterNode CirctQueryNewLiteralFilterWithType(char *literal, CirctQueryValueType type) {
  CirctQueryFilterNode node = new FilterNode;
  auto s = std::string(literal);
  *node = FilterNode::newLiteral(s, *type);
  return node;
}

CirctQueryFilterNode CirctQueryNewRegexFilter(char *regex) {
  CirctQueryFilterNode node = new FilterNode;
  auto s = std::string(regex);
  *node = FilterNode::newRegex(s);
  return node;
}

CirctQueryFilterNode CirctQueryNewRegexFilterWithType(char *regex, CirctQueryValueType type) {
  CirctQueryFilterNode node = new FilterNode;
  auto s = std::string(regex);
  *node = FilterNode::newRegex(s, *type);
  return node;
}

void CirctQueryDeleteFilterNode(CirctQueryFilterNode node) {
  delete node;
}

CirctQueryFilter CirctQueryNewFilterArray(size_t count, CirctQueryFilterNode *nodes) {
  auto ns = std::vector<FilterNode>();
  for (size_t i = 0; i < count; i++) {
    ns.push_back(*nodes[i]);
  }
  CirctQueryFilter filter = new Filter(ns);
  return filter;
}

CirctQueryFilter CirctQueryNewFilter(size_t count, ...) {
  va_list va;
  va_start(va, count);

  auto ns = std::vector<FilterNode>();
  for (size_t i = 0; i < count; i++) {
    auto *node = va_arg(va, CirctQueryFilterNode);
    ns.push_back(*node);
  }

  CirctQueryFilter filter = new Filter(ns);
  va_end(va);
  return filter;
}

void CirctQueryDeleteFilter(CirctQueryFilter filter) {
  delete filter;
}
