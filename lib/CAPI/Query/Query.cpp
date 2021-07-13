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

CIRCTQueryFilterNode CIRCTQueryNewGlobFilter() {
  CIRCTQueryFilterNode node = (CIRCTQueryFilterNode) malloc(sizeof(FilterNode));
  *node = FilterNode::newGlob();
  return node;
}

CIRCTQueryFilterNode CIRCTQueryNewRecursiveGlobFilter() {
  CIRCTQueryFilterNode node = (CIRCTQueryFilterNode) malloc(sizeof(FilterNode));
  *node = FilterNode::newRecursiveGlob();
  return node;
}

CIRCTQueryFilterNode CIRCTQueryNewLiteralFilter(char *literal) {
  CIRCTQueryFilterNode node = (CIRCTQueryFilterNode) malloc(sizeof(FilterNode));
  auto s = std::string(literal);
  *node = FilterNode::newLiteral(s);
  return node;
}

CIRCTQueryFilterNode CIRCTQueryNewRegexFilter(char *regex) {
  CIRCTQueryFilterNode node = (CIRCTQueryFilterNode) malloc(sizeof(FilterNode));
  auto s = std::string(regex);
  *node = FilterNode::newRegex(s);
  return node;
}

CIRCTQueryFilter CIRCTQueryNewFilterArray(size_t count, CIRCTQueryFilterNode *nodes) {
  FilterNode rawNodes[count];

  for (size_t i = 0; i < count; i++) {
    rawNodes[i] = *nodes[i];
    free(nodes[i]);
  }

  CIRCTQueryFilter filter = (CIRCTQueryFilter) malloc(sizeof(Filter));
  *filter = Filter::newFilter(count, rawNodes);
  return filter;
}

CIRCTQueryFilter CIRCTQueryNewFilter(size_t count, ...) {
  va_list va;
  va_start(va, count);
  FilterNode nodes[count];

  for (size_t i = 0; i < count; i++) {
    auto *node = va_arg(va, CIRCTQueryFilterNode);
    nodes[i] = *node;
    free(node);
  }

  CIRCTQueryFilter filter = (CIRCTQueryFilter) malloc(sizeof(Filter));
  *filter = Filter::newFilter(count, nodes);
  va_end(va);
  return filter;
}
