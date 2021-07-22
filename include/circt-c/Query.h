//===-- circt-c/Query.h - C API for querying ----------*- C -*-===//
//
// This header declares the C interface for performing queries on MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_QUERY_H
#define CIRCT_C_QUERY_H

#include "mlir-c/IR.h"
#include "circt/Query/Query.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CIRCT_QUERY_VALUE_TYPE_TYPE_MODULE    = 1,
  CIRCT_QUERY_VALUE_TYPE_TYPE_WIRE      = 2,
  CIRCT_QUERY_VALUE_TYPE_TYPE_REGISTER  = 4,
} CirctQueryValueTypeType;

typedef enum {
  CIRCT_QUERY_PORT_TYPE_NONE    = 1,
  CIRCT_QUERY_PORT_TYPE_INPUT   = 2,
  CIRCT_QUERY_PORT_TYPE_OUTPUT  = 4,
} CirctQueryPortType;

typedef circt::query::Range             *CirctQueryWidthRange;
typedef circt::query::ValueType         *CirctQueryValueType;
typedef circt::query::FilterNode        *CirctQueryFilterNode;
typedef circt::query::Filter            *CirctQueryFilter;
typedef std::vector<mlir::Operation *>  *CirctQueryFilterResult;

CirctQueryWidthRange CirctQueryNewWidthRange(size_t start, size_t end);
CirctQueryValueType CirctQueryNewValueType(CirctQueryValueTypeType typeType, CirctQueryPortType port, size_t count, ...);
CirctQueryValueType CirctQueryNewValueTypeArray(CirctQueryValueTypeType typeType, CirctQueryPortType port, size_t count, CirctQueryWidthRange ranges[]);
void CirctQueryDeleteValueType(CirctQueryValueType type);
void CirctQueryDeleteWidthRange(CirctQueryWidthRange range);

CirctQueryFilterNode CirctQueryNewGlobFilter();
CirctQueryFilterNode CirctQueryNewGlobFilterWithType(CirctQueryValueType type);
CirctQueryFilterNode CirctQueryNewRecursiveGlobFilter();
CirctQueryFilterNode CirctQueryNewLiteralFilter(char *literal);
CirctQueryFilterNode CirctQueryNewLiteralFilterWithType(char *literal, CirctQueryValueType type);
CirctQueryFilterNode CirctQueryNewRegexFilter(char *regex);
CirctQueryFilterNode CirctQueryNewRegexFilterWithType(char *regex, CirctQueryValueType type);
void CirctQueryDeleteFilterNode(CirctQueryFilterNode node);

CirctQueryFilter CirctQueryNewFilterArray(size_t count, CirctQueryFilterNode *nodes);
CirctQueryFilter CirctQueryNewFilter(size_t count, ...);
CirctQueryFilterResult CirctQueryFilterFromRoot(CirctQueryFilter filter, MlirOperation root);
size_t CirctQueryFilterResultSize(CirctQueryFilterResult result);
MlirOperation CirctQueryGetFromFilterResult(CirctQueryFilterResult result, size_t i);
void CirctQueryDeleteFilterResult(CirctQueryFilterResult result);
void CirctQueryDeleteFilter(CirctQueryFilter filter);

#ifdef __cplusplus
}
#endif

#endif /* CIRCT_C_QUERY_H */
