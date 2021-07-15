//===-- circt-c/Query.h - C API for querying ----------*- C -*-===//
//
// This header declares the C interface for performing queries on MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef Circt_C_QUERY_H
#define Circt_C_QUERY_H

#include "mlir-c/IR.h"
#include "circt/Query/Query.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef circt::query::FilterNode  *CirctQueryFilterNode;
typedef circt::query::Filter      *CirctQueryFilter;

CirctQueryFilterNode CirctQueryNewGlobFilter();
CirctQueryFilterNode CirctQueryNewRecursiveGlobFilter();
CirctQueryFilterNode CirctQueryNewLiteralFilter(char *literal);
CirctQueryFilterNode CirctQueryNewRegexFilter(char *regex);
void CirctQueryDeleteFilterNode(CirctQueryFilterNode node);

CirctQueryFilter CirctQueryNewFilterArray(size_t count, CirctQueryFilterNode *nodes);
CirctQueryFilter CirctQueryNewFilter(size_t count, ...);

#ifdef __cplusplus
}
#endif

#endif /* Circt_C_QUERY_H */
