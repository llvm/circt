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

typedef struct { void* ptr; } CirctQueryFilterType;
typedef struct { void* ptr; } CirctQueryFilter;
typedef struct { void* ptr; } CirctQueryFilterResult;
typedef struct { void* ptr; } CirctQueryAttributeList;
typedef struct {
  MlirOperation op;
  CirctQueryAttributeList list;
} CirctQueryOperationAttributesPair;
typedef struct { void* ptr; } CirctQueryAttributeDump;

CirctQueryFilterType circtQueryNewGlobFilterType();
CirctQueryFilterType circtQueryNewRecursiveGlobFilterType();
CirctQueryFilterType circtQueryNewLiteralFilterType(char *literal);
CirctQueryFilterType circtQueryNewRegexFilterType(char *regex);
void circtQueryDeleteFilterType(CirctQueryFilterType type);

CirctQueryFilter circtQueryNewAttributeFilter(char *key, CirctQueryFilterType type);
CirctQueryFilter circtQueryNewNameFilter(CirctQueryFilterType type);
CirctQueryFilter circtQueryNewOperatorFilter(CirctQueryFilterType type);
CirctQueryFilter circtQueryNewAndFilter(size_t count, CirctQueryFilter *filters);
CirctQueryFilter circtQueryNewOrFilter(size_t count, CirctQueryFilter *filters);
CirctQueryFilter circtQueryNewInstanceFilter(CirctQueryFilter filter, CirctQueryFilter child);
CirctQueryFilter circtQueryCloneFilter(CirctQueryFilter filter);
CirctQueryFilter circtQueryNewUsageFilter(CirctQueryFilter filter);
void circtQueryDeleteFilter(CirctQueryFilter filter);

CirctQueryFilterResult circtQueryFilterFromRoot(CirctQueryFilter filter, MlirOperation root);
CirctQueryFilterResult circtQueryFilterFromResult(CirctQueryFilter filter, CirctQueryFilterResult result);
MlirOperation circtQueryGetFromFilterResult(CirctQueryFilterResult result, size_t index);
void circtQueryDeleteFilterResult(CirctQueryFilterResult result);

CirctQueryAttributeDump circtQueryDumpAttributes(CirctQueryFilterResult result, size_t count, char **filter);
CirctQueryOperationAttributesPair circtQueryGetFromAttributeDump(CirctQueryAttributeDump dump, size_t i);
bool circtQueryIsOperationAttributePairNull(CirctQueryOperationAttributesPair pair);
MlirAttribute circtQueryGetFromOperationAttributePair(CirctQueryOperationAttributesPair pair, size_t i);
void circtQueryDeleteAttributeDump(CirctQueryAttributeDump dump);

#ifdef __cplusplus
}
#endif

#endif /* CIRCT_C_QUERY_H */
