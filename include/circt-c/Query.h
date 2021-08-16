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

typedef struct { void *ptr; } CirctQueryFilterType;
typedef struct { void *ptr; } CirctQueryFilterData;
typedef struct { void *ptr; } CirctQueryFilter;
typedef struct { void *ptr; } CirctQueryFilterResult;
typedef struct {
  MlirStringRef ident;
  MlirAttribute attr;
} CirctQueryIdentifierAttributePair;
typedef struct { void *ptr; } CirctQueryAttributeMap;
typedef struct {
  MlirOperation op;
  CirctQueryAttributeMap map;
} CirctQueryOperationAttributesPair;
typedef struct { void *ptr; } CirctQueryAttributeDumpIter;
typedef struct { void *ptr; } CirctQueryAttributeDump;

CirctQueryFilterType circtQueryNewGlobFilterType();
CirctQueryFilterType circtQueryNewRecursiveGlobFilterType();
CirctQueryFilterType circtQueryNewLiteralFilterType(char *literal);
CirctQueryFilterType circtQueryNewRegexFilterType(char *regex);
void circtQueryDeleteFilterType(CirctQueryFilterType type);

CirctQueryFilterData circtQueryNewFilterData(MlirOperation root);
void circtQueryDeleteFilterData(CirctQueryFilterData data);

CirctQueryFilter circtQueryNewAttributeFilter(char *key, CirctQueryFilterType type);
CirctQueryFilter circtQueryNewNameFilter(CirctQueryFilterType type);
CirctQueryFilter circtQueryNewOperatorFilter(CirctQueryFilterType type);
CirctQueryFilter circtQueryNewAndFilter(size_t count, CirctQueryFilter *filters);
CirctQueryFilter circtQueryNewOrFilter(size_t count, CirctQueryFilter *filters);
CirctQueryFilter circtQueryNewInstanceFilter(CirctQueryFilter filter, CirctQueryFilter child);
CirctQueryFilter circtQueryNewUsageFilter(CirctQueryFilter filter);
CirctQueryFilter circtQueryCloneFilter(CirctQueryFilter filter);
void circtQueryDeleteFilter(CirctQueryFilter filter);

CirctQueryFilterResult circtQueryFilterFromRoot(CirctQueryFilter filter, MlirOperation root, CirctQueryFilterData data);
CirctQueryFilterResult circtQueryFilterFromResult(CirctQueryFilter filter, CirctQueryFilterResult result, CirctQueryFilterData data);
MlirOperation circtQueryGetFromFilterResult(CirctQueryFilterResult result, size_t index);
void circtQueryDeleteFilterResult(CirctQueryFilterResult result);

CirctQueryAttributeDump circtQueryDumpAttributes(CirctQueryFilterResult result, size_t count, char **filter);
CirctQueryOperationAttributesPair circtQueryGetFromAttributeDumpByOp(CirctQueryAttributeDump dump, MlirOperation op);
CirctQueryOperationAttributesPair circtQueryGetFromAttributeDumpByIndex(CirctQueryAttributeDump dump, size_t i);
bool circtQueryIsOperationAttributePairNull(CirctQueryOperationAttributesPair pair);

CirctQueryIdentifierAttributePair circtQueryGetFromOperationAttributePairByKey(CirctQueryOperationAttributesPair pair, char *key);
CirctQueryIdentifierAttributePair circtQueryGetFromOperationAttributePairByIndex(CirctQueryOperationAttributesPair pair, size_t i);
bool circtQueryIsIdentifierAttributePairNull(CirctQueryIdentifierAttributePair pair);

void circtQueryDeleteAttributeDump(CirctQueryAttributeDump dump);


#ifdef __cplusplus
}
#endif

#endif /* CIRCT_C_QUERY_H */
