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

typedef circt::query::FilterNode  *CIRCTQueryFilterNode;
typedef circt::query::Filter      *CIRCTQueryFilter;

CIRCTQueryFilterNode CIRCTQueryNewGlobFilter();
CIRCTQueryFilterNode CIRCTQueryNewRecursiveGlobFilter();
CIRCTQueryFilterNode CIRCTQueryNewLiteralFilter(char *literal);
CIRCTQueryFilterNode CIRCTQueryNewRegexFilter(char *regex);


CIRCTQueryFilter CIRCTQueryNewFilterArray(size_t count, CIRCTQueryFilterNode *nodes);
CIRCTQueryFilter CIRCTQueryNewFilter(size_t count, ...);

#ifdef __cplusplus
}
#endif

#endif /* CIRCT_C_QUERY_H */
