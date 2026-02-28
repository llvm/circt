//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_SUPPORT_INSTANCEGRAPH_H
#define CIRCT_C_SUPPORT_INSTANCEGRAPH_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

struct IgraphInstancePath {
  void *ptr;
  size_t size;
};

typedef struct IgraphInstancePath IgraphInstancePath;

//===----------------------------------------------------------------------===//
// InstancePath API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirOperation igraphInstancePathGet(IgraphInstancePath path,
                                                       size_t index);
MLIR_CAPI_EXPORTED size_t igraphInstancePathSize(IgraphInstancePath path);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_SUPPORT_INSTANCEPATH_H
