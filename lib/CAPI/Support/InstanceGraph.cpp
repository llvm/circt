//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Support/InstanceGraph.h"

#include "circt/Support/InstanceGraph.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include <string>

using namespace circt;

//===----------------------------------------------------------------------===//
// C API Helpers
//===----------------------------------------------------------------------===//

ArrayRef<igraph::InstanceOpInterface> unwrap(IgraphInstancePath instancePath) {
  return ArrayRef(
      reinterpret_cast<igraph::InstanceOpInterface *>(instancePath.ptr),
      instancePath.size);
}

IgraphInstancePath wrap(ArrayRef<igraph::InstanceOpInterface> instancePath) {
  return IgraphInstancePath{
      const_cast<igraph::InstanceOpInterface *>(instancePath.data()),
      instancePath.size()};
}

//===----------------------------------------------------------------------===//
// InstancePath
//===----------------------------------------------------------------------===//

size_t igraphInstancePathSize(IgraphInstancePath instancePath) {
  return unwrap(instancePath).size();
}

MlirOperation igraphInstancePathGet(IgraphInstancePath instancePath,
                                    size_t index) {
  assert(instancePath.ptr);
  auto path = unwrap(instancePath);
  Operation *operation = path[index];
  return wrap(operation);
}
