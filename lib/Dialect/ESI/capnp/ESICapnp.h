//===- ESICapnp.h - ESI Cap'nProto library utilies --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI utility code which requires libcapnp and libcapnpc.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
#define CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H

#include "mlir/IR/Types.h"

namespace circt {
namespace esi {
namespace capnp {

class TypeSchema {
public:
  TypeSchema(mlir::Type);

  ssize_t size();

private:
  mlir::Type type;
};

} // namespace capnp
} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
