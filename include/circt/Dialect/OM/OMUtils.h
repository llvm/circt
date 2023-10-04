//===- OMUtils.h - OM Utility Functions -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMUTILS_H
#define CIRCT_DIALECT_OM_OMUTILS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

namespace circt::om {

struct PathElement;
class PathAttr;

/// Parse a target string of the form "Foo/bar:Bar/baz" in to a base path.
ParseResult parseBasePath(MLIRContext *context, StringRef spelling,
                          PathAttr &path);

/// Parse a target string in to a path.
/// "Foo/bar:Bar/baz:Baz>wire.a[1]"
///  |--------------|              Path
///                  |--|          Module
///                      |--|      Ref
///                          |---| Field
ParseResult parsePath(MLIRContext *context, StringRef spelling, PathAttr &path,
                      StringAttr &module, StringAttr &ref, StringAttr &field);

} // namespace circt::om

#endif // CIRCT_DIALECT_OM_OMUTILS_H
