//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the registration for the Liberty file importer.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_IMPORTLIBERTY_H
#define CIRCT_CONVERSION_IMPORTLIBERTY_H

namespace circt::liberty {

/// Register the Liberty importer in the translation registry.
void registerImportLibertyTranslation();

} // namespace circt::liberty

#endif // CIRCT_CONVERSION_IMPORTLIBERTY_H
