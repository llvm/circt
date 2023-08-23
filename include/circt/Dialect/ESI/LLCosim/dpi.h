//===- dpi.h - DPI function C++ declarations --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Originally generated from 'Cosim_DpiPkg.sv' by an RTL simulator. All these
// functions are called from RTL. Some of the funky types are produced by the
// RTL simulators when it did the conversion.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_LLCOSIM_DPI_H
#define CIRCT_DIALECT_ESI_LLCOSIM_DPI_H

#include "external/dpi/svdpi.h"

#ifdef WIN32
#define DPI extern "C" __declspec(dllexport)
#else
#define DPI extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
/// Start the server. Provided if one wants to start the server early.
DPI int sv2cLLCosimserverInit();
/// Shutdown the RPC server.
DPI void sv2cLLCosimserverFinish();
#ifdef __cplusplus
}

#endif

#endif // CIRCT_DIALECT_ESI_LLCOSIM_DPI_H
