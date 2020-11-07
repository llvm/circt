//===- dpi.h - DPI function C++ declarations -------------------*- C++ -*-===//
//
// Originally generated from 'Cosim_DpiPkg.sv'.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_COSIM_DPI_H
#define CIRCT_DIALECT_ESI_COSIM_DPI_H

#include "svdpi.h"

#ifdef _WIN32
#define DPI extern "C" __declspec(dllexport)
#else
#define DPI extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
// DPI IMPORTS
// DPI import at Cosim_DpiPkg.sv:29:16
extern int sv2cCosimserverEpRegister(int endpointId, long long esiTypeId,
                                        int typeSize);
// DPI import at Cosim_DpiPkg.sv:56:16
extern int sv2cCosimserverEpTryGet(unsigned int endpointId,
                                      const svOpenArrayHandle data,
                                      unsigned int *sizeBytes);
// DPI import at Cosim_DpiPkg.sv:45:16
extern int sv2cCosimserverEpTryPut(unsigned int endpointId,
                                      const svOpenArrayHandle data,
                                      int dataLimit);
// DPI import at Cosim_DpiPkg.sv:22:54
extern void sv2cCosimserverFini();
// DPI import at Cosim_DpiPkg.sv:19:53
extern int sv2cCosimserverInit();
#ifdef __cplusplus
}

#endif

#endif
