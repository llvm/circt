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
extern int sv2cCosimserverEpRegister(int endpointId, long long sendTypeId,
                                     int sendTypeSize, long long recvTypeId,
                                     int recvTypeSize);
extern int sv2cCosimserverEpTryGet(unsigned int endpointId,
                                   const svOpenArrayHandle data,
                                   unsigned int *sizeBytes);
extern int sv2cCosimserverEpTryPut(unsigned int endpointId,
                                   const svOpenArrayHandle data, int dataLimit);
extern void sv2cCosimserverFini();
extern int sv2cCosimserverInit();
#ifdef __cplusplus
}

#endif

#endif
