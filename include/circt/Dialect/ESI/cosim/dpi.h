//===- dpi.h - DPI function C++ declarations --------------------*- C++ -*-===//
//
// Originally generated from 'Cosim_DpiPkg.sv' by an RTL simulator. All these
// functions are called from RTL. Some of the funky types are produced by the
// RTL simulators when it did the conversion.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_COSIM_DPI_H
#define CIRCT_DIALECT_ESI_COSIM_DPI_H

#include "external/dpi/svdpi.h"

#ifdef _WIN32
#define DPI extern "C" __declspec(dllexport)
#else
#define DPI extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
/// Register an endpoint.
extern int sv2cCosimserverEpRegister(int endpointId, long long sendTypeId,
                                     int sendTypeSize, long long recvTypeId,
                                     int recvTypeSize);
/// Try to get a message from a client.
extern int sv2cCosimserverEpTryGet(unsigned int endpointId,
                                   // NOLINTNEXTLINE(misc-misplaced-const)
                                   const svOpenArrayHandle data,
                                   unsigned int *sizeBytes);
/// Send a message to a client.
extern int sv2cCosimserverEpTryPut(unsigned int endpointId,
                                   // NOLINTNEXTLINE(misc-misplaced-const)
                                   const svOpenArrayHandle data, int dataLimit);

/// Start the server. Not required as the first endpoint registration will do
/// this. Provided if one wants to start the server early.
extern int sv2cCosimserverInit();
/// Shutdown the RPC server.
extern void sv2cCosimserverFinish();
#ifdef __cplusplus
}

#endif

#endif
