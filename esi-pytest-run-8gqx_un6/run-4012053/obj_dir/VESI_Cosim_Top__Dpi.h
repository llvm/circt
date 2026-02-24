// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Prototypes for DPI import and export functions.
//
// Verilator includes this file in all generated .cpp files that use DPI functions.
// Manually include this file where DPI .c import functions are declared to ensure
// the C functions match the expectations of the DPI imports.

#ifndef VERILATED_VESI_COSIM_TOP__DPI_H_
#define VERILATED_VESI_COSIM_TOP__DPI_H_  // guard

#include "svdpi.h"

#ifdef __cplusplus
extern "C" {
#endif


    // DPI IMPORTS
    // DPI import at /workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_DpiPkg.sv:32:16
    extern int sv2cCosimserverEpRegister(const char* endpoint_id, const char* from_host_type_id, int from_host_type_size, const char* to_host_type_id, int to_host_type_size);
    // DPI import at /workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_DpiPkg.sv:65:16
    extern int sv2cCosimserverEpTryGet(const char* endpoint_id, const svOpenArrayHandle data, unsigned int* data_size);
    // DPI import at /workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_DpiPkg.sv:49:16
    extern int sv2cCosimserverEpTryPut(const char* endpoint_id, const svOpenArrayHandle data, int data_size);
    // DPI import at /workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_DpiPkg.sv:25:54
    extern void sv2cCosimserverFinish();
    // DPI import at /workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_DpiPkg.sv:21:51
    extern int sv2cCosimserverInit();
    // DPI import at /workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_DpiPkg.sv:80:17
    extern void sv2cCosimserverSetManifest(int esi_version, const svOpenArrayHandle compressed_manifest);

#ifdef __cplusplus
}
#endif

#endif  // guard
