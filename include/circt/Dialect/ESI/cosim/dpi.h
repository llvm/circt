// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Prototypes for DPI import and export
// functions.
//
// Verilator includes this file in all generated .cpp files that use DPI
// functions. Manually include this file where DPI .c import functions are
// declared to ensure the C functions match the expectations of the DPI imports.

#include "svdpi.h"

#ifndef __COSIM_DPI_HPP__
#define __COSIM_DPI_HPP__

#ifdef _WIN32
#define DPI extern "C" __declspec(dllexport)
#else
#define DPI extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
// DPI IMPORTS
// DPI import at Cosim_DpiPkg.sv:51:17
extern int sv2c_cosimserver_conn_connected(unsigned int endpoint_id);
// DPI import at Cosim_DpiPkg.sv:29:16
extern int sv2c_cosimserver_ep_register(int endpoint_id, long long esi_type_id,
                                        int type_size);
// DPI import at Cosim_DpiPkg.sv:45:16
extern int sv2c_cosimserver_ep_test(unsigned int endpoint_id,
                                    unsigned int *msg_size);
// DPI import at Cosim_DpiPkg.sv:67:16
extern int sv2c_cosimserver_ep_tryget(unsigned int endpoint_id,
                                      const svOpenArrayHandle data,
                                      unsigned int *size_bytes);
// DPI import at Cosim_DpiPkg.sv:56:16
extern int sv2c_cosimserver_ep_tryput(unsigned int endpoint_id,
                                      const svOpenArrayHandle data,
                                      int data_limit);
// DPI import at Cosim_DpiPkg.sv:22:54
extern void sv2c_cosimserver_fini();
// DPI import at Cosim_DpiPkg.sv:19:53
extern int sv2c_cosimserver_init();
#ifdef __cplusplus
}
#endif

#endif
