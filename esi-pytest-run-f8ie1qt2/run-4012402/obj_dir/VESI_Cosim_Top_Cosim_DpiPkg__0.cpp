// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

extern "C" int sv2cCosimserverInit();

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(IData/*31:0*/ &cosim_init__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg\n"); );
    // Body
    int cosim_init__Vfuncrtn__Vcvt;
    cosim_init__Vfuncrtn__Vcvt = sv2cCosimserverInit();
    cosim_init__Vfuncrtn = (cosim_init__Vfuncrtn__Vcvt);
}

extern "C" void sv2cCosimserverFinish();

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_finish_TOP__Cosim_DpiPkg() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_finish_TOP__Cosim_DpiPkg\n"); );
    // Body
    sv2cCosimserverFinish();
}

extern "C" int sv2cCosimserverEpRegister(const char* endpoint_id, const char* from_host_type_id, int from_host_type_size, const char* to_host_type_id, int to_host_type_size);

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg(std::string endpoint_id, std::string from_host_type_id, IData/*31:0*/ from_host_type_size, std::string to_host_type_id, IData/*31:0*/ to_host_type_size, IData/*31:0*/ &cosim_ep_register__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    const char* from_host_type_id__Vcvt;
    from_host_type_id__Vcvt = from_host_type_id.c_str();
    int from_host_type_size__Vcvt;
    from_host_type_size__Vcvt = from_host_type_size;
    const char* to_host_type_id__Vcvt;
    to_host_type_id__Vcvt = to_host_type_id.c_str();
    int to_host_type_size__Vcvt;
    to_host_type_size__Vcvt = to_host_type_size;
    int cosim_ep_register__Vfuncrtn__Vcvt;
    cosim_ep_register__Vfuncrtn__Vcvt = sv2cCosimserverEpRegister(endpoint_id__Vcvt, from_host_type_id__Vcvt, from_host_type_size__Vcvt, to_host_type_id__Vcvt, to_host_type_size__Vcvt);
    cosim_ep_register__Vfuncrtn = (cosim_ep_register__Vfuncrtn__Vcvt);
}

extern "C" int sv2cCosimserverEpTryPut(const char* endpoint_id, const svOpenArrayHandle data, int data_size);

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc7_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 8> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc7_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    static const int data__Vopenprops__ulims[2] = {7, 0};
    static const int data__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps data__Vopenprops(VLVT_UINT8, VLVD_IN|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, data__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, data__Vopenprops__plims);
    VerilatedDpiOpenVar data__Vopenarray (&data__Vopenprops, &data);
    int data_size__Vcvt;
    data_size__Vcvt = data_size;
    int cosim_ep_tryput__Vfuncrtn__Vcvt;
    cosim_ep_tryput__Vfuncrtn__Vcvt = sv2cCosimserverEpTryPut(endpoint_id__Vcvt, &data__Vopenarray, data_size__Vcvt);
    cosim_ep_tryput__Vfuncrtn = (cosim_ep_tryput__Vfuncrtn__Vcvt);
}

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc6_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 1> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc6_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    static const int data__Vopenprops__ulims[2] = {0, 0};
    static const int data__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps data__Vopenprops(VLVT_UINT8, VLVD_IN|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, data__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, data__Vopenprops__plims);
    VerilatedDpiOpenVar data__Vopenarray (&data__Vopenprops, &data);
    int data_size__Vcvt;
    data_size__Vcvt = data_size;
    int cosim_ep_tryput__Vfuncrtn__Vcvt;
    cosim_ep_tryput__Vfuncrtn__Vcvt = sv2cCosimserverEpTryPut(endpoint_id__Vcvt, &data__Vopenarray, data_size__Vcvt);
    cosim_ep_tryput__Vfuncrtn = (cosim_ep_tryput__Vfuncrtn__Vcvt);
}

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc4_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 13> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc4_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    static const int data__Vopenprops__ulims[2] = {12, 0};
    static const int data__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps data__Vopenprops(VLVT_UINT8, VLVD_IN|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, data__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, data__Vopenprops__plims);
    VerilatedDpiOpenVar data__Vopenarray (&data__Vopenprops, &data);
    int data_size__Vcvt;
    data_size__Vcvt = data_size;
    int cosim_ep_tryput__Vfuncrtn__Vcvt;
    cosim_ep_tryput__Vfuncrtn__Vcvt = sv2cCosimserverEpTryPut(endpoint_id__Vcvt, &data__Vopenarray, data_size__Vcvt);
    cosim_ep_tryput__Vfuncrtn = (cosim_ep_tryput__Vfuncrtn__Vcvt);
}

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc3_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 18> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc3_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    static const int data__Vopenprops__ulims[2] = {17, 0};
    static const int data__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps data__Vopenprops(VLVT_UINT8, VLVD_IN|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, data__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, data__Vopenprops__plims);
    VerilatedDpiOpenVar data__Vopenarray (&data__Vopenprops, &data);
    int data_size__Vcvt;
    data_size__Vcvt = data_size;
    int cosim_ep_tryput__Vfuncrtn__Vcvt;
    cosim_ep_tryput__Vfuncrtn__Vcvt = sv2cCosimserverEpTryPut(endpoint_id__Vcvt, &data__Vopenarray, data_size__Vcvt);
    cosim_ep_tryput__Vfuncrtn = (cosim_ep_tryput__Vfuncrtn__Vcvt);
}

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc2_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 16> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc2_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    static const int data__Vopenprops__ulims[2] = {15, 0};
    static const int data__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps data__Vopenprops(VLVT_UINT8, VLVD_IN|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, data__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, data__Vopenprops__plims);
    VerilatedDpiOpenVar data__Vopenarray (&data__Vopenprops, &data);
    int data_size__Vcvt;
    data_size__Vcvt = data_size;
    int cosim_ep_tryput__Vfuncrtn__Vcvt;
    cosim_ep_tryput__Vfuncrtn__Vcvt = sv2cCosimserverEpTryPut(endpoint_id__Vcvt, &data__Vopenarray, data_size__Vcvt);
    cosim_ep_tryput__Vfuncrtn = (cosim_ep_tryput__Vfuncrtn__Vcvt);
}

extern "C" int sv2cCosimserverEpTryGet(const char* endpoint_id, const svOpenArrayHandle data, unsigned int* data_size);

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc7_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 13> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc7_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    static const int data__Vopenprops__ulims[2] = {12, 0};
    static const int data__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps data__Vopenprops(VLVT_UINT8, VLVD_INOUT|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, data__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, data__Vopenprops__plims);
    VerilatedDpiOpenVar data__Vopenarray (&data__Vopenprops, &data);
    unsigned int data_size__Vcvt;
    data_size__Vcvt = data_size;
    int cosim_ep_tryget__Vfuncrtn__Vcvt;
    cosim_ep_tryget__Vfuncrtn__Vcvt = sv2cCosimserverEpTryGet(endpoint_id__Vcvt, &data__Vopenarray, &data_size__Vcvt);
    data_size = (data_size__Vcvt);
    cosim_ep_tryget__Vfuncrtn = (cosim_ep_tryget__Vfuncrtn__Vcvt);
}

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc6_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 9> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc6_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    static const int data__Vopenprops__ulims[2] = {8, 0};
    static const int data__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps data__Vopenprops(VLVT_UINT8, VLVD_INOUT|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, data__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, data__Vopenprops__plims);
    VerilatedDpiOpenVar data__Vopenarray (&data__Vopenprops, &data);
    unsigned int data_size__Vcvt;
    data_size__Vcvt = data_size;
    int cosim_ep_tryget__Vfuncrtn__Vcvt;
    cosim_ep_tryget__Vfuncrtn__Vcvt = sv2cCosimserverEpTryGet(endpoint_id__Vcvt, &data__Vopenarray, &data_size__Vcvt);
    data_size = (data_size__Vcvt);
    cosim_ep_tryget__Vfuncrtn = (cosim_ep_tryget__Vfuncrtn__Vcvt);
}

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc5_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 1> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc5_TOP__Cosim_DpiPkg\n"); );
    // Body
    const char* endpoint_id__Vcvt;
    endpoint_id__Vcvt = endpoint_id.c_str();
    static const int data__Vopenprops__ulims[2] = {0, 0};
    static const int data__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps data__Vopenprops(VLVT_UINT8, VLVD_INOUT|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, data__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, data__Vopenprops__plims);
    VerilatedDpiOpenVar data__Vopenarray (&data__Vopenprops, &data);
    unsigned int data_size__Vcvt;
    data_size__Vcvt = data_size;
    int cosim_ep_tryget__Vfuncrtn__Vcvt;
    cosim_ep_tryget__Vfuncrtn__Vcvt = sv2cCosimserverEpTryGet(endpoint_id__Vcvt, &data__Vopenarray, &data_size__Vcvt);
    data_size = (data_size__Vcvt);
    cosim_ep_tryget__Vfuncrtn = (cosim_ep_tryget__Vfuncrtn__Vcvt);
}

extern "C" void sv2cCosimserverSetManifest(int esi_version, const svOpenArrayHandle compressed_manifest);

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_set_manifest__Vdpioc2_TOP__Cosim_DpiPkg(IData/*31:0*/ esi_version, const VlUnpacked<CData/*7:0*/, 2252> &compressed_manifest) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_set_manifest__Vdpioc2_TOP__Cosim_DpiPkg\n"); );
    // Body
    int esi_version__Vcvt;
    esi_version__Vcvt = esi_version;
    static const int compressed_manifest__Vopenprops__ulims[2] = {2251, 0};
    static const int compressed_manifest__Vopenprops__plims[2] = {7, 0};
    static const VerilatedVarProps compressed_manifest__Vopenprops(VLVT_UINT8, VLVD_IN|VLVF_DPI_CLAY, VerilatedVarProps::Unpacked{}, 1, compressed_manifest__Vopenprops__ulims, VerilatedVarProps::Packed{}, 1, compressed_manifest__Vopenprops__plims);
    VerilatedDpiOpenVar compressed_manifest__Vopenarray (&compressed_manifest__Vopenprops, &compressed_manifest);
    sv2cCosimserverSetManifest(esi_version__Vcvt, &compressed_manifest__Vopenarray);
}
