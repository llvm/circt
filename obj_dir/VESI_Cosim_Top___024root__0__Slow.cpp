// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

VL_ATTR_COLD void VESI_Cosim_Top___024root___eval_static(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_static\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
}

VL_ATTR_COLD void VESI_Cosim_Top___024root___eval_initial__TOP(VESI_Cosim_Top___024root* vlSelf);

VL_ATTR_COLD void VESI_Cosim_Top___024root___eval_initial(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_initial\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VESI_Cosim_Top___024root___eval_initial__TOP(vlSelf);
}

extern const VlWide<829>/*26527:0*/ VESI_Cosim_Top__ConstPool__CONST_h873fb073_0;
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(IData/*31:0*/ &cosim_init__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg(std::string endpoint_id, std::string from_host_type_id, IData/*31:0*/ from_host_type_size, std::string to_host_type_id, IData/*31:0*/ to_host_type_size, IData/*31:0*/ &cosim_ep_register__Vfuncrtn);

VL_ATTR_COLD void VESI_Cosim_Top___024root___eval_initial__TOP(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_initial__TOP\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk1__DOT__rc = 0;
    VlUnpacked<CData/*7:0*/, 13> ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer;
    for (int __Vi0 = 0; __Vi0 < 13; ++__Vi0) {
        ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[__Vi0] = 0;
    }
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk1__DOT__rc = 0;
    VlUnpacked<CData/*7:0*/, 18> ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer;
    for (int __Vi0 = 0; __Vi0 < 18; ++__Vi0) {
        ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[__Vi0] = 0;
    }
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk1__DOT__rc;
    ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk1__DOT__rc = 0;
    IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i;
    ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i = 0;
    IData/*31:0*/ __Vfunc_cosim_init__0__Vfuncout;
    __Vfunc_cosim_init__0__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__1__Vfuncout;
    __Vfunc_cosim_ep_register__1__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__3__Vfuncout;
    __Vfunc_cosim_init__3__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__4__Vfuncout;
    __Vfunc_cosim_ep_register__4__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__6__Vfuncout;
    __Vfunc_cosim_init__6__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__7__Vfuncout;
    __Vfunc_cosim_ep_register__7__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__9__Vfuncout;
    __Vfunc_cosim_init__9__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__10__Vfuncout;
    __Vfunc_cosim_ep_register__10__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__12__Vfuncout;
    __Vfunc_cosim_init__12__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__13__Vfuncout;
    __Vfunc_cosim_ep_register__13__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__15__Vfuncout;
    __Vfunc_cosim_init__15__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__16__Vfuncout;
    __Vfunc_cosim_ep_register__16__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__18__Vfuncout;
    __Vfunc_cosim_init__18__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__19__Vfuncout;
    __Vfunc_cosim_ep_register__19__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__21__Vfuncout;
    __Vfunc_cosim_init__21__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__22__Vfuncout;
    __Vfunc_cosim_ep_register__22__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__24__Vfuncout;
    __Vfunc_cosim_init__24__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__25__Vfuncout;
    __Vfunc_cosim_ep_register__25__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__27__Vfuncout;
    __Vfunc_cosim_init__27__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__28__Vfuncout;
    __Vfunc_cosim_ep_register__28__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__30__Vfuncout;
    __Vfunc_cosim_init__30__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__31__Vfuncout;
    __Vfunc_cosim_ep_register__31__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__33__Vfuncout;
    __Vfunc_cosim_init__33__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__34__Vfuncout;
    __Vfunc_cosim_ep_register__34__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__36__Vfuncout;
    __Vfunc_cosim_init__36__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__37__Vfuncout;
    __Vfunc_cosim_ep_register__37__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__39__Vfuncout;
    __Vfunc_cosim_init__39__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__40__Vfuncout;
    __Vfunc_cosim_ep_register__40__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__42__Vfuncout;
    __Vfunc_cosim_init__42__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__43__Vfuncout;
    __Vfunc_cosim_ep_register__43__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__45__Vfuncout;
    __Vfunc_cosim_init__45__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__46__Vfuncout;
    __Vfunc_cosim_ep_register__46__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__48__Vfuncout;
    __Vfunc_cosim_init__48__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__49__Vfuncout;
    __Vfunc_cosim_ep_register__49__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__51__Vfuncout;
    __Vfunc_cosim_init__51__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__52__Vfuncout;
    __Vfunc_cosim_ep_register__52__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__54__Vfuncout;
    __Vfunc_cosim_init__54__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__55__Vfuncout;
    __Vfunc_cosim_ep_register__55__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__57__Vfuncout;
    __Vfunc_cosim_init__57__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__58__Vfuncout;
    __Vfunc_cosim_ep_register__58__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__60__Vfuncout;
    __Vfunc_cosim_init__60__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__61__Vfuncout;
    __Vfunc_cosim_ep_register__61__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__63__Vfuncout;
    __Vfunc_cosim_init__63__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__64__Vfuncout;
    __Vfunc_cosim_ep_register__64__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__66__Vfuncout;
    __Vfunc_cosim_init__66__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__67__Vfuncout;
    __Vfunc_cosim_ep_register__67__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_init__69__Vfuncout;
    __Vfunc_cosim_init__69__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_register__70__Vfuncout;
    __Vfunc_cosim_ep_register__70__Vfuncout = 0;
    // Body
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0U] = 0x0000000000000cf2ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[1U] = 0x38db6f5b5dedda78ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[2U] = 0x08d9f8d0afcf7e16ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[3U] = 0x9b3450413c4e272cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[4U] = 0xd8a4c5a76c0362eeULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[5U] = 0xd585b1491430d697ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[6U] = 0xfbfe419e24e491c5ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[7U] = 0xaba9222f1bac9292ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[8U] = 0xcefc8a2991b53e6dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[9U] = 0x46514cfbfc90f0fdULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000000aU] = 0xe79607e9affac6daULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000000bU] = 0xcc323e18c9946e8eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000000cU] = 0x817f06ff3fc15ac0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000000dU] = 0xbf37573508372dbfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000000eU] = 0x7c7cffcfa367833dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000000fU] = 0x787f2ce5817bde58ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000010U] = 0xe996dba36e38d19bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000011U] = 0x9fa3400c3f3ddf86ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000012U] = 0x6da06a20e92543d1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000013U] = 0xae87933ed25f0f36ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000014U] = 0x3a8f52e5cfb098e6ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000015U] = 0x2c66a6fa5bcb1c74ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000016U] = 0x8fdb6947342b7cdfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000017U] = 0x575198dbb853d671ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000018U] = 0xaeb69fcf40c0dfa3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000019U] = 0xb5f43fa33fb79b61ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000001aU] = 0xedfadbfb7da6bae6ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000001bU] = 0x3d46de1eadfd083cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000001cU] = 0xe7e7bc563a9628ddULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000001dU] = 0x1d3175ad946f0cc0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000001eU] = 0xe5d5f9e06a16862bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000001fU] = 0x9a46bf29e84a4fa7ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000020U] = 0xe08e852b1a35fabfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000021U] = 0xdf77e039efb3cacfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000022U] = 0x34f8b46bf876b60cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000023U] = 0x374b57fd30270dcaULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000024U] = 0x46cfc4dbd3cbd4e1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000025U] = 0xfe97b257b9db66baULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000026U] = 0xabadaff979ffe648ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000027U] = 0xcd2b3eb3730fc63fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000028U] = 0xd5df4d0dcb4f159eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000029U] = 0x3fcc030cd46106ecULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000002aU] = 0xbce8a5d88acde080ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000002bU] = 0x13ffead0f64de8a0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000002cU] = 0xa6afb8153d1e2dc0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000002dU] = 0x557e00cf6000d773ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000002eU] = 0x2898515ea5ada71cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000002fU] = 0xec3ca628b8e0f9feULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000030U] = 0x696d9a76c1071eb3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000031U] = 0x9bcc374ecde7edc1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000032U] = 0x1dea31eafe77cf9bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000033U] = 0xd5ea5dc831e39520ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000034U] = 0x89fe6b0251ba7958ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000035U] = 0xc5d08cff8e849826ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000036U] = 0x69765639751efdf4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000037U] = 0xdb1e30cf0f4707d7ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000038U] = 0x4c9218ef7db7c923ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000039U] = 0xe1f40b158f5c9213ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000003aU] = 0x4283601e4ee428dfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000003bU] = 0x2125fe8007d81b1fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000003cU] = 0x76eedafbe6baf241ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000003dU] = 0x01ec4221c88dbefaULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000003eU] = 0x7d429a2898e0f90cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000003fU] = 0x125923df50918a06ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000040U] = 0xec00b8c7d83bd294ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000041U] = 0x417b5981dee68aa1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000042U] = 0x6fe180c3ec4e98e8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000043U] = 0x11f84ab6a1bdb11aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000044U] = 0xc020800103e059f1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000045U] = 0x93078fba502b60d1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000046U] = 0x7ac0f6edd4c5f500ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000047U] = 0xb62b1cbab94fda69ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000048U] = 0x03dbe5def5c2aee9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000049U] = 0x1837a17e2098b199ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000004aU] = 0xae187420f1715af4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000004bU] = 0xd3e58d9240af2b1bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000004cU] = 0xbf78903cbb41342eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000004dU] = 0x0f7c929b69b09e67ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000004eU] = 0x449177fa3925909aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000004fU] = 0x508914d72c198029ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000050U] = 0x707a032728f1cc22ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000051U] = 0x7136b9d6851ff636ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000052U] = 0x7fcecb350cc5f478ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000053U] = 0x5dde0c69b88438beULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000054U] = 0xc69415bc07308300ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000055U] = 0x87571e615fd60f14ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000056U] = 0x641afa31c5f0e754ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000057U] = 0x63cd26d9913ccc15ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000058U] = 0x107b9779208d8a7aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000059U] = 0x9077d90181219467ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000005aU] = 0xc517c25d70b5df7eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000005bU] = 0xfa517e7dc4f62f98ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000005cU] = 0x9b61ccdce993993cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000005dU] = 0xbe43c420202883caULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000005eU] = 0xe6de7b6849d2bfa9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000005fU] = 0x39f802984b6ec409ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000060U] = 0x8de80f420fa79c59ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000061U] = 0xf986b968e9be83a8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000062U] = 0xbb92a3ccb05dcfbeULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000063U] = 0xfe45856ff3810059ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000064U] = 0xf281bfc8b08a61c4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000065U] = 0x023f9368271cf7bfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000066U] = 0x68055a03f94435d3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000067U] = 0x023121c9a7176059ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000068U] = 0x787ed50e23c8f0adULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000069U] = 0x5f55e0bb38874044ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000006aU] = 0xef936481504f17c2ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000006bU] = 0x045d6075dd7c7b51ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000006cU] = 0x1504f17c0e0c47b4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000006dU] = 0x9c2746751ef93647ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000006eU] = 0xaa1dfb41d433abc1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000006fU] = 0xfdc0ea1cd67a3bf4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000070U] = 0x4321d0431dfa550eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000071U] = 0x3dacf477e9543bf5ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000072U] = 0x9268efd2a877ea86ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000073U] = 0x87702ad166732a33ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000074U] = 0x9587cb037bf8b219ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000075U] = 0x0dc955fb368fc296ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000076U] = 0xe8fb912c8777f2d9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000077U] = 0xccba700d5702f8c0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000078U] = 0xdf8a3f90ef8b5851ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000079U] = 0x974e140a108b181dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000007aU] = 0xfb24390ef6c5d24bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000007bU] = 0xb470795303b8c0e6ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000007cU] = 0x90c756d94c57019cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000007dU] = 0x8eada1dde8f3818aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000007eU] = 0x2cbe22241a1b0a51ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000007fU] = 0x397d04407d717f81ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000080U] = 0x955c4e42b4be7a08ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000081U] = 0xa85e78bcc98810c4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000082U] = 0x81fdf8bce961e8b1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000083U] = 0xedd7ba510a139c2eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000084U] = 0x2fffd27b841042d8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000085U] = 0x223522af492225b0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000086U] = 0x4cb05648856a0400ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000087U] = 0x5a2bb0975e091eafULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000088U] = 0x6d34287abc42c8f9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000089U] = 0xe5d63d577f4331d3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000008aU] = 0x5f5d51c47d2db48fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000008bU] = 0xfdd01833cace035bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000008cU] = 0xc11226b1ffadf0faULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000008dU] = 0xbad22d7e95cc3000ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000008eU] = 0xdb2dafacbc0b90deULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000008fU] = 0x5e4bb19783974df0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000090U] = 0x459b4af4fca74955ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000091U] = 0x57bcbe8e81bc93feULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000092U] = 0x165d642371cda304ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000093U] = 0xb95cb85a9639eb29ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000094U] = 0x5adc1b562a5afe9aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000095U] = 0x78510115b8383b0aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000096U] = 0x5001ac2e983f6a92ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000097U] = 0xe272910095090b27ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000098U] = 0xe99a9d64ba4cd842ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000099U] = 0x3228b09cd0644652ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000009aU] = 0x871867360b0d725dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000009bU] = 0x6536004db92da0a4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000009cU] = 0x4a6f529bd4a6f52fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000009dU] = 0x611f6cf37a8279bdULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000009eU] = 0x9af6060c7fdf5345ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000009fU] = 0x3f6c2a66e79685afULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a0U] = 0xcd1b3c9e73b051eeULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a1U] = 0x25688643f0cf8dd3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a2U] = 0x572b8738d083ce8fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a3U] = 0xfd85f127a20de359ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a4U] = 0x0ff0135da60306c8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a5U] = 0xd04da53eb6d1ddcdULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a6U] = 0x84b7188536c84cddULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a7U] = 0x53bc24326080b04dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a8U] = 0x6bba6f9811fa6c25ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000a9U] = 0x986ec6dfb0ca6706ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000aaU] = 0x88088ec77972fbe1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000abU] = 0xa41815572b010cb9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000acU] = 0xa29129aa6ca2ae2bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000adU] = 0x211c157e01a1571eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000aeU] = 0x242919be262306faULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000afU] = 0x6ac332b11352546eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b0U] = 0x1541e4851ec1ab29ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b1U] = 0xe0cd021f823e510eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b2U] = 0x4f88204b5dee0007ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b3U] = 0xb99b2fc0529ea030ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b4U] = 0x82d0722b2cfc2f4dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b5U] = 0x72e8e66053bc18eeULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b6U] = 0x2f0d5cfb8e8aa546ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b7U] = 0xe52eddb155e3eb29ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b8U] = 0xe03b16d90b09ea6cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000b9U] = 0x636f94fc56d0feb1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000baU] = 0x26e6b6765331f40eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000bbU] = 0x27ce69511b6ed7bdULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000bcU] = 0x5a32738d6aa17b5aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000bdU] = 0x01e263665b1820a6ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000beU] = 0x908959ed649205e3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000bfU] = 0xe1569c4094bc2285ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c0U] = 0x0c1086244b198b50ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c1U] = 0xed2578877578057eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c2U] = 0x4c2b09a61ee934c2ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c3U] = 0x6b633c7231784967ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c4U] = 0xdc5118770140f49bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c5U] = 0x3a51510af67dc1e2ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c6U] = 0x23428588853be281ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c7U] = 0x95e7b3c8ff832324ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c8U] = 0x45512039d5fb4373ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000c9U] = 0xb1517f0b7b355edcULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000caU] = 0xf87e1ff770fda119ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000cbU] = 0xbc068e3ad826585eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000ccU] = 0xb6b6f2bbe1e0d030ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000cdU] = 0x8a77b1b4a602b5f4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000ceU] = 0xf3b47b44a6a303a9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000cfU] = 0x275e8d3cc7a6bc6fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d0U] = 0x73e705c5395193e7ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d1U] = 0x30c8e99ad2b3020fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d2U] = 0xc2d04e7620d2d137ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d3U] = 0xe8cec5ec233527e3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d4U] = 0x0a18e13750c0fb0bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d5U] = 0x4b9930a260161bf3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d6U] = 0x28562e83f8c68de8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d7U] = 0x41b34be832612297ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d8U] = 0x6fd7b08e506b8967ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000d9U] = 0xf80088cb37c2b0ffULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000daU] = 0x2f3ca465abf4bafdULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000dbU] = 0x82551d17bf950fe8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000dcU] = 0x83437eb986c83fadULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000ddU] = 0x7eb3008db4cb67f9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000deU] = 0x5c3a3bdaa5c15b8bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000dfU] = 0x5cb43b2d6d3cf1b8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e0U] = 0x18a3425508652f6cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e1U] = 0xe3a6b8e275254b49ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e2U] = 0x066a456e1c9e96b9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e3U] = 0x5a2ce605c34dfd57ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e4U] = 0xc698fff0d97f0547ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e5U] = 0x31741f318fb3c81fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e6U] = 0x3c0d0eb00ec43b4dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e7U] = 0xbfe375b7c2eb1244ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e8U] = 0x7816d315508f81dbULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000e9U] = 0x7735495e72a53bc0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000eaU] = 0xa9be440b27a57accULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000ebU] = 0xd507ca5c44d3287aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000ecU] = 0x98d3d0fdc6a9f922ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000edU] = 0x85d3611c0289f6f9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000eeU] = 0x8243a1c8a0a29724ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000efU] = 0xbcdafa97c298e557ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f0U] = 0x4989818e6d3811ccULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f1U] = 0xdc29278ab3389345ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f2U] = 0xc14e281c4c87c958ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f3U] = 0xfbe92ea0926b12b3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f4U] = 0xcf5edf0e43fa59abULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f5U] = 0xeb08c91e3a4c2c6cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f6U] = 0x697dfe34223a71fcULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f7U] = 0x41884572e77f799aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f8U] = 0x0d0eb03247fedf69ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000f9U] = 0x3d0da4ce88c2acc1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000faU] = 0x250d9c20165a1b05ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000fbU] = 0x2a0231270cd0cd49ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000fcU] = 0x260c2676a1340f94ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000fdU] = 0xe60c909cc5a2d273ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000feU] = 0xb98885f5e5ec48e4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x000000ffU] = 0x7763ecb6de6cf66cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000100U] = 0x99d1a7cd557aba81ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000101U] = 0x0b3908796a71edbcULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000102U] = 0x3959bc5666d55ea3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000103U] = 0x738513b0d25cf72aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000104U] = 0x7d721e6a70acc571ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000105U] = 0xd774e6c97a656af8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000106U] = 0xee3a9c7f29baef38ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000107U] = 0xa4abf4583e23af82ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000108U] = 0x2428411e978e56d1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000109U] = 0xbaa4a1c93525363fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000010aU] = 0x51606ab5708e3cc1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000010bU] = 0xf381345730fe9c11ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000010cU] = 0x9d4f9a254511b189ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000010dU] = 0xe35bb01300af39a3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000010eU] = 0xb8605b3c89237258ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000010fU] = 0xae494651b0ee42b8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000110U] = 0x99f6e05f1551c270ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000111U] = 0x3dd7951c0489dfe5ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000112U] = 0xf6a27f713f8153c5ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000113U] = 0x521ba26fa8ad029dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000114U] = 0xe552efe051c3b877ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000115U] = 0x9466c91277bca434ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000116U] = 0x5224bbcaa3c2ce37ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000117U] = 0x4a1bf7b4bceee83dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000118U] = 0x56ee0e2ee5725e5cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000119U] = 0x0db0aa66600fc398ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000011aU] = 0x4ceb6c5695aaffecULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000011bU] = 0x41d270c3bdadb3e5ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000011cU] = 0x8ba0ca9ae83ab4d7ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000011dU] = 0x907d7dbb4aebcd11ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000011eU] = 0x4ac4d1f50f2c501eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000011fU] = 0x54112e830e5d4f0cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000120U] = 0x14e701959181e403ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000121U] = 0x365108189d0394abULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000122U] = 0x39dd844610b10f94ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000123U] = 0x667babcc974455ddULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000124U] = 0xaa0c1342d5210315ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000125U] = 0x1368e0e5e6d27080ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000126U] = 0x816fc650219b4074ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000127U] = 0xbc5b479a4f995432ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000128U] = 0xd373aa3167a190dcULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000129U] = 0x99282f68052a4b9dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000012aU] = 0x464c631b49e65415ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000012bU] = 0x0b26f47bb1f3e75eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000012cU] = 0xe3ce2a67b905ffc3ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000012dU] = 0x3c68c2bb1416dfc4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000012eU] = 0xb499a4758b36e171ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000012fU] = 0x39b10cb092bc58eaULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000130U] = 0x62c26adf4b7bc722ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000131U] = 0x7a5508bb9b67b2a6ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000132U] = 0xb1e711d9d1a485c8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000133U] = 0xcbc31978632f0c65ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000134U] = 0x31978632f0c65e18ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000135U] = 0x78632f0c65e18cbcULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000136U] = 0x32f0c65e18cbc319ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000137U] = 0xea0cc5637c319786ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000138U] = 0x36e1ff43356eaae4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000139U] = 0xf5cc7c8600c49d77ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000013aU] = 0xfa8a0c4dd4b67317ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000013bU] = 0xfcd0994e13d22b8eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000013cU] = 0x9ea9ef695ffb3601ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000013dU] = 0xe83edef2673d831dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000013eU] = 0x95c794172539a0c9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000013fU] = 0x56c6f77c6f037dbeULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000140U] = 0x883bab1ac4b25595ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000141U] = 0x258d2b83db20b8dcULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000142U] = 0x6701dd1cbcd9b18dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000143U] = 0x8a808aec9c19b975ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000144U] = 0x566a3e9c0c6691c9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000145U] = 0x072411d2aee1e64dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000146U] = 0x3119455036f2e498ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000147U] = 0x76e5ce74c77ebf46ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000148U] = 0x6451245960854c14ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000149U] = 0x55f5ee9d918b6407ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000014aU] = 0x49f28508eb94f5ddULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000014bU] = 0xc3aa0c013e388385ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000014cU] = 0xf2e613a38af3fffdULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000014dU] = 0xc27474c7741204e8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000014eU] = 0x74b49c04e8fab4f5ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000014fU] = 0x914bfbeee0826c79ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000150U] = 0xb3393a50cd58f7ddULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000151U] = 0xe4ea9cd05a67b3bfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000152U] = 0x6dac487819f4f452ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000153U] = 0x3b27d13fbd70abbaULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000154U] = 0x0ad0a9a2cef4f632ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000155U] = 0xa6352abb42d5a6dbULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000156U] = 0xcb693e104f103cabULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000157U] = 0xf2ce2a8ea211ed05ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000158U] = 0xbcde289be1a0e8a1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000159U] = 0xcfef600e460a9f24ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000015aU] = 0xb32d65cc2a70b56eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000015bU] = 0x544415e563f33185ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000015cU] = 0x46913c176a3a32c6ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000015dU] = 0xa232c2d4988de0ecULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000015eU] = 0x46a72ec4e2a2206aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000015fU] = 0x21ab5f330d7dc306ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000160U] = 0x7d26005c19daa229ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000161U] = 0x3018b2b5fd47287eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000162U] = 0x5c01e2de6b813967ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000163U] = 0x7e3923f35bf9e273ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000164U] = 0x26b9b1cee1359d4bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000165U] = 0x7a2c5bb9b6a54bb4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000166U] = 0x66da260bd8faf64fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000167U] = 0xc81e4601b9385ee5ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000168U] = 0x16df263e6613cf41ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000169U] = 0x914faf6011947ddaULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000016aU] = 0xf343ba46ac4476acULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000016bU] = 0xd5edb78e3c9ab408ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000016cU] = 0x9352b51c9b4e7658ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000016dU] = 0x6b4e8a1b4e28369cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000016eU] = 0x01afedc2cb874924ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000016fU] = 0x808fc513f428e114ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000170U] = 0x6d71dddcbfaa2b7eULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000171U] = 0x32adc59984663d20ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000172U] = 0xc65761f3a577aabcULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000173U] = 0xdc59a390e7e10e48ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000174U] = 0x91ad91fe22e7c497ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000175U] = 0xa52aa53b6df9ba9dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000176U] = 0x992877e6e8ba444bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000177U] = 0x40afab40ca1995caULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000178U] = 0x9a770861e74d2c6bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000179U] = 0x667753ed52e2011bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000017aU] = 0xbeb5d8b98d424957ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000017bU] = 0xa62ed316594c5f5cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000017cU] = 0x869cb894c5f68a2dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000017dU] = 0xa85a892bec3b9429ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000017eU] = 0x9fdb927685f7957aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000017fU] = 0x2c5625c40447dc70ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000180U] = 0xad5ded022f52c8b1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000181U] = 0x7de45723ef218ee8ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000182U] = 0xef26be59723ef2e4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000183U] = 0x2177a5c8fbe83f23ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000184U] = 0xfc6c0e46dec7236fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000185U] = 0x55af22ef96909c30ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000186U] = 0xb51e58b27c469f38ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000187U] = 0xe4f95cb05e58b299ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000188U] = 0x1184b622a971e58aULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000189U] = 0xbbade035dc48da4bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000018aU] = 0x0ffaad7c86cbe526ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000018bU] = 0xbad6a00fad6b256dULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000018cU] = 0x54eed168ccea92b9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000018dU] = 0x5b06392663a8cacfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000018eU] = 0xf2d2a8020cd0e75fULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000018fU] = 0x0e59959ba27fcb4bULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000190U] = 0xda466d01cd4920cfULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000191U] = 0xa290a8370c631db4ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000192U] = 0xf17a7203fe629d21ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000193U] = 0x93137ee8a3f43b82ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000194U] = 0xaecbbe9b1f55a760ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000195U] = 0x44b0bb170343ce30ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000196U] = 0x023119802316b300ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000197U] = 0xc9678770eae41d98ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000198U] = 0x705d73a7176decc9ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x00000199U] = 0x5a58b24ab2cf0d2cULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000019aU] = 0x3214bc859ad979f1ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000019bU] = 0x541c910b2183c891ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000019cU] = 0xa4d7bca2e58f0832ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000019dU] = 0x4344a2983ac752b0ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000019eU] = 0x555b0ffe3ece2e92ULL;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[0x0000019fU] = 0x000000000000d842ULL;
    ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i = 0U;
    while (VL_GTS_III(32, 0x00000cf2U, ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i)) {
        vlSelfRef.ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT____Vlvbound_hd0458421__0 
            = ((0x678fU >= (0x00007fffU & VL_SHIFTL_III(15,32,32, ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i, 3U)))
                ? (0x000000ffU & (((0U == (0x0000001fU 
                                           & VL_SHIFTL_III(15,32,32, ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i, 3U)))
                                    ? 0U : (VESI_Cosim_Top__ConstPool__CONST_h873fb073_0[
                                            (((IData)(7U) 
                                              + (0x00007fffU 
                                                 & VL_SHIFTL_III(15,32,32, ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i, 3U))) 
                                             >> 5U)] 
                                            << ((IData)(0x00000020U) 
                                                - (0x0000001fU 
                                                   & VL_SHIFTL_III(15,32,32, ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i, 3U))))) 
                                  | (VESI_Cosim_Top__ConstPool__CONST_h873fb073_0[
                                     (0x000003ffU & 
                                      (VL_SHIFTL_III(15,32,32, ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i, 3U) 
                                       >> 5U))] >> 
                                     (0x0000001fU & 
                                      VL_SHIFTL_III(15,32,32, ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i, 3U)))))
                : 0U);
        if (VL_LIKELY(((0x0cf1U >= (0x00000fffU & ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i))))) {
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__compressed_manifest_bytes[(0x00000fffU 
                                                                                & ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i)] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT____Vlvbound_hd0458421__0;
        }
        ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i 
            = ((IData)(1U) + ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__unnamedblk1__DOT__i);
    }
    vlSelfRef.ESI_Cosim_Top__DOT__func1_result__DOT__DataInBuffer[0U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT__func1_result__DOT__DataInBuffer[1U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[0U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[1U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[2U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[3U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[4U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[5U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[6U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[7U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[8U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[9U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[0x0000000aU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[0x0000000bU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__DataInBuffer[0x0000000cU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[1U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[2U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[3U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[4U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[5U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[6U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[7U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[8U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[9U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0x0000000aU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0x0000000bU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0x0000000cU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0x0000000dU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0x0000000eU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0x0000000fU] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0x00000010U] = 0U;
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__DataInBuffer[0x00000011U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[1U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[2U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[3U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[4U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[5U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[6U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[7U] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__DataInBuffer[0U] 
        = (0xfeU & vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__DataInBuffer
           [0U]);
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__DataInBuffer[0U] 
        = (0xfeU & vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__DataInBuffer
           [0U]);
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__0__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__0__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_arg.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("__cosim_mmio_read_write.arg"s, "!esi.channel<!hw.struct<write: i1, offset: ui32, data: i64>>"s, 0x0000000dU, ""s, 0U, __Vfunc_cosim_ep_register__1__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__1__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_arg.unnamedblk1: Cosim endpoint (__cosim_mmio_read_write.arg) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__3__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__3__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_result.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("__cosim_mmio_read_write.result"s, ""s, 0U, "!esi.channel<i64>"s, 8U, __Vfunc_cosim_ep_register__4__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__4__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_result.unnamedblk1: Cosim endpoint (__cosim_mmio_read_write.result) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__6__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__6__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("loopback_inst[0].loopback_tohw.recv"s, "!esi.channel<i8>"s, 1U, ""s, 0U, __Vfunc_cosim_ep_register__7__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__7__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.unnamedblk1: Cosim endpoint (loopback_inst[0].loopback_tohw.recv) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__9__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__9__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_fromhw_send.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("loopback_inst[0].loopback_fromhw.send"s, ""s, 0U, "!esi.channel<i8>"s, 1U, __Vfunc_cosim_ep_register__10__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__10__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_fromhw_send.unnamedblk1: Cosim endpoint (loopback_inst[0].loopback_fromhw.send) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__12__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__12__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("loopback_inst[0].mysvc_recv.recv"s, "!esi.channel<i0>"s, 1U, ""s, 0U, __Vfunc_cosim_ep_register__13__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__13__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.unnamedblk1: Cosim endpoint (loopback_inst[0].mysvc_recv.recv) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__15__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__15__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_send_send.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("loopback_inst[0].mysvc_send.send"s, ""s, 0U, "!esi.channel<i0>"s, 1U, __Vfunc_cosim_ep_register__16__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__16__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_send_send.unnamedblk1: Cosim endpoint (loopback_inst[0].mysvc_send.send) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__18__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__18__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("loopback_inst[1].loopback_tohw.recv"s, "!esi.channel<i8>"s, 1U, ""s, 0U, __Vfunc_cosim_ep_register__19__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__19__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.unnamedblk1: Cosim endpoint (loopback_inst[1].loopback_tohw.recv) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__21__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__21__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_fromhw_send.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("loopback_inst[1].loopback_fromhw.send"s, ""s, 0U, "!esi.channel<i8>"s, 1U, __Vfunc_cosim_ep_register__22__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__22__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_fromhw_send.unnamedblk1: Cosim endpoint (loopback_inst[1].loopback_fromhw.send) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__24__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__24__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("loopback_inst[1].mysvc_recv.recv"s, "!esi.channel<i0>"s, 1U, ""s, 0U, __Vfunc_cosim_ep_register__25__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__25__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.unnamedblk1: Cosim endpoint (loopback_inst[1].mysvc_recv.recv) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__27__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__27__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_send_send.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("loopback_inst[1].mysvc_send.send"s, ""s, 0U, "!esi.channel<i0>"s, 1U, __Vfunc_cosim_ep_register__28__Vfuncout);
    ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__28__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_send_send.unnamedblk1: Cosim endpoint (loopback_inst[1].mysvc_send.send) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__30__Vfuncout);
    ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__30__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.func1_arg.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("func1.arg"s, "!esi.channel<ui16>"s, 2U, ""s, 0U, __Vfunc_cosim_ep_register__31__Vfuncout);
    ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__31__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.func1_arg.unnamedblk1: Cosim endpoint (func1.arg) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__33__Vfuncout);
    ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__33__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.func1_result.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("func1.result"s, ""s, 0U, "!esi.channel<ui16>"s, 2U, __Vfunc_cosim_ep_register__34__Vfuncout);
    ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__34__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.func1_result.unnamedblk1: Cosim endpoint (func1.result) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__36__Vfuncout);
    ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__36__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.structFunc_arg.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("structFunc.arg"s, "!esi.channel<!hw.typealias<@pycde::@ArgStruct, !hw.struct<a: ui16, b: si8>>>"s, 3U, ""s, 0U, __Vfunc_cosim_ep_register__37__Vfuncout);
    ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__37__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.structFunc_arg.unnamedblk1: Cosim endpoint (structFunc.arg) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__39__Vfuncout);
    ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__39__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.structFunc_result.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("structFunc.result"s, ""s, 0U, "!esi.channel<!hw.typealias<@pycde::@ResultStruct, !hw.struct<x: si8, y: si8>>>"s, 2U, __Vfunc_cosim_ep_register__40__Vfuncout);
    ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__40__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.structFunc_result.unnamedblk1: Cosim endpoint (structFunc.result) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__42__Vfuncout);
    ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__42__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.oddStructFunc_arg.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("oddStructFunc.arg"s, "!esi.channel<!hw.typealias<@pycde::@OddStruct, !hw.struct<a: ui12, b: si7, inner: !hw.typealias<@pycde::@OddInner, !hw.struct<p: ui8, q: si8, r: !hw.array<2xui8>>>>>>"s, 7U, ""s, 0U, __Vfunc_cosim_ep_register__43__Vfuncout);
    ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__43__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.oddStructFunc_arg.unnamedblk1: Cosim endpoint (oddStructFunc.arg) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__45__Vfuncout);
    ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__45__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.oddStructFunc_result.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("oddStructFunc.result"s, ""s, 0U, "!esi.channel<!hw.typealias<@pycde::@OddStruct, !hw.struct<a: ui12, b: si7, inner: !hw.typealias<@pycde::@OddInner, !hw.struct<p: ui8, q: si8, r: !hw.array<2xui8>>>>>>"s, 7U, __Vfunc_cosim_ep_register__46__Vfuncout);
    ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__46__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.oddStructFunc_result.unnamedblk1: Cosim endpoint (oddStructFunc.result) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__48__Vfuncout);
    ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__48__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.arrayFunc_arg.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("arrayFunc.arg"s, "!esi.channel<!hw.array<1xsi8>>"s, 1U, ""s, 0U, __Vfunc_cosim_ep_register__49__Vfuncout);
    ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__49__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.arrayFunc_arg.unnamedblk1: Cosim endpoint (arrayFunc.arg) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__51__Vfuncout);
    ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__51__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.arrayFunc_result.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("arrayFunc.result"s, ""s, 0U, "!esi.channel<!hw.typealias<@pycde::@ResultArray, !hw.array<2xsi8>>>"s, 2U, __Vfunc_cosim_ep_register__52__Vfuncout);
    ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__52__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.arrayFunc_result.unnamedblk1: Cosim endpoint (arrayFunc.result) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__54__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__54__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_read_resp_data.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("__cosim_hostmem_read_resp.data"s, "!esi.channel<!hw.struct<tag: ui8, data: i64>>"s, 9U, ""s, 0U, __Vfunc_cosim_ep_register__55__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__55__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_read_resp_data.unnamedblk1: Cosim endpoint (__cosim_hostmem_read_resp.data) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__57__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__57__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_read_req_data.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("__cosim_hostmem_read_req.data"s, ""s, 0U, "!esi.channel<!hw.struct<address: ui64, length: ui32, tag: ui8>>"s, 0x0000000dU, __Vfunc_cosim_ep_register__58__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__58__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_read_req_data.unnamedblk1: Cosim endpoint (__cosim_hostmem_read_req.data) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_req_data__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__60__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__60__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_write_result.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("__cosim_hostmem_write.result"s, "!esi.channel<ui8>"s, 1U, ""s, 0U, __Vfunc_cosim_ep_register__61__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__61__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_write_result.unnamedblk1: Cosim endpoint (__cosim_hostmem_write.result) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__63__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__63__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_write_arg.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("__cosim_hostmem_write.arg"s, ""s, 0U, "!esi.channel<!hw.struct<address: ui64, tag: ui8, data: i64, valid_bytes: i8>>"s, 0x00000012U, __Vfunc_cosim_ep_register__64__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__64__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_write_arg.unnamedblk1: Cosim endpoint (__cosim_hostmem_write.arg) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_arg__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__66__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__66__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:107: Assertion failed in %NESI_Cosim_Top.__cycle_counter.req_ep.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 107, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("__cosim_cycle_count.arg"s, "!esi.channel<i1>"s, 1U, ""s, 0U, __Vfunc_cosim_ep_register__67__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__67__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:111: Assertion failed in %NESI_Cosim_Top.__cycle_counter.req_ep.unnamedblk1: Cosim endpoint (__cosim_cycle_count.arg) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 111, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_init_TOP__Cosim_DpiPkg(__Vfunc_cosim_init__69__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_init__69__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:38: Assertion failed in %NESI_Cosim_Top.__cycle_counter.resp_ep.unnamedblk1: Cosim init failed (%11d)\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 38, "");
    }
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_register_TOP__Cosim_DpiPkg("__cosim_cycle_count.result"s, ""s, 0U, "struct{cycle:int<64>,freq:int<64>}"s, 0x00000010U, __Vfunc_cosim_ep_register__70__Vfuncout);
    ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk1__DOT__rc 
        = __Vfunc_cosim_ep_register__70__Vfuncout;
    if (VL_UNLIKELY(((0U != ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk1__DOT__rc)))) {
        VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:42: Assertion failed in %NESI_Cosim_Top.__cycle_counter.resp_ep.unnamedblk1: Cosim endpoint (__cosim_cycle_count.result) register failed: %11d\n",0,
                     64,VL_TIME_UNITED_Q(1),-12,vlSymsp->name(),
                     32,ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk1__DOT__rc);
        VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 42, "");
    }
}

VL_ATTR_COLD void VESI_Cosim_Top___024root___eval_final(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_final\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VESI_Cosim_Top___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool VESI_Cosim_Top___024root___eval_phase__stl(VESI_Cosim_Top___024root* vlSelf);

VL_ATTR_COLD void VESI_Cosim_Top___024root___eval_settle(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_settle\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VstlIterCount;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00000064U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            VESI_Cosim_Top___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
            VL_FATAL_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESI_Cosim_Top.sv", 18, "", "Settle region did not converge after 100 tries");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
    } while (VESI_Cosim_Top___024root___eval_phase__stl(vlSelf));
}

VL_ATTR_COLD void VESI_Cosim_Top___024root___eval_triggers__stl(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_triggers__stl\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VstlTriggered
                                      [0U]) | (IData)((IData)(vlSelfRef.__VstlFirstIteration)));
    vlSelfRef.__VstlFirstIteration = 0U;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        VESI_Cosim_Top___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
}

VL_ATTR_COLD bool VESI_Cosim_Top___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void VESI_Cosim_Top___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(VESI_Cosim_Top___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool VESI_Cosim_Top___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___trigger_anySet__stl\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_set_manifest__Vdpioc2_TOP__Cosim_DpiPkg(IData/*31:0*/ esi_version, const VlUnpacked<CData/*7:0*/, 3314> &compressed_manifest);
extern const VlWide<10>/*319:0*/ VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0;

VL_ATTR_COLD void VESI_Cosim_Top___024root___stl_sequent__TOP__0(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___stl_sequent__TOP__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    VlWide<16>/*511:0*/ __Vtemp_2;
    // Body
    VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_set_manifest__Vdpioc2_TOP__Cosim_DpiPkg(0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__compressed_manifest_bytes);
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_0 
        = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[0U] 
        = (IData)((((QData)((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[1U])) 
                    << 0x00000020U) | (QData)((IData)(
                                                      vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[0U]))));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[1U] 
        = (IData)(((((QData)((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[1U])) 
                     << 0x00000020U) | (QData)((IData)(
                                                       vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[0U]))) 
                   >> 0x00000020U));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[2U] 
        = ((0U != (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U] 
                   >> 0x0000000aU)) ? (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U] 
                                       - (IData)(0x00000400U))
            : (0x000003ffU & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U]));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[3U] 
        = (1U & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[3U]);
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__DataInBuffer[0U] 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__DataInBuffer[0U] 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[0U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
             [3U] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                                     [2U] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
               [1U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
              [0U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[1U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
             [7U] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                                     [6U] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
               [5U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
              [4U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[2U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
             [0x0bU] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                                        [0x0aU] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
               [9U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
              [8U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[3U] 
        = (1U & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
           [0x0cU]);
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a 
        = ((vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer
            [1U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer
           [0U]);
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a 
        = ((vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
            [2U] << 0x00000010U) | ((vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
                                     [1U] << 8U) | 
                                    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
                                    [0U]));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a 
        = (((QData)((IData)(((0x00070000U & (vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                             [6U] << 0x00000010U)) 
                             | ((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                 [5U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                [4U])))) << 0x00000020U) 
           | (QData)((IData)((((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                [3U] << 0x00000018U) 
                               | (vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                  [2U] << 0x00000010U)) 
                              | ((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                  [1U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                 [0U])))));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a[0U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
             [3U] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                                     [2U] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
               [1U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
              [0U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a[1U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
             [7U] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                                     [6U] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
               [5U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
              [4U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a[2U] 
        = (0x000000ffU & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
           [8U]);
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_1 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state) 
           & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state)));
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_result__DOT__DataInBuffer[0U] 
        = (0x000000ffU & vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg);
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_result__DOT__DataInBuffer[1U] 
        = (0x000000ffU & ((IData)(1U) + vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg));
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_result__DOT__DataInBuffer[0U] 
        = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_result__DOT__DataInBuffer[1U] 
        = (0x000000ffU & ((IData)(1U) + (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[0U] 
        = (0x000000ffU & ((IData)(1U) + (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[1U] 
        = (0x000000ffU & ((IData)(2U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                 >> 8U))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[2U] 
        = (0x000000ffU & ((IData)(2U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                 >> 0x00000010U))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[3U] 
        = (0x000000ffU & ((IData)(5U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                 >> 0x00000018U))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[4U] 
        = ((0x00000080U & (((IData)(1U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                   >> 0x00000027U))) 
                           << 7U)) | (0x0000007fU & 
                                      ((IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                >> 0x00000020U)) 
                                       - (IData)(3U))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[5U] 
        = (0x000000ffU & (((IData)(1U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                  >> 0x00000027U))) 
                          >> 1U));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[6U] 
        = ((0xf8U & vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer
            [6U]) | (7U & (((IData)(1U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                   >> 0x00000027U))) 
                           >> 9U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[8U] 
        = (0x000000ffU & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[9U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 8U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000aU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000010U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000bU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000018U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000cU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000020U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000dU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000028U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000eU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000030U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000fU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000038U)));
    vlSelfRef.__VdfgRegularize_he50b618e_0_7 = (IData)(
                                                       (0U 
                                                        != 
                                                        (0xfffffc00U 
                                                         & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U])));
    __Vtemp_2[0U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[0U];
    __Vtemp_2[1U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[1U];
    __Vtemp_2[2U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[2U];
    __Vtemp_2[3U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[3U];
    __Vtemp_2[4U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[4U];
    __Vtemp_2[5U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[5U];
    __Vtemp_2[6U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[6U];
    __Vtemp_2[7U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[7U];
    __Vtemp_2[8U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[8U];
    __Vtemp_2[9U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[9U];
    __Vtemp_2[0x0000000aU] = (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN);
    __Vtemp_2[0x0000000bU] = (IData)((vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN 
                                      >> 0x00000020U));
    __Vtemp_2[0x0000000cU] = 0U;
    __Vtemp_2[0x0000000dU] = 0U;
    __Vtemp_2[0x0000000eU] = 0U;
    __Vtemp_2[0x0000000fU] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state)
            ? (((QData)((IData)(__Vtemp_2[(((IData)(0x0000003fU) 
                                            + (0x000001ffU 
                                               & VL_SHIFTL_III(9,9,32, 
                                                               (7U 
                                                                & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                                   >> 3U)), 6U))) 
                                           >> 5U)])) 
                << ((0U == (0x0000001fU & VL_SHIFTL_III(9,9,32, 
                                                        (7U 
                                                         & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                            >> 3U)), 6U)))
                     ? 0x00000020U : ((IData)(0x00000040U) 
                                      - (0x0000001fU 
                                         & VL_SHIFTL_III(9,9,32, 
                                                         (7U 
                                                          & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                             >> 3U)), 6U))))) 
               | (((0U == (0x0000001fU & VL_SHIFTL_III(9,9,32, 
                                                       (7U 
                                                        & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                           >> 3U)), 6U)))
                    ? 0ULL : ((QData)((IData)(__Vtemp_2[
                                              (((IData)(0x0000001fU) 
                                                + (0x000001ffU 
                                                   & VL_SHIFTL_III(9,9,32, 
                                                                   (7U 
                                                                    & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                                       >> 3U)), 6U))) 
                                               >> 5U)])) 
                              << ((IData)(0x00000020U) 
                                  - (0x0000001fU & 
                                     VL_SHIFTL_III(9,9,32, 
                                                   (7U 
                                                    & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                       >> 3U)), 6U))))) 
                  | ((QData)((IData)(__Vtemp_2[(0x0000000fU 
                                                & (VL_SHIFTL_III(9,9,32, 
                                                                 (7U 
                                                                  & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                                     >> 3U)), 6U) 
                                                   >> 5U))])) 
                     >> (0x0000001fU & VL_SHIFTL_III(9,9,32, 
                                                     (7U 
                                                      & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                         >> 3U)), 6U)))))
            : vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_1);
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT___ESI_Cosim_UserTopWrapper_mmio_cmd_ready 
        = (1U & (~ (((~ (IData)(vlSelfRef.__VdfgRegularize_he50b618e_0_7)) 
                     & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state)) 
                    | ((IData)(vlSelfRef.__VdfgRegularize_he50b618e_0_7) 
                       & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state)))));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[0U] 
        = (0x000000ffU & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[1U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 8U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[2U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000010U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[3U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000018U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[4U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000020U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[5U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000028U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[6U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000030U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[7U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000038U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT___ESI_Cosim_UserTopWrapper_mmio_cmd_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_0 
        = ((~ (IData)(vlSelfRef.__VdfgRegularize_he50b618e_0_7)) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_1 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit) 
           & (IData)(vlSelfRef.__VdfgRegularize_he50b618e_0_7));
}

VL_ATTR_COLD void VESI_Cosim_Top___024root___eval_stl(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_stl\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered[0U])) {
        VESI_Cosim_Top___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD bool VESI_Cosim_Top___024root___eval_phase__stl(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_phase__stl\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VstlExecute;
    // Body
    VESI_Cosim_Top___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = VESI_Cosim_Top___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        VESI_Cosim_Top___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

bool VESI_Cosim_Top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void VESI_Cosim_Top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(VESI_Cosim_Top___024root___trigger_anySet__act(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void VESI_Cosim_Top___024root___ctor_var_reset(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___ctor_var_reset\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18209466448985614591ull);
    vlSelf->ESI_Cosim_Top__DOT___ESI_Cosim_UserTopWrapper_mmio_cmd_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 41420702173694259ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 8579712916342621071ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14669889388341205251ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid_0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17359063088583450908ull);
    for (int __Vi0 = 0; __Vi0 < 416; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN[__Vi0] = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 3036458792222551677ull);
    }
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_0 = VL_SCOPED_RAND_RESET_I(9, __VscopeHash, 18254345305488834150ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_1 = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 7176367569398322535ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4765681672168793083ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12617600079505875908ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7458161116464035244ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1, __VscopeHash, 6569079360018095821ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg, __VscopeHash, 16466332573749151235ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16138274176031677842ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg, __VscopeHash, 7411393006328025379ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13931494277802891808ull);
    vlSelf->ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1421156512019534714ull);
    for (int __Vi0 = 0; __Vi0 < 13; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8183166814260162184ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a, __VscopeHash, 7919667789404140234ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17866775178300427780ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc = 0;
    VL_SCOPED_RAND_RESET_W(97, vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg, __VscopeHash, 1236029576151022939ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2673463910591709263ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7609492501422207120ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10561791408360756044ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l, __VscopeHash, 2458283771030084910ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8471307147724410060ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16830900316153162936ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 9398338701224086134ull);
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk2__DOT__rc = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOutBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4208007211673721040ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15944576725308479098ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 2152615680314058441ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10716512558037176707ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9538085538384350644ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10818954403043852700ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3785345044771801355ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6061796281159755954ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15903419951953836875ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16281098812422414687ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5605596175443766963ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2558850644475874696ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7328152566718764407ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12984495641365183725ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15287661461511026337ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1574058382515380789ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk2__DOT__rc = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOutBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16075130513644247786ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2544299729129239292ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 12338470691977730914ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3982287430905341711ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 838956080348419747ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7339542865415286043ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 10071255732552299244ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17998993309779240876ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14122293531219565704ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2655713230658931325ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8975309172248417891ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8107954454708682522ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18187240427690370673ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11084220820053264621ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5643197310272997554ull);
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3216795216611501470ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk2__DOT__rc = 0;
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3584656748882551752ull);
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 7635217638409165569ull);
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14427986627937711524ull);
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12896498615090890241ull);
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9363146779553788771ull);
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15191727429858623307ull);
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11200692534915556104ull);
    vlSelf->ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17429976214861347132ull);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__func1_result__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk2__DOT__rc = 0;
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5006071676703678729ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a = VL_SCOPED_RAND_RESET_I(24, __VscopeHash, 10060773782825007175ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5737293943350587967ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg = VL_SCOPED_RAND_RESET_I(24, __VscopeHash, 8476033024324026807ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4983007097080441595ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11517159602740867060ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10232187082768362945ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l = VL_SCOPED_RAND_RESET_I(24, __VscopeHash, 6149282570279526855ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9602586915724350497ull);
    vlSelf->ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14045983151048075019ull);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__structFunc_result__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk2__DOT__rc = 0;
    for (int __Vi0 = 0; __Vi0 < 7; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7383623142159449233ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a = VL_SCOPED_RAND_RESET_Q(51, __VscopeHash, 3152838873795431984ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15930269163580190582ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg = VL_SCOPED_RAND_RESET_Q(51, __VscopeHash, 8519977638562236852ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2936561690928745491ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12036083783760341409ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9636399296037386077ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l = VL_SCOPED_RAND_RESET_Q(51, __VscopeHash, 10430674300929739797ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7680552979458303904ull);
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4868721914486459152ull);
    for (int __Vi0 = 0; __Vi0 < 7; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk2__DOT__rc = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOutBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5362568146629322814ull);
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2673760904654593623ull);
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 190548281340748323ull);
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13688408804182858059ull);
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18010066831030073211ull);
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15811264178687992895ull);
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6485218489912489345ull);
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9170505790292510167ull);
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16195157289382100343ull);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT__arrayFunc_result__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk2__DOT__rc = 0;
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2305372722010190646ull);
    VL_SCOPED_RAND_RESET_W(72, vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a, __VscopeHash, 2990749180915613059ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5197437540241952304ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3612505899187398998ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5530976533515380448ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1231038192026439108ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5571591108872234243ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7426133085902712859ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17688646462277782760ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4327244349090099189ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6714604024856335525ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15440606833364512884ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12116630330641953189ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14097063689800488915ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16584113337763289055ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count = 0;
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14683305574197524076ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15160585000400120194ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit = 0;
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc = 0;
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9540753860520621166ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15531888128148889152ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13590169668561307276ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4340219233450388417ull);
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10942673575651260843ull);
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk2__DOT__rc = 0;
    for (int __Vi0 = 0; __Vi0 < 3314; ++__Vi0) {
        vlSelf->ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__compressed_manifest_bytes[__Vi0] = 0;
    }
    vlSelf->ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT____Vlvbound_hd0458421__0 = 0;
    vlSelf->__VdfgRegularize_he50b618e_0_7 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18224466492814703005ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VstlTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggered[__Vi0] = 0;
    }
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9526919608049418986ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VnbaTriggered[__Vi0] = 0;
    }
}
