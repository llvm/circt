// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VESI_Cosim_Top.h for the primary calling header

#ifndef VERILATED_VESI_COSIM_TOP___024ROOT_H_
#define VERILATED_VESI_COSIM_TOP___024ROOT_H_  // guard

#include "verilated.h"
class VESI_Cosim_Top_Cosim_DpiPkg;


class VESI_Cosim_Top__Syms;

class alignas(VL_CACHE_LINE_BYTES) VESI_Cosim_Top___024root final : public VerilatedModule {
  public:
    // CELLS
    VESI_Cosim_Top_Cosim_DpiPkg* __PVT__Cosim_DpiPkg;

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        VL_IN8(clk,0,0);
        VL_IN8(rst,0,0);
        CData/*0:0*/ ESI_Cosim_Top__DOT___ESI_Cosim_UserTopWrapper_mmio_cmd_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_ready;
        CData/*7:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit;
        CData/*7:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_ready;
        CData/*7:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit;
        CData/*7:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__a_rcv;
    };
    struct {
        CData/*0:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_ready;
        CData/*7:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__xmit;
        CData/*7:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l;
        CData/*0:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__a_rcv;
        CData/*7:0*/ ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT____Vlvbound_hd0458421__0;
        CData/*0:0*/ __VdfgRegularize_he50b618e_0_7;
        CData/*0:0*/ __VstlFirstIteration;
        CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
        SData/*8:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_0;
        SData/*15:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk2__DOT__rc;
    };
    struct {
        IData/*23:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a;
        IData/*31:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__rc;
        IData/*23:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg;
        IData/*23:0*/ ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l;
        IData/*31:0*/ ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk2__DOT__rc;
        VlWide<3>/*71:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ __VactIterCount;
        QData/*63:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN;
        QData/*63:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_1;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg;
        QData/*63:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn;
        QData/*50:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a;
        QData/*50:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg;
        QData/*50:0*/ ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l;
        QData/*63:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count;
        VlUnpacked<QData/*63:0*/, 416> ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN;
        VlUnpacked<CData/*7:0*/, 13> ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 8> ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 2> ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 2> ESI_Cosim_Top__DOT__func1_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 3> ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 2> ESI_Cosim_Top__DOT__structFunc_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 7> ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 7> ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 2> ESI_Cosim_Top__DOT__arrayFunc_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 9> ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 16> ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 3314> ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__compressed_manifest_bytes;
        VlUnpacked<QData/*63:0*/, 1> __VstlTriggered;
        VlUnpacked<QData/*63:0*/, 1> __VactTriggered;
        VlUnpacked<QData/*63:0*/, 1> __VnbaTriggered;
    };

    // INTERNAL VARIABLES
    VESI_Cosim_Top__Syms* const vlSymsp;

    // CONSTRUCTORS
    VESI_Cosim_Top___024root(VESI_Cosim_Top__Syms* symsp, const char* v__name);
    ~VESI_Cosim_Top___024root();
    VL_UNCOPYABLE(VESI_Cosim_Top___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
