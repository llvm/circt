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
        CData/*0:0*/ ESI_Cosim_Top__DOT___ESI_Cosim_UserTopWrapper_loopback_add_arg_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT___ChannelMux2_input0_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__cmd_cmd_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT___pipelineStage_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT___pipelineStage_a_ready_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT___GEN_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT___GEN_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT___GEN_3;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage_0__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage_0__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage_0__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage_0__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage_0__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__LoopbackInOutAdd__DOT__pipelineStage__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__LoopbackInOutAdd__DOT__pipelineStage__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__LoopbackInOutAdd__DOT__pipelineStage__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__LoopbackInOutAdd__DOT__pipelineStage__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__ConstProducer__DOT___GEN;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__ConstProducer__DOT__ControlReg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__WindowToStructFunc__DOT___GEN;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__WindowToStructFunc__DOT__expecting_frame2___05Freg1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__StructToWindowFunc__DOT___GEN;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__StructToWindowFunc__DOT__sending_frame2___05Freg1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__StructToWindowFunc__DOT__have_struct___05Freg1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_3;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_2;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_2;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_3;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out2_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out3_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__will_write_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__will_write_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__will_write_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__will_write_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__will_write_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__will_write_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_valid_reg__DOT__state;
    };
    struct {
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__will_write_0;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__will_write_1;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_valid_reg__DOT__state;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_3__DOT__output_channel_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_4__DOT__input0_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_5__DOT__input0_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__pipelineStage__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__pipelineStage__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__pipelineStage__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__pipelineStage__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__pipelineStage__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__out_pipe__DOT__a_rcv;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__DataOut_a_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__DataOut_a_ready;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__out_pipe__DOT__a_rcv;
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
    };
    struct {
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_ready_reg;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__xmit;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid;
        CData/*0:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__a_rcv;
        CData/*7:0*/ ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT____Vlvbound_h5916ab55__0;
        CData/*0:0*/ __VstlFirstIteration;
        CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
        SData/*15:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__LoopbackInOutAdd__DOT__pipelineStage__DOT__x_reg;
        SData/*15:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__LoopbackInOutAdd__DOT__pipelineStage__DOT__l;
        SData/*8:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_0;
        SData/*15:0*/ ESI_Cosim_Top__DOT__pipelineStage__DOT__x_reg;
        SData/*15:0*/ ESI_Cosim_Top__DOT__pipelineStage__DOT__l;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage__DOT__x_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage__DOT__l;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage_0__DOT__x_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT__pipelineStage_0__DOT__l;
        IData/*31:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__WindowToStructFunc__DOT__a_reg___05Freg1;
        IData/*31:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__WindowToStructFunc__DOT__b_reg___05Freg1;
        IData/*31:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__StructToWindowFunc__DOT__a_reg___05Freg1;
        IData/*31:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__StructToWindowFunc__DOT__b_reg___05Freg1;
        IData/*31:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__StructToWindowFunc__DOT__c_reg___05Freg1;
        IData/*31:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__StructToWindowFunc__DOT__d_reg___05Freg1;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__callback_cb_arg__DOT__unnamedblk2__DOT__rc;
        IData/*23:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__DataOut_a;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__unnamedblk2__DOT__rc;
        IData/*23:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__out_pipe__DOT__x_reg;
        IData/*23:0*/ ESI_Cosim_Top__DOT__loopback_add_arg__DOT__out_pipe__DOT__l;
        IData/*31:0*/ ESI_Cosim_Top__DOT__loopback_add_result__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__const_producer_data__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT__struct_from_window_result__DOT__unnamedblk2__DOT__rc;
        VlWide<4>/*127:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__DataOut_a;
        IData/*31:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__unnamedblk2__DOT__rc;
        VlWide<4>/*127:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__out_pipe__DOT__x_reg;
        VlWide<4>/*127:0*/ ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__out_pipe__DOT__l;
        IData/*31:0*/ ESI_Cosim_Top__DOT__struct_to_window_result__DOT__unnamedblk2__DOT__rc;
        VlWide<3>/*71:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc;
        IData/*31:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk2__DOT__rc;
        IData/*21:0*/ __VdfgRegularize_he50b618e_0_0;
        IData/*31:0*/ __VactIterCount;
        QData/*63:0*/ ESI_Cosim_Top__DOT____Vcellinp__struct_to_window_result__DataIn;
        QData/*63:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__CallbackTest__DOT___GEN_2;
        QData/*63:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__Top__DOT__MMIOReadWriteClient__DOT__add_amt___05Freg1;
        QData/*63:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN;
        QData/*63:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_1;
        VlWide<4>/*97:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_2;
    };
    struct {
        VlWide<4>/*97:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg;
        VlWide<4>/*97:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg;
        VlWide<4>/*97:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out2_msg_reg;
        VlWide<4>/*97:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out3_msg_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg;
        VlWide<4>/*96:0*/ ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg;
        QData/*63:0*/ ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn;
        QData/*63:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__DataOut_a;
        QData/*63:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__out_pipe__DOT__x_reg;
        QData/*63:0*/ ESI_Cosim_Top__DOT__callback_cb_result__DOT__out_pipe__DOT__l;
        QData/*63:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__DataOut_a;
        QData/*63:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__out_pipe__DOT__x_reg;
        QData/*63:0*/ ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__out_pipe__DOT__l;
        QData/*63:0*/ ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count;
        VlUnpacked<QData/*63:0*/, 379> ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN;
        VlUnpacked<CData/*7:0*/, 13> ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 8> ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 8> ESI_Cosim_Top__DOT__callback_cb_result__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 8> ESI_Cosim_Top__DOT__callback_cb_arg__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 3> ESI_Cosim_Top__DOT__loopback_add_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 2> ESI_Cosim_Top__DOT__loopback_add_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 4> ESI_Cosim_Top__DOT__const_producer_data__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 8> ESI_Cosim_Top__DOT__struct_from_window_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 16> ESI_Cosim_Top__DOT__struct_from_window_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 16> ESI_Cosim_Top__DOT__struct_to_window_arg__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 8> ESI_Cosim_Top__DOT__struct_to_window_result__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 9> ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer;
        VlUnpacked<CData/*7:0*/, 16> ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer;
        VlUnpacked<CData/*7:0*/, 3018> ESI_Cosim_Top__DOT_____05Fmanifest__DOT_____05Fmanifest__DOT__compressed_manifest_bytes;
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
