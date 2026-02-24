// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_2 = vlSelfRef.__PVT__out2_valid_reg__DOT__state;
    vlSelfRef.__PVT__consume_3 = vlSelfRef.__PVT__out3_valid_reg__DOT__state;
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__cb_test__DOT__cmd_cmd_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_0__DOT__output_channel_ready));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = vlSelfRef.__PVT__out0_valid_reg__DOT__state;
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_6_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_32__DOT__address_command__DOT__cmd_32_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_32__DOT__address_command__DOT__cmd_32_data_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_22_0__DOT__output_channel_ready));
    vlSelfRef.output_0_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4.__PVT__out3_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_15_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6.__PVT__out3_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_24_0__DOT__output_channel_ready));
    vlSelfRef.output_3_ready = ((~ (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_26_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_30_0__DOT__output_channel_ready));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_31_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8.__PVT__out3_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_31_0__DOT__output_channel_ready));
    vlSelfRef.output_3_ready = ((~ (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_34_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_37_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_88__DOT__output_channel_valid)) 
                                     & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_36_0__DOT__output_channel_ready)));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                          >> 9U)) & 
                                      (((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                            >> 8U)) 
                                        & (((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                >> 7U)) 
                                            & (((~ 
                                                 (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 6U)) 
                                                & (((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_32__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_32__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_32__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000001__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_32__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_32__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_32__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000011__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U)))) 
                                               | ((((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000100__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000100__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_64__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_64__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_64__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000110__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_64__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_64__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U))) 
                                                  & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                     >> 6U)))) 
                                           | ((((~ 
                                                 (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 6U)) 
                                                & (((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_64__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001000__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001001__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001001__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_128__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_128__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_128__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001011__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U)))) 
                                               | ((((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_128__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_128__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_128__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001101__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001110__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001110__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_256__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_256__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U))) 
                                                  & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                     >> 6U))) 
                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                 >> 7U)))) 
                                       | ((((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                >> 7U)) 
                                            & (((~ 
                                                 (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 6U)) 
                                                & (((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_256__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010000__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_256__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_256__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_256__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010010__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010011__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010011__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U)))) 
                                               | ((((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010101__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010111__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U))) 
                                                  & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                     >> 6U)))) 
                                           | ((((~ 
                                                 (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 6U)) 
                                                & (((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011000__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011000__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_534__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_534__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_534__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011010__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_534__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_534__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U)))) 
                                               | ((((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_534__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011100__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011101__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011101__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011111__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U))) 
                                                  & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                     >> 6U))) 
                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                 >> 7U))) 
                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                             >> 8U)))) 
                                     | ((((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                              >> 8U)) 
                                          & (((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 7U)) 
                                              & (((~ 
                                                   (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                    >> 6U)) 
                                                  & (((~ 
                                                       (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                      & (((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100001__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U)))) 
                                                     | ((((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100010__DOT__output0_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100010__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U))) 
                                                        & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                           >> 5U)))) 
                                                 | ((((~ 
                                                       (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                      & (((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100100__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U)))) 
                                                     | ((((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100110__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100111__DOT__output0_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100111__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U))) 
                                                        & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                           >> 5U))) 
                                                    & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                       >> 6U)))) 
                                             | ((((~ 
                                                   (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                    >> 6U)) 
                                                  & (((~ 
                                                       (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                      & (((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101001__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U)))) 
                                                     | ((((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101011__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U))) 
                                                        & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                           >> 5U)))) 
                                                 | ((((~ 
                                                       (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                      & (((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101100__DOT__output0_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101100__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U))) 
                                                     | (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                    & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                       >> 6U))) 
                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                   >> 7U)))) 
                                         | (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                            >> 8U)) 
                                        & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                           >> 9U))));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___ctor_var_reset(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___ctor_var_reset\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18209466448985614591ull);
    VL_SCOPED_RAND_RESET_W(99, vlSelf->inp, __VscopeHash, 11674450839837347981ull);
    vlSelf->inp_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9997336024544461226ull);
    vlSelf->output_0_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13248880085342039060ull);
    vlSelf->output_1_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7902298621141030093ull);
    vlSelf->output_2_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5285606803710789641ull);
    vlSelf->output_3_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4465621261820349703ull);
    vlSelf->inp_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17541595593609939190ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->output_0, __VscopeHash, 3565184587078972990ull);
    vlSelf->output_0_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7386072530874344436ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->output_1, __VscopeHash, 5600550167531168213ull);
    vlSelf->output_1_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11328874110889888298ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->output_2, __VscopeHash, 13958066754333152748ull);
    vlSelf->output_2_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8143702230589480795ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->output_3, __VscopeHash, 1839122939389980251ull);
    vlSelf->output_3_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18159941315985790452ull);
    vlSelf->__PVT__consume_3 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3851712065953137537ull);
    vlSelf->__PVT__consume_2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16535812106909167653ull);
    vlSelf->__PVT__consume_1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9423595460782033938ull);
    vlSelf->__PVT__consume_0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1656316240931420215ull);
    vlSelf->__PVT__will_write_0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16671588214008449789ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->__PVT__out0_msg_reg, __VscopeHash, 17986564789705952593ull);
    vlSelf->__PVT__will_write_1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6565345871816459926ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->__PVT__out1_msg_reg, __VscopeHash, 5169445537384967357ull);
    vlSelf->__PVT__will_write_2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5422800433389864155ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->__PVT__out2_msg_reg, __VscopeHash, 18426009575718852139ull);
    vlSelf->__PVT__will_write_3 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4142536949964877498ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->__PVT__out3_msg_reg, __VscopeHash, 17590883977631325723ull);
    vlSelf->__PVT__out0_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3599503515460236611ull);
    vlSelf->__PVT__out1_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7087440448375990745ull);
    vlSelf->__PVT__out2_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13317179179747457449ull);
    vlSelf->__PVT__out3_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6494359942246635194ull);
}
