// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VESI_Cosim_Top.h for the primary calling header

#ifndef VERILATED_VESI_COSIM_TOP_CHANNELDEMUXNIMPL_DATA_TYPE_TYPE_TYPE_STRUCT_WRITE_I1_OFFSET_UI32_DATA_I64_NEXT_SEL_WIDTH0_NUM_OUTS4_H_
#define VERILATED_VESI_COSIM_TOP_CHANNELDEMUXNIMPL_DATA_TYPE_TYPE_TYPE_STRUCT_WRITE_I1_OFFSET_UI32_DATA_I64_NEXT_SEL_WIDTH0_NUM_OUTS4_H_  // guard

#include "verilated.h"


class VESI_Cosim_Top__Syms;

class alignas(VL_CACHE_LINE_BYTES) VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4 final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst,0,0);
    VL_IN8(inp_valid,0,0);
    VL_IN8(output_0_ready,0,0);
    VL_IN8(output_1_ready,0,0);
    VL_IN8(output_2_ready,0,0);
    VL_IN8(output_3_ready,0,0);
    VL_OUT8(inp_ready,0,0);
    VL_OUT8(output_0_valid,0,0);
    VL_OUT8(output_1_valid,0,0);
    VL_OUT8(output_2_valid,0,0);
    VL_OUT8(output_3_valid,0,0);
    CData/*0:0*/ __PVT__consume_3;
    CData/*0:0*/ __PVT__consume_2;
    CData/*0:0*/ __PVT__consume_1;
    CData/*0:0*/ __PVT__consume_0;
    CData/*0:0*/ __PVT__will_write_0;
    CData/*0:0*/ __PVT__will_write_1;
    CData/*0:0*/ __PVT__will_write_2;
    CData/*0:0*/ __PVT__will_write_3;
    CData/*0:0*/ __PVT__out0_valid_reg__DOT__state;
    CData/*0:0*/ __PVT__out1_valid_reg__DOT__state;
    CData/*0:0*/ __PVT__out2_valid_reg__DOT__state;
    CData/*0:0*/ __PVT__out3_valid_reg__DOT__state;
    VL_INW(inp,98,0,4);
    VL_OUTW(output_0,96,0,4);
    VL_OUTW(output_1,96,0,4);
    VL_OUTW(output_2,96,0,4);
    VL_OUTW(output_3,96,0,4);
    VlWide<4>/*96:0*/ __PVT__out0_msg_reg;
    VlWide<4>/*96:0*/ __PVT__out1_msg_reg;
    VlWide<4>/*96:0*/ __PVT__out2_msg_reg;
    VlWide<4>/*96:0*/ __PVT__out3_msg_reg;

    // INTERNAL VARIABLES
    VESI_Cosim_Top__Syms* const vlSymsp;

    // CONSTRUCTORS
    VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4(VESI_Cosim_Top__Syms* symsp, const char* v__name);
    ~VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4();
    VL_UNCOPYABLE(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
