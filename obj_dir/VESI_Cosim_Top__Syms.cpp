// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "VESI_Cosim_Top__pch.h"
#include "VESI_Cosim_Top.h"
#include "VESI_Cosim_Top___024root.h"
#include "VESI_Cosim_Top___024unit.h"
#include "VESI_Cosim_Top_Cosim_DpiPkg.h"

// FUNCTIONS
VESI_Cosim_Top__Syms::~VESI_Cosim_Top__Syms()
{
}

VESI_Cosim_Top__Syms::VESI_Cosim_Top__Syms(VerilatedContext* contextp, const char* namep, VESI_Cosim_Top* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
    , TOP__Cosim_DpiPkg{this, Verilated::catName(namep, "Cosim_DpiPkg")}
{
    // Check resources
    Verilated::stackCheck(18071);
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-12);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    TOP.__PVT__Cosim_DpiPkg = &TOP__Cosim_DpiPkg;
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    TOP__Cosim_DpiPkg.__Vconfigure(true);
    // Setup scopes
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_req_data.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_read_req_data", "__cosim_hostmem_read_req_data", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_req_data__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_read_req_data.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_resp_data.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_read_resp_data", "__cosim_hostmem_read_resp_data", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_resp_data__out_pipe.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_read_resp_data.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_resp_data__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_read_resp_data.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_resp_data__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_read_resp_data.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_arg.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_write_arg", "__cosim_hostmem_write_arg", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_arg__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_write_arg.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_result.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_write_result", "__cosim_hostmem_write_result", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_result__out_pipe.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_write_result.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_result__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_write_result.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_result__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.__cosim_hostmem_write_result.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_arg.configure(this, name(), "ESI_Cosim_Top.__cosim_mmio_read_write_arg", "__cosim_mmio_read_write_arg", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_arg__out_pipe.configure(this, name(), "ESI_Cosim_Top.__cosim_mmio_read_write_arg.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_arg__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.__cosim_mmio_read_write_arg.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_arg__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.__cosim_mmio_read_write_arg.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_result.configure(this, name(), "ESI_Cosim_Top.__cosim_mmio_read_write_result", "__cosim_mmio_read_write_result", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_result__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.__cosim_mmio_read_write_result.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_result__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.__cosim_mmio_read_write_result.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcycle_counter__req_ep.configure(this, name(), "ESI_Cosim_Top.__cycle_counter.req_ep", "req_ep", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcycle_counter__req_ep__out_pipe.configure(this, name(), "ESI_Cosim_Top.__cycle_counter.req_ep.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcycle_counter__req_ep__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.__cycle_counter.req_ep.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcycle_counter__req_ep__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.__cycle_counter.req_ep.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcycle_counter__resp_ep.configure(this, name(), "ESI_Cosim_Top.__cycle_counter.resp_ep", "resp_ep", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcycle_counter__resp_ep__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.__cycle_counter.resp_ep.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top_____05Fcycle_counter__resp_ep__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.__cycle_counter.resp_ep.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__arrayFunc_arg.configure(this, name(), "ESI_Cosim_Top.arrayFunc_arg", "arrayFunc_arg", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__arrayFunc_arg__out_pipe.configure(this, name(), "ESI_Cosim_Top.arrayFunc_arg.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__arrayFunc_arg__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.arrayFunc_arg.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__arrayFunc_arg__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.arrayFunc_arg.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__arrayFunc_result.configure(this, name(), "ESI_Cosim_Top.arrayFunc_result", "arrayFunc_result", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__arrayFunc_result__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.arrayFunc_result.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__arrayFunc_result__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.arrayFunc_result.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__func1_arg.configure(this, name(), "ESI_Cosim_Top.func1_arg", "func1_arg", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__func1_arg__out_pipe.configure(this, name(), "ESI_Cosim_Top.func1_arg.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__func1_arg__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.func1_arg.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__func1_arg__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.func1_arg.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__func1_result.configure(this, name(), "ESI_Cosim_Top.func1_result", "func1_result", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__func1_result__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.func1_result.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__func1_result__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.func1_result.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_fromhw_send.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_loopback_fromhw_send", "loopback_inst5B05D_loopback_fromhw_send", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_fromhw_send__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_loopback_fromhw_send.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_fromhw_send__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_loopback_fromhw_send.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_tohw_recv.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv", "loopback_inst5B05D_loopback_tohw_recv", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_tohw_recv__out_pipe.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_tohw_recv__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_tohw_recv__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_recv_recv.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv", "loopback_inst5B05D_mysvc_recv_recv", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_recv_recv__out_pipe.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_recv_recv__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_recv_recv__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_send_send.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_mysvc_send_send", "loopback_inst5B05D_mysvc_send_send", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_send_send__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_mysvc_send_send.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_send_send__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B05D_mysvc_send_send.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_fromhw_send.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_loopback_fromhw_send", "loopback_inst5B15D_loopback_fromhw_send", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_fromhw_send__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_loopback_fromhw_send.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_fromhw_send__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_loopback_fromhw_send.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_tohw_recv.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv", "loopback_inst5B15D_loopback_tohw_recv", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_tohw_recv__out_pipe.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_tohw_recv__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_tohw_recv__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_recv_recv.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv", "loopback_inst5B15D_mysvc_recv_recv", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_recv_recv__out_pipe.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_recv_recv__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_recv_recv__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_send_send.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_mysvc_send_send", "loopback_inst5B15D_mysvc_send_send", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_send_send__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_mysvc_send_send.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_send_send__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.loopback_inst5B15D_mysvc_send_send.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__oddStructFunc_arg.configure(this, name(), "ESI_Cosim_Top.oddStructFunc_arg", "oddStructFunc_arg", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__oddStructFunc_arg__out_pipe.configure(this, name(), "ESI_Cosim_Top.oddStructFunc_arg.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__oddStructFunc_arg__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.oddStructFunc_arg.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__oddStructFunc_arg__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.oddStructFunc_arg.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__oddStructFunc_result.configure(this, name(), "ESI_Cosim_Top.oddStructFunc_result", "oddStructFunc_result", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__oddStructFunc_result__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.oddStructFunc_result.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__oddStructFunc_result__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.oddStructFunc_result.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__structFunc_arg.configure(this, name(), "ESI_Cosim_Top.structFunc_arg", "structFunc_arg", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__structFunc_arg__out_pipe.configure(this, name(), "ESI_Cosim_Top.structFunc_arg.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__structFunc_arg__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.structFunc_arg.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__structFunc_arg__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.structFunc_arg.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__structFunc_result.configure(this, name(), "ESI_Cosim_Top.structFunc_result", "structFunc_result", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__structFunc_result__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.structFunc_result.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__structFunc_result__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.structFunc_result.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    // Setup export functions
    for (int __Vfinal = 0; __Vfinal < 2; ++__Vfinal) {
    }
}
