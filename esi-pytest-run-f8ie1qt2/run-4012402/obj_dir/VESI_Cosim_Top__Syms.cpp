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
    Verilated::stackCheck(11703);
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
    __Vscope_ESI_Cosim_Top__read_address.configure(this, name(), "ESI_Cosim_Top.read_address", "read_address", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__read_address__out_pipe.configure(this, name(), "ESI_Cosim_Top.read_address.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__read_address__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.read_address.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__read_address__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.read_address.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__read_data.configure(this, name(), "ESI_Cosim_Top.read_data", "read_data", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__read_data__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.read_data.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__read_data__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.read_data.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__write_ack.configure(this, name(), "ESI_Cosim_Top.write_ack", "write_ack", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__write_ack__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.write_ack.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__write_ack__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.write_ack.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__write_req.configure(this, name(), "ESI_Cosim_Top.write_req", "write_req", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__write_req__out_pipe.configure(this, name(), "ESI_Cosim_Top.write_req.out_pipe", "out_pipe", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__write_req__unnamedblk1.configure(this, name(), "ESI_Cosim_Top.write_req.unnamedblk1", "unnamedblk1", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    __Vscope_ESI_Cosim_Top__write_req__unnamedblk2.configure(this, name(), "ESI_Cosim_Top.write_req.unnamedblk2", "unnamedblk2", "<null>", -12, VerilatedScope::SCOPE_OTHER);
    // Setup export functions
    for (int __Vfinal = 0; __Vfinal < 2; ++__Vfinal) {
    }
}
