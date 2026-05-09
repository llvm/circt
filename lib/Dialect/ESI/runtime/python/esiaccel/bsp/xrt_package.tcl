# Built from Xilinx provided files

if { $::argc != 5 } {
    puts "ERROR: Program \"$::argv0\" requires 5 arguments!\n"
    puts "Usage: $::argv0 <src_dir> <xoname> <target> <xpfm_path> <device>\n"
    exit 1
}

set srcs      [lindex $::argv 0]
set xoname    [lindex $::argv 1]
set target    [lindex $::argv 2]
set xpfm_path [lindex $::argv 3]
set device    [lindex $::argv 4]

set krnl_name esi_kernel
set suffix "${krnl_name}_${target}_${device}"

set project_path "./temp_kernel"
set package_path "./kernel"

# Create a temporary project that groups the kernel RTL together
create_project -force kernel $project_path

# Collect all the necessary SystemVerilog files
add_files -norecurse [glob $srcs/*.sv]

# Use the correct top level module
set_property top XrtTop [current_fileset]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# Package the temporary project
ipx::package_project -root_dir $package_path -taxonomy /KernelIP -import_files -set_current false

# Load a new project to edit the packaged kernel IP
ipx::unload_core $package_path/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_prj -directory $package_path $package_path/component.xml

set core [ipx::current_core]

ipx::infer_bus_interface ap_clk xilinx.com:signal:clock_rtl:1.0 $core
ipx::infer_bus_interface ap_resetn xilinx.com:signal:reset_rtl:1.0 $core

# Associate AXI data & AXI-Lite control interfaces
ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk $core
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core

# Create the address space for CSRs
set mem_map     [::ipx::add_memory_map "s_axi_control" $core]
set addr_block  [::ipx::add_address_block "reg0" $mem_map]

set_property range 0x1000 $addr_block
set_property range_resolve_type "immediate" $addr_block
set_property range_minimum 0x1000 $addr_block

set reg      [::ipx::add_register "IndirectionMagicNumberLow" $addr_block]
  set_property address_offset  8  $reg
  set_property size           32 $reg

set reg      [::ipx::add_register "IndirectionMagicNumberHigh" $addr_block]
  set_property address_offset 12  $reg
  set_property size           32 $reg

set reg      [::ipx::add_register "IndirectionVersionNumber" $addr_block]
  set_property address_offset 16  $reg
  set_property size           32 $reg

set reg [::ipx::add_register "IndirectionLocation" $addr_block]
  set_property address_offset 24 $reg
  set_property size           32 $reg

set reg [::ipx::add_register "IndirectionRegLow" $addr_block]
  set_property address_offset 32 $reg
  set_property size           32 $reg

set reg [::ipx::add_register "IndirectionRegHigh" $addr_block]
  set_property address_offset 36 $reg
  set_property size           32 $reg

set_property slave_memory_map_ref "s_axi_control" [::ipx::get_bus_interfaces -of $core "s_axi_control"]

# Associatate the hostmem AXI bus with some register on the control bus.
# Necessary for cfgen to work.
ipx::add_register_parameter ASSOCIATED_BUSIF $reg
set_property value m_axi_gmem [ipx::get_register_parameters ASSOCIATED_BUSIF -of_objects $reg]

set_property xpm_libraries {XPM_CDC XPM_FIFO} $core
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
set_property supported_families { } $core
set_property auto_family_support_level level_2 $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -kernel $core
ipx::save_core $core
close_project -delete

if {[file exists "${xoname}"]} {
    file delete -force "${xoname}"
}

package_xo -ctrl_protocol user_managed -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory ${package_path}
