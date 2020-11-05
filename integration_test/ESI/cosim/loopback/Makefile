ESI_ROOT = ../../../../
ESI_BIN = /tmp/esibuild
build:
	cmake -S ${ESI_ROOT} -B${ESI_BIN}
	cmake --build ${ESI_BIN}
	# cmake --build ../../../../bin --target install
	verilator --cc --trace --top-module cosim_test ../../rtl/Cosim_DpiPkg.sv ../../rtl/Cosim_Endpoint.sv driver.sv --exe driver.cpp ${ESI_BIN}/src/cosim/cosim_dpi_server/libEsiCosimDpiServer.so -LDFLAGS "-lpthread"
	make USER_CPPFLAGS="-DVL_USER_STOP -O0 -ggdb" -j -C obj_dir -f Vcosim_test.mk Vcosim_test

run:
	LD_LIBRARY_PATH=${ESI_BIN}/src/cosim/cosim_dpi_server/ ./obj_dir/Vcosim_test +trace

valgrind:
	LD_LIBRARY_PATH=${ESI_BIN}/src/cosim/cosim_dpi_server/ valgrind ./obj_dir/Vcosim_test +trace

debug:
	LD_LIBRARY_PATH=${ESI_BIN}/src/cosim/cosim_dpi_server/ gdb ./obj_dir/Vcosim_test +trace

questa:
	cmake -S ${ESI_ROOT} -B${ESI_BIN}
	cmake --build ${ESI_BIN}
	vlog -sv ../../rtl/Cosim_DpiPkg.sv ../../rtl/Cosim_Endpoint.sv driver.sv +acc
	vopt cosim_test -o cosim_test_opt +acc

qrun:
	vsim -sv_lib ${ESI_BIN}/src/cosim/cosim_dpi_server/libEsiCosimDpiServer cosim_test_opt -voptargs="+acc" -do wave.do -c

qgui:
	vsim -sv_lib ${ESI_BIN}/src/cosim/cosim_dpi_server/libEsiCosimDpiServer cosim_test_opt -voptargs="+acc" -do wave.do
