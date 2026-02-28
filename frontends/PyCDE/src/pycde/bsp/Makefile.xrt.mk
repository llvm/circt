# Please source Vitis and XRT before running this makefile
# $ source /opt/xilinx/Vitis/2022.1/settings64.sh
# $ source /opt/xilinx/xrt/setup.sh

PYTHON ?= python3
CXX ?= g++
VIVADO := $(XILINX_VIVADO)/bin/vivado
VPP := $(XILINX_VITIS)/bin/v++

# Specify 'hw_emu' for hardware emulation support instead of a bitfile
# Note, the Azure shell verison is not officially supported in hw_emu mode
TARGET := hw_emu

NAME := esi_image
SRC := hw
BUILD := build_$(TARGET)
TEMP := $(BUILD)/temp

# Limit number of Vivado jobs to avoid running out of memory. 0 is unlimited.
JOBS := 0

# Frequency of the kernel in MHz
FREQ := 150

# Optimization level (set to 0-3, s, or quick)
OPT := 1

# Toggle to automatically set custom options for running in Azure NP-series VMs
AZURE := true

XO_OUT := $(TEMP)/kernel.xo
LINK_OUT := $(BUILD)/$(NAME).link.xclbin
XCL_OUT := $(NAME).$(TARGET).xclbin
HOST_APP := $(BUILD)/host_app

VPPFLAGS = --config xrt_vitis.cfg
VPPFLAGS += --save-temps --kernel_frequency $(FREQ) -O $(OPT)
VPPFLAGS += --remote_ip_cache cache

# Platform must match the device + shell you're using
# For Azure NP-series, use the official Azure Shell
# For a local card or hw_emu mode, use the latest U250 XDMA Shell

ifeq ($(AZURE), true)
	ifeq ($(TARGET), hw_emu)
		PLATFORM := xilinx_u250_gen3x16_xdma_4_1_202210_1
	else
		PLATFORM := xilinx_u250_gen3x16_xdma_2_1_202010_1
	endif
# For Azure NP-series, output the routed netlist as a DCP instead of a bitstream!
VPPFLAGS += --advanced.param compiler.acceleratorBinaryContent=dcp
else
PLATFORM := xilinx_u250_gen3x16_xdma_4_1_202210_1
endif

VPPFLAGS += -t $(TARGET) --platform $(PLATFORM)
PACKAGE := $(BUILD)/package

ifneq ($(TARGET), hw)
VPPFLAGS += -g
endif

ifneq ($(JOBS), 0)
	VPPFLAGS += --jobs $(JOBS)
endif

device2xsa = $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))
XSA := $(call device2xsa, $(PLATFORM))

.PHONY: clean emconfig exec

all: $(XCL_OUT) emconfig

$(BUILD):
	mkdir -p $(BUILD)

$(TEMP):
	mkdir -p $(TEMP)

# Package everything into a Vitis compatible kernel (.xo format)
$(XO_OUT): $(TEMP)
	$(VIVADO) -mode batch -source $(SRC)/xrt_package.tcl -tclargs $(SRC) $(XO_OUT) $(TARGET) $(PLATFORM) $(XSA)

# Link Vitis system using the generated kernel for the chosen platform
$(LINK_OUT): $(XO_OUT) | $(BUILD)
	$(VPP) $(VPPFLAGS) -l --temp_dir $(TEMP) -o'$(LINK_OUT)' $(+)

# Build the xclbin
$(XCL_OUT): $(LINK_OUT)
	$(VPP) -p $(LINK_OUT) $(VPPFLAGS) --package.out_dir $(PACKAGE) -o $(XCL_OUT)

# Generate configuration for use with hw_emu mode
emconfig: $(BUILD)/emconfig.json
$(BUILD)/emconfig.json:
	$(XILINX_VITIS)/bin/emconfigutil --platform $(PLATFORM) --od $(BUILD)

clean:
	rm -rf $(BUILD) temp_kernel .Xil vivado* kernel *.jou *.log *.wdb *.wcfg *.protoinst *.csv

# Targets which only apply to image builds.
ifeq ($(TARGET), hw)

# Submit the image to Azure for attestation. Follows the instructions at:
# https://learn.microsoft.com/en-us/azure/virtual-machines/field-programmable-gate-arrays-attestation
IMAGE_AZ_BASENAME ?= $(NAME)_$(shell date +%s).hw
IMAGE_AZ_NAME := $(USER)_$(IMAGE_AZ_BASENAME)
azure: azure_creds $(IMAGE_AZ_NAME).azure.xclbin
azure_creds:
	@echo "*************************"
	@echo "* Getting Azure credentials. MUST 'az login' first!"
	@echo "*************************"
	@ if [ "${AZ_FPGA_SUB}" = "" ] || [ "${AZ_FPGA_STORAGE_ACCOUNT}" = "" ] || \
	     [ "${AZ_FPGA_STORAGE_CONTAINER}" = "" ]; then \
		@echo "** AZ_FPGA_SUB, AZ_FPGA_STORAGE_ACCOUNT, and AZ_FPGA_STORAGE_CONTAINER" \
		exit 1; \
	fi

	$(eval SAS_EXPIRY=$(shell date --date "now + 16hours" +"%Y-%m-%dT%0k:%MZ"))
	$(eval SAS=$(shell \
		az storage container generate-sas \
			--subscription $(AZ_FPGA_SUB) \
			--account-name $(AZ_FPGA_STORAGE_ACCOUNT) \
			--name $(AZ_FPGA_STORAGE_CONTAINER) \
			--https-only --permissions rwc --as-user --auth-mode login \
			--expiry $(SAS_EXPIRY) --output tsv))

$(IMAGE_AZ_NAME).azure.xclbin: azure_creds $(XCL_OUT) validate-fpgaimage.sh
	@echo "*************************"
	@echo "* Submitting job to Azure attestation."
	@echo "* This step WILL take a LONG time (between 30 mins and 1.5 hours)."
	@echo "*   Using name $(IMAGE_AZ_NAME)"
	@echo "*************************"

	az account set --subscription "$(AZ_FPGA_SUB)"
	az storage blob upload \
		--subscription $(AZ_FPGA_SUB) \
		--account-name $(AZ_FPGA_STORAGE_ACCOUNT) \
		--container-name $(AZ_FPGA_STORAGE_CONTAINER) \
	  --sas-token "$(SAS)" --overwrite \
		--name $(IMAGE_AZ_NAME).xclbin --file $(NAME).hw.xclbin

	bash validate-fpgaimage.sh --storage-account $(AZ_FPGA_STORAGE_ACCOUNT) \
														 --container $(AZ_FPGA_STORAGE_CONTAINER) \
														 --netlist-name $(IMAGE_AZ_NAME).xclbin \
														 --blob-container-sas "$(SAS)" \
														 --endpoint fpga-attestation-alternate-vitis.azurewebsites.net

	az storage blob download \
		--subscription $(AZ_FPGA_SUB) \
		--account-name $(AZ_FPGA_STORAGE_ACCOUNT) \
		--container-name $(AZ_FPGA_STORAGE_CONTAINER) \
		--sas-token "$(SAS)" \
		--name $(IMAGE_AZ_NAME).azure.xclbin --file $(IMAGE_AZ_NAME).azure.xclbin

validate-fpgaimage.sh:
	wget https://raw.githubusercontent.com/teqdruid/azhpc-fpga-attestation/refs/heads/jodemme/alt-url/scripts/validate-fpgaimage.sh

azpackage: $(NAME)_azpackage.tar.gz
$(NAME)_azpackage.tar.gz: $(IMAGE_AZ_NAME).azure.xclbin
	mkdir -p package
	cp -r runtime/* package
	cp $(IMAGE_AZ_NAME).azure.xclbin package/$(NAME)/$(NAME).hw.azure.xclbin
	cd package && tar -zcf ../$(NAME)_azpackage.tar.gz *
	cd ..
endif
