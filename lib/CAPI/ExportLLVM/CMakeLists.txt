add_circt_public_c_api_library(CIRCTCAPIExportLLVM
  ExportLLVM.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
)
