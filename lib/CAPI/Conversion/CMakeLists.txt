set(LLVM_OPTIONAL_SOURCES
  SeqToSV.cpp
)
add_mlir_public_c_api_library(CIRCTCAPISeqToSV
  SeqToSV.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSeqToSV
  )
