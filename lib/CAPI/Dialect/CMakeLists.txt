# TODO: Make the check source feature optional as an argument on *_add_library.
set(LLVM_OPTIONAL_SOURCES
  Arc.cpp
  CHIRRTL.cpp
  Comb.cpp
  DC.cpp
  Debug.cpp
  Emit.cpp
  ESI.cpp
  FIRRTL.cpp
  FSM.cpp
  HW.cpp
  HWArith.cpp
  Handshake.cpp
  Kanagawa.cpp
  LLHD.cpp
  LTL.cpp
  MSFT.cpp
  Moore.cpp
  OM.cpp
  Pipeline.cpp
  RTG.cpp
  RTGTest.cpp
  SV.cpp
  Seq.cpp
  Sim.cpp
  Synth.cpp
  Verif.cpp
)



add_circt_public_c_api_library(CIRCTCAPIArc
  Arc.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTArc
  CIRCTArcTransforms
)

add_circt_public_c_api_library(CIRCTCAPIComb
  Comb.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTComb
  CIRCTCombTransforms
)

add_circt_public_c_api_library(CIRCTCAPIDC
  DC.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTDC
  CIRCTDCTransforms
)

add_circt_public_c_api_library(CIRCTCAPIDebug
  Debug.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTDebug
)

add_circt_public_c_api_library(CIRCTCAPIESI
  ESI.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTESI
)

add_circt_public_c_api_library(CIRCTCAPIFIRRTL
  FIRRTL.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTFIRRTL
  CIRCTImportFIRFile
)

add_circt_public_c_api_library(CIRCTCAPICHIRRTL
  CHIRRTL.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTFIRRTL
)

add_circt_public_c_api_library(CIRCTCAPIMSFT
  MSFT.cpp

  DEPENDS
  MLIRTransforms

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTransforms
  CIRCTMSFT
  CIRCTMSFTTransforms
)

add_circt_public_c_api_library(CIRCTCAPIHW
  HW.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTHW
  CIRCTHWTransforms
)

add_circt_public_c_api_library(CIRCTCAPILLHD
  LLHD.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTLLHD
)

add_circt_public_c_api_library(CIRCTCAPIMoore
  Moore.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTMoore
)

add_circt_public_c_api_library(CIRCTCAPIOM
  OM.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTOM
  CIRCTOMEvaluator
)

add_circt_public_c_api_library(CIRCTCAPIKanagawa
  Kanagawa.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTKanagawa
  CIRCTKanagawaTransforms
)

add_circt_public_c_api_library(CIRCTCAPIPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTPipelineOps
  CIRCTPipelineTransforms
)

add_circt_public_c_api_library(CIRCTCAPIRTG
  RTG.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTRTGDialect
  CIRCTRTGTransforms
)

if(CIRCT_INCLUDE_TESTS)
  add_circt_public_c_api_library(CIRCTCAPIRTGTest
    RTGTest.cpp

    LINK_LIBS PUBLIC
    MLIRCAPIIR
    CIRCTRTGTestDialect
  )
endif()

add_circt_public_c_api_library(CIRCTCAPISeq
  Seq.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSeq
  CIRCTSeqTransforms
)

add_circt_public_c_api_library(CIRCTCAPISV
  SV.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSV
  CIRCTSVTransforms
)

add_circt_public_c_api_library(CIRCTCAPIFSM
  FSM.cpp

  DEPENDS
  CIRCTTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTFSM
  CIRCTFSMTransforms
  CIRCTFSMToSV
  CIRCTFSMToSMT
  CIRCTFSMToCore
)

add_circt_public_c_api_library(CIRCTCAPIHandshake
  Handshake.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTHandshakeToHW
  CIRCTCFToHandshake
)

add_circt_public_c_api_library(CIRCTCAPIHWArith
  HWArith.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTHWArith
  CIRCTHWArithToHW
)

add_circt_public_c_api_library(CIRCTCAPIVerif
  Verif.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTVerif
  CIRCTVerifToSV
)

add_circt_public_c_api_library(CIRCTCAPILTL
  LTL.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTLTL
)

add_circt_public_c_api_library(CIRCTCAPIEmit
  Emit.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTEmit
)

add_circt_public_c_api_library(CIRCTCAPISynth
  Synth.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSynth
  CIRCTSynthAnalysis
  CIRCTSynthTransforms
  CIRCTCAPISupport
)

add_circt_public_c_api_library(CIRCTCAPISim
  Sim.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSim
)
