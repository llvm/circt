if(CIRCT_LEC_ENABLED)
  add_circt_verification_library(CIRCTSMT
    Solver.cpp
    Circuit.cpp
    LogicExporter.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS
    ${Z3_LIBRARIES}
    
    LINK_LIBS PUBLIC
    MLIRTransforms
    MLIRTranslateLib
    CIRCTComb
    CIRCTHW
    CIRCTSupport
  )
  target_include_directories(CIRCTSMT
    PRIVATE
    ${Z3_INCLUDE_DIR}
    ${Z3_CXX_INCLUDE_DIRS}
  )
endif()
