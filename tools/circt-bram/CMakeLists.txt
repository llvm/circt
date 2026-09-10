get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)

add_llvm_executable(bram-opt main.cpp)

target_link_libraries(bram-opt PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        HLSInterfaces
        HLSPasses
        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRRegisterAllExtensions
        MLIROptLib
)