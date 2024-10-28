configure_file(rtgtool.py ${CIRCT_TOOLS_DIR}/rtgtool.py)
add_custom_target(rtgtool SOURCES ${CIRCT_TOOLS_DIR}/rtgtool.py)
