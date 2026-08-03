configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CIM_TEST_DEPENDS
  FileCheck
  cim-opt
)

add_lit_testsuite(check-cim-opt "Running the cim-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIM_TEST_DEPENDS}
)
set_target_properties(check-cim-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIM_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CIM_TEST_DEPENDS})
