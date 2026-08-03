include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(cimrt STATIC
  src/simulator/simulator.cpp
  src/erbium/erbium_backend.cpp
)

target_include_directories(cimrt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/simulator
)

target_link_libraries(cimrt PUBLIC
  MLIRCIMTarget
  LLVMSupport
)
