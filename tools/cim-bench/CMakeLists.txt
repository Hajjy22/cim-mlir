add_llvm_executable(cim-bench cim-bench.cpp)

llvm_update_compile_flags(cim-bench)

target_link_libraries(cim-bench PRIVATE
  LLVMSupport
)
