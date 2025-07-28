cmake_minimum_required(VERSION 3.10.0)
project(GOHSTRUNNER VERSION 0.1.0 LANGUAGES C CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置编译选项
if (MSVC)
    # Visual Studio下的编译选项
    add_compile_options(/W4) # /W4表示警告级别4
else()
    # GCC或Clang下的编译选项
    add_compile_options(-Wall -Wextra -pedantic) # -Wall表示开启所有警告，-Wextra表示额外警告，-pedantic表示严格遵守标准
endif()

# 设置字符集，解决4819警告
if(MSVC)
add_compile_options("/source-charset:utf-8")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}) # 调试模式输出到源代码目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}) # 发布模式输出到源代码目录

# 设置项目路径
set(TARGET ${PROJECT_NAME}-${CMAKE_SYSTEM_NAME})

# 第三方库无CMAKE配置文件，手动指定路径
# if (WIN32)
# # windows下SDL2的路径
#     include_directories("E:/DEV/Game/SDL/include") # windows目录复制过来\要变为/
#     link_directories("E:/DEV/Game/SDL/lib/x64") 
# elseif (UNIX)
# # linux下SDL2的路径
#     include_directories(/usr/include/SDL2)
#     link_directories(/usr/lib/x86_64-linux-gnu)
# elseif (APPLE)
# # macOS下SDL2的路径
#     include_directories(/usr/local/include/SDL2)
#     link_directories(/usr/local/lib)
# endif()

# 使用find_package查找SDL2
find_package(SDL3 REQUIRED)
find_package(SDL3_image REQUIRED)
find_package(SDL3_mixer REQUIRED)
find_package(SDL3_ttf REQUIRED)
find_package(glm REQUIRED)

add_executable(${TARGET} 
                "src/main.cpp"
                
)

# 添加SDL2的头文件目录,手动指定的时候，不需要SDL2::
target_link_libraries(${TARGET}
                        ${SDL3_LIBRARIES}
                        SDL3_image::SDL3_image
                        SDL3_mixer::SDL3_mixer
                        SDL3_ttf::SDL3_ttf
                        glm::glm
                        )

