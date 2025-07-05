#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

// 顶点缓冲区、着色器
// opengl是一个状态机

//  编译着色器
static unsigned int CompileShader(unsigned int type, const std::string &source)
{
    unsigned int id = glCreateShader(type);
    const char *src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);
    //  TODE Error handing
    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char *message = (char *)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cerr << "Failed to compile shader!" << std::endl;
        std::cerr << message << std::endl;
        glDeleteShader(id);
        return 0;
    }
    return id;
}

// 创建着色器
// vertexShader 是顶点着色器的代码, fragmentShader 是片段着色器的代码
static unsigned int CreateShader(const std::string &vertexShader, const std::string &fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);
    //  删除着色器, 因为已经链接到程序中
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

int main(void)
{
    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;
    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        return -1;
    }
    unsigned char *glVersion = 0;
    // GLCall(glVersion = (unsigned char*)glGetString(GL_VERSION));
    /* Loop until the user closes the window */
    float points[6] = {
        0.0f, 0.0f, // Vertex 1 (X, Y, Z)
        1.0f, 0.0f, // Vertex 2 (X, Y, Z)
        0.5f, 1.0f  // Vertex 3 (X, Y, Z)
    };
    //  声明顶点缓冲区
    unsigned int buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    //  将顶点数据复制到缓冲区对象中
    //  GL_STATIC_DRAW表示数据不会被改变
    //  GL_DYNAMIC_DRAW表示数据会被改变
    //  GL_STREAM_DRAW表示数据会被频繁改变
    //  points是一个数组，包含了三个顶点的坐标
    //  每个顶点有两个坐标（X, Y）
    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), points, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);                                          // 启用顶点属性数组
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0); // 设置顶点属性指针

    glBindBuffer(GL_ARRAY_BUFFER, 0); // 解绑缓冲区
                                      //  声明着色器
    std::string vertexShader =
        "#version 330 core\n"
        "\n"
        "layout(location = 0) in vec4 position;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = position;\n"
        "}\n";
    std::string fragmentShader =
        "#version 330 core\n"
        "\n"
        "layout(location = 0) out vec4 color;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    color = vec4(1.0, 0.0, 0.0, 1.0);\n" // 设置颜色为红色
        "}\n";
    unsigned int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader); // 使用着色器程序

    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /*glBegin(GL_TRIANGLES);
        glVertex2f(points[0], points[1]);
        glVertex2f(points[2], points[3]);
        glVertex2f(points[4], points[5]);*/
        glDrawArrays(GL_TRIANGLES, 0, 3); // 绘制三角形
        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}