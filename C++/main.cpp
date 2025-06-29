#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

// 顶点缓冲区、着色器
// opengl是一个状态机

int main(void)
{
    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    glewInit();
    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    /* Loop until the user closes the window */
    float points[] = {
        0.0f, 0.0f, 0.0f, // Vertex 1 (X, Y, Z)
        1.0f, 0.0f, 0.0f, // Vertex 2 (X, Y, Z)
        0.5f, 1.0f, 0.0f  // Vertex 3 (X, Y, Z)
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
    glBindBuffer(GL_ARRAY_BUFFER, 0); // 解绑缓冲区
    //  声明着色器

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

    glfwTerminate();
    return 0;
}