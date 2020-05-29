//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
//#include <math.h>
//
//#ifdef _WIN32
//#  define WINDOWS_LEAN_AND_MEAN
//#  define NOMINMAX
//#  include <windows.h>
//#endif
//
//// OpenGL Graphics includes
//#include <helper_gl.h>
//#include <freeglut.h>
//
//// includes, cuda
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
//
//// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
//
//// CUDA helper functions
//#include <helper_cuda.h>         // helper functions for CUDA error check
//
//#include <vector_types.h>
//
//const unsigned int window_width = 512;
//const unsigned int window_height = 512;
//
//const unsigned int mesh_width = 256;
//const unsigned int mesh_height = 256;
//
//// vbo variables
//GLuint vbo;
//struct cudaGraphicsResource* cuda_vbo_resource;
//void* d_vbo_buffer = NULL;
//
//float g_fAnim = 0.0;
//
//// mouse controls
//int mouse_old_x, mouse_old_y;
//int mouse_buttons = 0;
//float rotate_x = 0.0, rotate_y = 0.0;
//float translate_z = -3.0;
//
//StopWatchInterface* timer = nullptr;
//int* pArgc = nullptr;
//char** pArgv = nullptr;
//
//// Auto-Verification Code
//int fpsCount = 0;        // FPS count for averaging
//int fpsLimit = 1;        // FPS limit for sampling
//int g_Index = 0;
//float avgFPS = 0.0f;
//unsigned int frameCount = 0;
//unsigned int g_TotalErrors = 0;
//bool g_bQAReadback = false;
//
//const char* sSDKsample = "simpleGL (VBO)";
//
//#define REFRESH_DELAY     10 //ms
//
//bool initGL(int* argc, char** argv);
//void computeFPS();
//void display();
//void keyboard(unsigned char key, int /*x*/, int /*y*/);
//void mouse(int button, int state, int x, int y);
//void motion(int x, int y);
//void timerEvent(int value);
//
//////////////////////////////////////////////////////////////////////////////////
//// Program main
//////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char** argv)
//{
//    char* ref_file = NULL;
//
//    pArgc = &argc;
//    pArgv = argv;
//
//#if defined(__linux__)
//    setenv("DISPLAY", ":0", 0);
//#endif
//
//    printf("%s starting...\n", sSDKsample);
//
//    if (argc > 1)
//    {
//        if (checkCmdLineFlag(argc, (const char**)argv, "file"))
//        {
//            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
//            getCmdLineArgumentString(argc, (const char**)argv, "file", (char**)&ref_file);
//        }
//    }
//
//    printf("\n");
//
//    initGL(&argc, argv);
//
//    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
//    //exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
//}
//
//void computeFPS()
//{
//    frameCount++;
//    fpsCount++;
//
//    if (fpsCount == fpsLimit)
//    {
//        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
//        fpsCount = 0;
//        fpsLimit = (int)MAX(avgFPS, 1.f);
//
//        sdkResetTimer(&timer);
//    }
//
//    char fps[256];
//    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
//    glutSetWindowTitle(fps);
//}
//
//void display()
//{
//    sdkStartTimer(&timer);
//
//    // run CUDA kernel to generate vertex positions
//    //runCuda(&cuda_vbo_resource);
//
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//    // set view matrix
//    glMatrixMode(GL_MODELVIEW);
//    glLoadIdentity();
//    glTranslatef(0.0, 0.0, translate_z);
//    glRotatef(rotate_x, 1.0, 0.0, 0.0);
//    glRotatef(rotate_y, 0.0, 1.0, 0.0);
//
//    // render from the vbo
//    glBindBuffer(GL_ARRAY_BUFFER, vbo);
//    glVertexPointer(4, GL_FLOAT, 0, 0);
//
//    glEnableClientState(GL_VERTEX_ARRAY);
//    glColor3f(1.0, 0.0, 0.0);
//    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
//    glDisableClientState(GL_VERTEX_ARRAY);
//
//    glutSwapBuffers();
//
//    g_fAnim += 0.01f;
//
//    sdkStopTimer(&timer);
//    computeFPS();
//}
//
//////////////////////////////////////////////////////////////////////////////////
////! Keyboard events handler
//////////////////////////////////////////////////////////////////////////////////
//void keyboard(unsigned char key, int /*x*/, int /*y*/)
//{
//    switch (key)
//    {
//    case (27):
//#if defined(__APPLE__) || defined(MACOSX)
//        exit(EXIT_SUCCESS);
//#else
//        glutDestroyWindow(glutGetWindow());
//        return;
//#endif
//    }
//}
//
//////////////////////////////////////////////////////////////////////////////////
////! Mouse event handlers
//////////////////////////////////////////////////////////////////////////////////
//void mouse(int button, int state, int x, int y)
//{
//    if (state == GLUT_DOWN)
//    {
//        mouse_buttons |= 1 << button;
//    }
//    else if (state == GLUT_UP)
//    {
//        mouse_buttons = 0;
//    }
//
//    mouse_old_x = x;
//    mouse_old_y = y;
//}
//
//void motion(int x, int y)
//{
//    float dx, dy;
//    dx = (float)(x - mouse_old_x);
//    dy = (float)(y - mouse_old_y);
//
//    if (mouse_buttons & 1)
//    {
//        rotate_x += dy * 0.2f;
//        rotate_y += dx * 0.2f;
//    }
//    else if (mouse_buttons & 4)
//    {
//        translate_z += dy * 0.01f;
//    }
//
//    mouse_old_x = x;
//    mouse_old_y = y;
//}
//
//void timerEvent(int value)
//{
//    if (glutGetWindow())
//    {
//        glutPostRedisplay();
//        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
//    }
//}
//
//bool initGL(int* argc, char** argv)
//{
//    glutInit(argc, argv);
//    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
//    glutInitWindowSize(window_width, window_height);
//    glutCreateWindow("Cuda GL Interop (VBO)");
//    glutDisplayFunc(display);
//    glutKeyboardFunc(keyboard);
//    glutMotionFunc(motion);
//    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
//
//    // initialize necessary OpenGL extensions
//    if (!isGLVersionSupported(2, 0))
//    {
//        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
//        fflush(stderr);
//        return false;
//    }
//
//    // default initialization
//    glClearColor(0.0, 0.0, 0.0, 1.0);
//    glDisable(GL_DEPTH_TEST);
//
//    // viewport
//    glViewport(0, 0, window_width, window_height);
//
//    // projection
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);
//
//    SDK_CHECK_ERROR_GL();
//
//    return true;
//}
//
