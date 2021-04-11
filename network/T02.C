/* Vasilisa Dmitrieva, 10-6, 06.06.2019 */

#include <glut.h>
#include <stdio.h>
#include "T02.H"
#include "IMAGE.H"
#include "NUMBER.H"
#include "LAYERS.H"

/* Array of pixels */
unsigned char Frame[FRAME_H][FRAME_W][3];

/* Zoom parameter */
double Zoom = 1.5;

/* Set pixel function
 * ARGUMENTS:
 *  Coordination pixel:
 *  -  int X, Y;
 *  Color:
 *  -  int Color;
 * RETURNS:
 *  None.
 */
void PutPixel(int X, int Y, int Color)
{
    if (X < 0 || Y < 0 || X >= FRAME_W || Y >= FRAME_H)
        return;
    Frame[Y][X][0] = Color & 0xFF;
    Frame[Y][X][1] = (Color >> 8) & 0xFF;
    Frame[Y][X][2] = (Color >> 16) & 0xFF;
} /* End of 'PutPixel' function */

/* Print iformation function
 * ARGUMENTS:
 *  None;
 * RETURNS:
 *  None.
 */
void printInfo(void)
{
    printf("You can try write number and Neural networks will try to guess the number.\n"
        "Draw with the right mouse button.\n"
        "'1' - select an area where a number will be written.\n"
        "'+', '-' - zoom.\n"
        "'f' - neural network try to guess the number.\n"
        "'c' - clear screen.\n");
} /* End of 'printInfo' function */


/* Mouse motion function.
 * ARGUMENTS:
 *   - coordinates mouse:
 *       int X, Y;
 * RETURNS: None.
 */
void MouseMotion(int X, int Y)
{
    int i;

    for (i = 0; i < 3; i++)
        Frame[(int)(Y / Zoom)][(int)(X / Zoom)][i] = 255;
} /* End of 'MouseMotion' function */

/* Clear screen function
 * ARGUMENTS:
 *   None;
 * RETURNS:
 *   None.
 */
void ClearArea(void)
{
    int i, j, k;

    for (i = 0; i < FRAME_H; i++)
        for (j = 0; j < FRAME_W; j++)
            for (k = 0; k < 3; k++)
                Frame[i][j][k] = 0;
} /* End of 'ClearArea' function */

/* Keyboard function
 * ARGUMENTS:
 *  - pressed key:
 *    unsigned char Key;
 *  - mouse coordinates:
 *    int X, Y.
 * RETURNS: none.
 */
void Keyboard( unsigned char Key, int X, int Y )
{
  int i, j;
  static int k = 0;
  VEC X1;

  if (Key == 27)
    exit(30);
  else if (Key == '+')
    Zoom *= 1.05;
  else if (Key == '-')
    Zoom /= 1.05;
  else if (Key == 'c')
    ClearArea();
  else if (Key == '1')
  {
    int i;
    for (i = 0; i < 29; i++)
    {
        PutPixel(29, i, 0x00FFFF);
        PutPixel(i, 29, 0x00FFFF);
    }
  }
  else if (Key == 'f')
  {
    VecSet(&X1, 1, 28 * 28, 1, 1);
    for (i = 0; i < 28; i++)
      for (j = 0; j < 28; j++)
        X1.X[(i * 28) + j] = Frame[i][j][0];
    printf("Your number is %i\n", FindNum(X1));
    free(X1.X);
  }
} /* End of 'Keyboard' function */

/* Display function
 * ARGUMENTS: none.
 * RETURNS: none.
 */
void Display( void )
{
  glClearColor(0.3, 0.5, 0.7, 1);
  glClear(GL_COLOR_BUFFER_BIT);

  glRasterPos2d(-1, 1);
  glPixelZoom(Zoom, -Zoom);
  glDrawPixels(FRAME_W, FRAME_H, GL_BGR_EXT, GL_UNSIGNED_BYTE, Frame);

  glFinish();
  glutSwapBuffers();
  glutPostRedisplay();
} /*End of 'Display' function */

/* Main programm function
 * ARGUMENTS:
 *  - arguments of command line:
 *    int argc, char *argv[].
 * RETURNS: none.
 */
int main( int argc, char *argv[] )
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowPosition(0, 0);
  glutInitWindowSize(512, 512);
  glutCreateWindow("abc");

  TakeImage8(200, 1000);

  printInfo();

  glutDisplayFunc(Display);
  glutKeyboardFunc(Keyboard);
  glutMotionFunc(MouseMotion);

  glutMainLoop();
  return 0;
} /*End of 'main' function */

/* END OF 'T02.C' FILE */
