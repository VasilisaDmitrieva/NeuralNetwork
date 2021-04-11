/* Vasilisa Dmitrieva, 10-6, 06.06.2019 */
#include <stdio.h>
#include <string.h>
#include "T02.H"
#include "IMAGE.H"
#include "LAYERS.H"

PARAM P[MAX_FC_LAY];
PARAM P8[MAX_PARAMS];

int ParamH[MAX_FC_LAY] = { 50, 10 }, ParamW[MAX_FC_LAY] = { 784, 50 };
int ParamH8[MAX_PARAMS] = { 5, 5, 500, 10 }, ParamW8[MAX_PARAMS] = { 5, 5, 800, 500 }, ParamC8[MAX_PARAMS] = { 20, 50, 1, 1 }, ParamL8[MAX_PARAMS] = { 1, 20, 1, 1 };
int ParamConv8[MAX_PARAMS] = { 1, 1, 0, 0 };

/* Load image function.
 * ARGUMENTS:
 *  - image:
 *    IMAGE *Im;
 *  - number of image in file:
 *    int Image;
 *  - file with images:
 *    FILE *F;
 *  - file with numbers:
 *    FILE *F1.
 * RETURNS: none.
 */
void ImageLoad_id2( IMAGE *Im, int Image, FILE *F, FILE *F1 )
{
  int i, j, NumOfRows = 28, NumOfColumns = 28;

  memset(Im, 0, sizeof(IMAGE));
  fseek(F, 16 + Image * NumOfColumns * NumOfRows, SEEK_SET);
  fseek(F1, 8 + Image, SEEK_SET);
  fread(&Im->Type, 1, 1, F1);
  for (i = 0; i < NumOfColumns; i++)
    for (j = 0; j < NumOfRows; j++)
      fread(&Im->Pixels[j][i], 1, 1, F);
  Im->W = NumOfRows;
  Im->H = NumOfColumns;
} /* End of 'ImageLoad_id2' functtion */

/* Save parametrs to file function.
 * ARGUMENTS:
 *  - parameters:
 *    PARAM P.
 * RETURNS: none.
 */
void SaveParams( void )
{
  int i, m;
  FILE *F;

  if (fopen_s(&F, "params.txt", "wb"))
    return;
  for (m = 0; m < MAX_PARAMS; m++)
  {
    for (i = 0; i < P8[m].W.H * P8[m].W.W * P8[m].W.Cnt * P8[m].W.L; i++)
      fwrite(&P8[m].W.X[i], sizeof(double), 1, F);
    for (i = 0; i < P8[m].B.H * P8[m].B.Cnt; i++)
      fwrite(&P8[m].B.X[i], sizeof(double), 1, F);
  }
  fclose(F);
} /* End of 'SaveParams' function */

/* Get parametrs from file function.
 * ARGUMENTS:
 *  - parameters:
 *    PARAM *P.
 * RETURNS: none.
 */
void GetParams( void )
{
  int i, m;
  FILE *F;

  if (fopen_s(&F, "params.txt", "rb"))
    return;
  for (m = 0; m < MAX_PARAMS; m++)
  {
    for (i = 0; i < P8[m].W.H * P8[m].W.W * P8[m].W.Cnt * P8[m].W.L; i++)
      fread(&P8[m].W.X[i], sizeof(double), 1, F);
    for (i = 0; i < P8[m].B.H * P8[m].B.Cnt; i++)
      fread(&P8[m].B.X[i], sizeof(double), 1, F);
  }
  fclose(F);
} /* End of 'GetParams' function */

/* Test processing of images function.
 * ARGUMENTS:
 *  - number of processing images:
 *    int NumOfImage1;
 *  - number of testing images:
 *    int NumOfImage2.
 * RETURNS: none.
 */
void TakeImage8( int NumOfImage1, int NumOfImage2 )
{
  int i, Num, j, k, pos = 0, m;
  IMAGE Im;
  FILE *F, *F1, *Ft, *Ft1;
  VEC Y[9], Y1, Cnt;
  MATR M;
  double max;

  VecSet(&Cnt, 10, 1, 1, 1);
  MatrSet(&M, 10, 10, 1);

  VecSet(&Y1, 10, 1, 1, 1);
  VecSet(&Y[0], 28, 28, 1, 1);
  VecSet(&Y[1], 24, 24, 20, 1);
  VecSet(&Y[2], 12, 12, 20, 1);
  VecSet(&Y[3], 8, 8, 50, 1);
  VecSet(&Y[4], 4, 4, 50, 1);
  VecSet(&Y[5], 500, 1, 1, 1);
  VecSet(&Y[6], 500, 1, 1, 1);
  VecSet(&Y[7], 10, 1, 1, 1);
  VecSet(&Y[8], 10, 1, 1, 1);

  MatrClear(&M, 0);

  VecClear(&Cnt, 0);

  for (i = 0; i < MAX_PARAMS; i++)
    ParamSet(&P8[i], ParamH8[i], ParamW8[i], ParamC8[i], ParamL8[i], ParamConv8[i]);

  if (fopen_s(&F, "train-images.idx3-ubyte", "rb"))
    return;
  if (fopen_s(&F1, "train-labels.idx1-ubyte", "rb"))
    return;
  if (fopen_s(&Ft, "t10k-images.idx3-ubyte", "rb"))
    return;
  if (fopen_s(&Ft1, "t10k-labels.idx1-ubyte", "rb"))
    return;
  
  for (i = 0; i < MAX_PARAMS; i++)
    ParamInit(&P8[i]);
    
  GetParams();
  
  printf("Training neural network:\n");
  for (i = 0; i < NumOfImage1; i++)
  {
    Num = (rand() * 239 + i + rand() * 47) % 60000;
    ImageLoad_id2(&Im, Num, F, F1);
    FindNum8L(Im, P8);
  }
  
  printf("Training neural network ('+' - guess; '-' - not guess):\n");
  for (m = 0; m < NumOfImage2; m++)
  {
    Num = m % 10000;
    ImageLoad_id2(&Im, Num, Ft, Ft1);

    for (j = 0; j < 28 * 28; j++)
      Y[0].X[j] = (double)Im.Pixels[(j % 28)][j / 28] / 255;
    ConvolutionF(&Y[1], Y[0], P8[0]);
    MaxPoolF(&Y[2], Y[1], 2);
    ConvolutionF(&Y[3], Y[2], P8[1]);
    MaxPoolF(&Y[4], Y[3], 2);
    FullyConnectedF(&Y[5], Y[4], P8[2]);
    ReLuF(&Y[6], Y[5]);
    FullyConnectedF(&Y[7], Y[6], P8[3]);
    SoftMaxF(&Y[8], Y[7]);
    max = Y[8].X[0];
    k = 0;
    for (j = 1; j < 10; j++)
      if (Y[8].X[j] > max)
        max = Y[8].X[j], k = j;
    Cnt.X[Im.Type]++;
    M.X[Im.Type][k]++;
    if (k == Im.Type)
      printf("+"), pos++;
    else
      printf("-");
  }
  SaveParams();
  printf("\n");
  printf("Table of numbers guessed by neural network:\n");
  for (i = 0; i < 10; i++)
    printf("    %i     ", i);
  printf("\n");
  for (i = 0; i < 10; i++)
  {
    printf("%i  ", i);
    for (j = 0; j < 10; j++)
      if (Cnt.X[j] == 0)
        printf("%f  ", 0.);
      else
        printf("%f  ", (double)M.X[j][i] / Cnt.X[j] * 100);
    printf("\n");
  }
  printf("Percent of correctly guessed numbers: %f \n", (double)pos / NumOfImage2 * 100);

  fclose(F);
  fclose(F1);
  fclose(Ft);
  fclose(Ft1);
  free(Cnt.X);
  free(Y1.X);
  for (i = 0; i < 9; i++)
    free(Y[i].X);
  free(M.X);
  for (i = 0; i < MAX_PARAMS; i++)
  {
    free(P8[i].B.X);
    free(P8[i].W.X);
  }
  for (i = 0; i < MAX_FC_LAY; i++)
  {
    free(P[i].B.X);
    free(P[i].W.X);
  }
} /* End of 'TakeImage8' function */

/* END OF 'IMAGE.C' FILE */
