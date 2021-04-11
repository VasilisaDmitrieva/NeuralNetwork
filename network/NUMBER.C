/* Vasilisa Dmitrieva, 10-6, 08.06.2019 */
#include <math.h>
#include <stdio.h>

#include "IMAGE.H"
#include "LAYERS.H"

/* Initialization of parametrs function
 * ARGUMENTS:
 *  - parametrs:
 *    PARAM *P1.
 * RETURNS: none.
 */
void ParamInit( PARAM *P1 )
{
  int i;

  for (i = 0; i < P1->W.H * P1->W.W * P1->W.Cnt * P1->W.L; i++)
    P1->W.X[i] = R1();
  for (i = 0; i < P1->B.H * P1->B.Cnt; i++)
    P1->B.X[i] = R1();
} /* End of 'ParamInit' function */

/* Processing of image function
 * ARGUMENTS:
 *  - image:
 *    IMAGE Im;
 *  - parametrs:
 *    PARAM *P.
 * RETURNS: none.
 */
void FindNum8L( IMAGE Im, PARAM *P )
{
  int i, j, k;
  double L, e = 1, alpha = 0.00001, mu = 0.95, max;
  static VEC Y[9], G[7], Y1, H, B, T1, T2, Test1, Test2;
  static int fl = 1, Num = 0;
  static double loss;
  static PARAM P1[MAX_PARAMS], P2;

  if (fl)
  {
    VecSet(&H, 5, 5, 20, 1);
    VecSet(&B, 1, 1, 20, 1);

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
    VecSet(&T2, P8[3].W.W, P8[3].W.H, 1, 1);
    VecSet(&T1, P8[2].W.W, P8[2].W.H, 1, 1);

    VecSet(&G[0], 10, 1, 1, 1);
    VecSet(&G[1], 500, 1, 1, 1);
    VecSet(&G[2], 500, 1, 1, 1);
    VecSet(&G[3], 800, 1, 1, 1);
    VecSet(&G[4], 8, 8, 50, 1);
    VecSet(&G[5], 12, 12, 20, 1);
    VecSet(&G[6], 24, 24, 20, 1);

    VecSet(&Test1, 6, 6, 1, 1);
    VecSet(&Test2, 4, 4, 1, 1);
    ParamSet(&P2, 3, 3, 1, 1, 1);

    for (i = 0; i < MAX_PARAMS; i++)
      ParamSet(&P1[i], ParamH8[i], ParamW8[i], ParamC8[i], ParamL8[i], ParamConv8[i]);
  }

  for (i = 0; i < 36; i++)
    Test1.X[i] = i;
  for (i = 0; i < 9; i++)
    P2.W.X[i] = i;
  P2.B.X[0] = 1;
  ConvolutionF(&Test2, Test1, P2);

  VecClear(&Y1, 0);

  Y1.X[Im.Type] = 1;

  for (i = 0; i < MAX_PARAMS; i++)
    ParamClear(&P1[i], 0);

  for (j = 0; j < 28 * 28; j++)
    Y[0].X[j] = (double)Im.Pixels[(j % 28)][j / 28] / 255;


  for (i = 0; i < MAX_PARAMS; i++)
    ParamAdd(&P[i], P1[i], mu * (-alpha));

  ConvolutionF(&Y[1], Y[0], P[0]);
  MaxPoolF(&Y[2], Y[1], 2);
  ConvolutionF(&Y[3], Y[2], P[1]);
  MaxPoolF(&Y[4], Y[3], 2);
  Y[4].Cnt = 1;
  Y[4].H = 800;
  Y[4].W = 1;
  FullyConnectedF(&Y[5], Y[4], P[2]);
  Y[4].Cnt = 50;
  Y[4].H = 4;
  Y[4].W = 4;
  ReLuF(&Y[6], Y[5]);
  FullyConnectedF(&Y[7], Y[6], P[3]);
  for (i = 0; i < 10; i++)
    Y[7].X[i] /= 100;
  SoftMaxF(&Y[8], Y[7]);
  L = Loss(Y1, Y[8]);

  max = Y[8].X[0];
  k = 0;
  for (j = 1; j < 10; j++)
      if (Y[8].X[j] > max)
          max = Y[8].X[j], k = j;
  
  SoftMaxB(&G[0], Y1, Y[8]);
  
  VecMulVecEx(&P1[3].W, Y[6], G[0]);
  for (i = 0; i < G[0].H * G[0].W; i++)
    P1[3].B.X[i] = G[0].X[i];
  
  MatrTranspose(&T2, P[3].W);
  FullyConnectedB(&G[1], T2, G[0]);
  ReLuB(&G[2], G[1], Y[5]);
  
  VecMulVecEx(&P1[2].W, Y[4], G[2]);
  for (i = 0; i < G[2].H * G[2].W; i++)
    P1[2].B.X[i] = G[2].X[i];
  
  MatrTranspose(&T1, P[2].W);
  G[3].H = 800;
  G[3].W = 1;
  G[3].Cnt = 1;
  FullyConnectedB(&G[3], T1, G[2]);
  G[3].H = 4;
  G[3].W = 4;
  G[3].Cnt = 50;
  MaxPoolB(&G[4], Y[3], Y[4], G[3], 2);
  
  ConvolutionG(&P1[1], Y[2], G[4]);
  
  ConvolutionB(&G[5], G[4], P[1].W);
  MaxPoolB(&G[6], Y[1], Y[2], G[5], 2);
  
  ConvolutionG(&P1[0], Y[0], G[6]);

  for (i = 0; i < MAX_PARAMS; i++)
    ParamAdd(&P[i], P1[i], -alpha);
  fl = 0;
  loss += L;
  Num++;
  printf("%i)  ", Num);
  printf("Loss: %f  ", loss / Num);
  printf("Neural network: %i  Real number: %i\n", k, Im.Type);
} /* End of 'FindNum8L' function */

/* Processing of image function
 * ARGUMENTS:
 *  - image:
 *    IMAGE Im;
 *  - parametrs:
 *    PARAM *P.
 * RETURNS: 
 *  (int) - number 0-9
 * .
 */
int FindNum(VEC V) 
{
  int i, j, k, pos = 0;
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

  GetParams();

  for (j = 0; j < 28 * 28; j++)
    Y[0].X[j] = (double)V.X[j] / 255;

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
  return k;
} /* End of 'FindNum' function */

/* END OF 'NUMBER.C' FILE */
