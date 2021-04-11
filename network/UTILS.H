/* Vasilisa Dmitrieva, 10-6, 14.06.2019 */

#ifndef __UTILS_H_
#define __UTILS_H_

#include <math.h>

#define MALLOC(N) malloc(## N ## * sizeof(double))

#define E 2.718281828

/* Max Fully Connection layers number */
#define MAX_FC_LAY 2

/* Max parameters number */
#define MAX_PARAMS 4

/* Vector representation structure */
typedef struct tagVEC
{
  double *X;   /* Vector components */
  int W, H; /* Vector size */
  int Cnt;  /* Vector count */
  int L;    /* Layers count */
} VEC;

/* Matrix representation structure */
typedef struct tagMATR
{
  double **X;  /* Matrix components */
  int W, H; /* Matrix size */
  int Cnt;  /* Matrix count */
} MATR;

/* Parametrs representation structure */
typedef struct tagPARAM
{
  VEC W;    /* Matrix of weights */
  VEC B;    /* Bias vector */
} PARAM;

/* Vector setting function.
 * ARGUMENTS:
 *  - pointer to setting vector:
 *    VEC *V;
 *  - size of setting vector:
 *    int W, H;
 *  - number of setting vector:
 *    int Cnt;
 *  - layers of setting vector:
 *    int L.
 * RETURNS: none.
 */
__inline void VecSet( VEC *V, int H, int W, int Cnt, int L )
{
  V->H = H;
  V->W = W;
  V->Cnt = Cnt;
  V->L = L;
  V->X = MALLOC(W * H * Cnt);
} /* End of 'VecSet' function */

/* Matrix setting function.
 * ARGUMENTS:
 *  - pointer to setting matrix:
 *    MATR *M;
 *  - size of setting matrix:
 *    int W, H;
 *  - number of setting matrix:
 *    int Cnt.
 * RETURNS: none.
 */
__inline void MatrSet( MATR *M, int H, int W, int Cnt )
{
  int i;

  M->H = H;
  M->W = W;
  M->Cnt = Cnt;
  M->X = (double**)MALLOC(H * Cnt);
  for (i = 0; i < H * Cnt; i++)
    M->X[i] = MALLOC(W);
} /* End of 'MatrSet' function */

/* Get matr value by coordinates function.
 * ARGUMENTS:
 *  - vector:
 *    VEC W;
 *  - coordinates:
 *    int x, y, cnt.
 * RETURNS: (double) vector value.
 */
__inline double GetMatr( VEC* W, int X, int Y, int Cnt )
{
  return W->X[Cnt * W->W * W->H + Y * W->W + X];
} /* End of 'GetMatr' function */

/* Get matr value by coordinates function.
 * ARGUMENTS:
 *  - pointer to vector:
 *    VEC *W;
 *  - value:
 *    double Val;
 *  - coordinates:
 *    int x, y, cnt.
 * RETURNS: none.
 */
__inline void PutMatr( VEC *W, double Val, int X, int Y, int Cnt )
{
  W->X[Cnt * W->W * W->H + Y * W->W + X] = Val;
} /* End of 'GetMatr' function */

/* Get filter value by coordinates function.
 * ARGUMENTS:
 *  - filter:
 *    VEC Y;
 *  - coordinates:
 *    int k, c, m, n.
 * RETURNS: (double) filter value.
 */
__inline double GetH( VEC H, int k, int c, int m, int n )
{
  return H.X[k * H.L * H.H * H.W + c * H.H * H.W + m * H.W + n];
} /* End of 'GetH' function */

/* Get filter value by coordinates function.
 * ARGUMENTS:
 *  - pointer to filter:
 *    VEC *H;
 *  - value:
 *    double Val;
 *  - coordinates:
 *    int k, c, m, n.
 * RETURNS: none.
 */
__inline void PutH( VEC *H, double Val, int k, int c, int m, int n )
{
  H->X[k * c * H->H * H->W + c * H->H * H->W + m * H->W + n] = Val;
} /* End of 'PutH' function */

/* Vector clearing function.
 * ARGUMENTS:
 *  - pointer to clearing vector:
 *    VEC *V;
 *  - clear value:
 *    int Num.
 * RETURNS: none.
 */
__inline void VecClear( VEC *V, int Num )
{
  int i;

  for (i = 0; i < V->W * V->H * V->Cnt; i++)
    V->X[i] = Num;
} /* End of 'VecClear' function */

/* Matrix clearing function.
 * ARGUMENTS:
 *  - pointer to clearing matrix:
 *    MATR *M;
 *  - clear value:
 *    int Num.
 * RETURNS: none.
 */
__inline void MatrClear( MATR *M, int Num )
{
  int i, j;

  for (i = 0; i < M->H * M->Cnt; i++)
	for (j = 0; j < M->W; j++)
      M->X[i][j] = Num;
} /* End of 'MatrClear' function */

/* Parameters clearing function.
 * ARGUMENTS:
 *  - pointer to clearing parameters:
 *    PARAM *P;
 *  - clear value:
 *    int Num.
 * RETURNS: none.
 */
__inline void ParamClear( PARAM *P, int Num )
{
  int i;

  for (i = 0; i < P->W.H * P->W.W * P->W.Cnt * P->W.L; i++)
    P->W.X[i] = Num;
  for (i = 0; i < P->W.H * P->W.Cnt; i++)
    P->B.X[i] = Num;
} /* End of 'ParamClear' function */

/* Parameters setting function.
 * ARGUMENTS:
 *  - pointer to setting parameters:
 *    PARAM *P;
 *  - size of setting parameters:
 *    int W, H;
 *  - number of setting matrix:
 *    int Cnt;
 *  - layers number:
 *    int L;
 *  - convolution layer flag:
 *    int Conv.
 * RETURNS: none.
 */
__inline void ParamSet( PARAM *P1, int H, int W, int Cnt, int L, int Conv )
{
  P1->W.H = H;
  P1->W.W = W;
  P1->W.L = L;
  P1->W.Cnt = Cnt;
  P1->B.H = H;
  if (Conv)
    P1->B.H = 1;
  P1->B.W = 1;
  P1->B.Cnt = Cnt;
  P1->B.L = 1;
  P1->B.X = MALLOC(H * Cnt);
  P1->W.X = MALLOC(H * W * Cnt * L);
} /* End of 'ParamSet' function */

/* Parameters setting function.
 * ARGUMENTS:
 *  - pointer to setting parameter:
 *    PARAM *P1;
 *  - add parameter
 *    PARAM P2;
 *  - multiplication parameter
 *    double alpha.
 * RETURNS: none.
 */
__inline void ParamAdd( PARAM *P1, PARAM P2, double alpha )
{
  int i;

  for (i = 0; i < P1->W.H * P1->W.W * P1->W.Cnt * P1->W.L; i++)
    P1->W.X[i] += alpha * P2.W.X[i];
  for (i = 0; i < P1->B.H * P1->B.Cnt; i++)
    P1->B.X[i] += alpha * P2.B.X[i];
} /* End of 'ParamAdd' function */

/* Random number obtain in range [-1.0 .. 1.0] function.
 * ARGUMENTS: none.
 * RETURNS:
 *  (double) random number.
 */
__inline double R1( void )
{
  return 2.0 * rand() / RAND_MAX - 1;
} /* End of 'R1' function */

/* Random from 0 to 1 function.
 * ARGUMENTS: none.
 * RETURNS: (double) random number from 0 to 1.
 */
__inline double R0( void )
{
  return (double)rand() / RAND_MAX;
} /* End of 'R0' function */

/* Vector multiplication on matrix.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *Y;
 *  - vector:
 *    VEC V;
 *  - matrix:
 *    VEC M.
 * RETURNS: none.
 */
__inline void VecMulMatr( VEC *Y, VEC V, VEC M )
{
  int i, j;

  VecClear(Y, 0);

  for (i = 0; i < M.H; i++)
    for (j = 0; j < M.W; j++)
      Y->X[i] += GetMatr(&M, j, i, 0) * V.X[j];
} /* End of 'VecMulMatr' function */

/* Vector multiplication on vector (external).
 * ARGUMENTS:
 *  - pointer to changing matrix:
 *    VEC *W;
 *  - vectors:
 *    VEC V, M.
 * RETURNS: none.
 */
__inline void VecMulVecEx( VEC *W, VEC V, VEC M )
{
  int i, j;

  for (i = 0; i < W->H; i++)
    for (j = 0; j < W->W; j++)
      PutMatr(W, V.X[j] * M.X[i], j, i, 0);
} /* End of 'VecMulVecEx' function */

/* Vector add vector.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *V;
 *  - vectors:
 *    VEC V1, V2.
 * RETURNS: none.
 */
__inline void VecAddVec( VEC *V, VEC V1, VEC V2 )
{
  int i;

  for (i = 0; i < V1.H * V1.W; i++)
    V->X[i] = V1.X[i] + V2.X[i];
} /* End of 'VecAddVec' function */

/* Scalar product of vectors function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *V;
 *  - vectors:
 *    VEC V1, V2.
 * RETURNS: none.
 */
__inline void VecDotVec( VEC *V, VEC V1, VEC V2 )
{
  int i;

  for (i = 0; i < V1.H * V1.W; i++)
    V->X[i] += V1.X[i] * V2.X[i];
} /* End of 'VecDotVec' function */

/* Matrix transpose function.
 * ARGUMENTS:
 *  - pointer to changing matrix:
 *    VEC *W;
 *   - matrix
 *       VEC M. 
 * RETURNS: none.
 */
__inline void MatrTranspose( VEC *W, VEC M )
{
  int i, j;

  for (i = 0; i < M.W; i++)
    for (j = 0; j < M.H; j++)
      PutMatr(W, GetMatr(&M, i, j, 0), j, i, 0);
} /* End of 'MatrTranspose' function */

/* Absolute value of a number function
 * ARGUMENTS:
 *  - number:
 *    double X.
 * RETURNS: (double) resulting number.
 */
__inline double ABS( double X )
{
  if (X > 0)
    return X;
  return -X;
} /* End of 'ABS' function */

/* Loss function.
 * ARGUMENTS:
 *  - right vector:
 *    VEC Y1;
 *  - real vector:
 *    VEC Y.
 * RETURNS: (double) result of loss function.
 */
__inline double Loss( VEC Y1, VEC Y )
{
  int i;
  double sum = 0;
  for (i = 0; i < 10; i++)
    sum += Y1.X[i] * log(Y.X[i]);

  return -sum;
} /* End of 'Loss' function */

/* Vector multiple by number function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *V;
 *   - vector 
 *       VEC V1;
 *   - number
 *        double N. 
 * RETURNS: none.
 */
__inline void VecMulNum( VEC *V, VEC V1, double N )
{
  int i;

  for (i = 0; i < V1.H * V1.W; i++)
    V->X[i] = V1.X[i] * N;
} /* End of 'VecMulNum' function */

#endif /* __UTILS_H_ */
/* END OF 'UTILS.H' FILE */
