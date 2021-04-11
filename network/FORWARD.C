/* Vasilisa Dmitrieva, 10-6, 10.06.2019 */

#include "LAYERS.H"

/* Forward SoftMax function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *C;
 *  - vector needed to be softmaxed:
 *    VEC X.
 * RETURNS: none.
 */
void SoftMaxF( VEC *C, VEC X )
{
  int i;
  double sum = 0, max = X.X[0];

  for (i = 1; i < 10; i++)
    if (X.X[i] > max)
      max = X.X[i];

  for (i = 0; i < X.H * X.W; i++)
    X.X[i] -= max, sum += pow(E, X.X[i]);
  for (i = 0; i < X.H * X.W; i++)
    C->X[i] = pow(E, X.X[i]) / sum;
} /* End of 'SoftMaxF' function */

/* Forward fully connected function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *Y;
 *  - parametres:
 *    PARAM P1;
 *  - argument:
 *    VEC X.
 * RETURNS: none.
 */ 
void FullyConnectedF( VEC *Y, VEC X, PARAM P1 )
{
  int i;

  VecMulMatr(Y, X, P1.W);

  for (i = 0; i < Y->W * Y->H; i++)
    Y->X[i] += P1.B.X[i];
} /* End of 'FullyConnectedF' function */

/* Forward ReLu function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *C;
 *  - argument:
 *    VEC X.
 * RETURNS: none.
 */
void ReLuF( VEC *C, VEC X )
{
  int i;

  for (i = 0; i < X.W * X.H; i++)
    if (X.X[i] < 0)
      C->X[i] = 0;
    else
      C->X[i] = X.X[i];
} /* End of 'ReLuF' function */

/* Forward Convolution function.
 * ARGUMENTS:
 *  - pointer to changing matrix:
 *    MATR *W;
 *  - argument:
 *    VEC X;
 *  - filters parameters:
 *    PARAM P1.
 * RETURNS: none.
 */
void ConvolutionF( VEC *Y1, VEC Y, PARAM P1 )
{
  int i, j, k, m, c, n;

  VecClear(Y1, 0);

  for (k = 0; k < P1.W.Cnt; k++)
    for (i = 0; i < Y1->H; i++)
      for (j = 0; j < Y1->W; j++)
        for (c = 0; c < Y.Cnt; c++)
          for (m = 0; m < P1.W.H; m++)
            for (n = 0; n < P1.W.W; n++)
              Y1->X[k * Y1->H * Y1->W + i * Y1->W + j] += GetH(P1.W, k, c, m, n) * GetMatr(&Y, j + n, i + m, c);
  for (k = 0; k < P1.W.Cnt; k++)
    for (i = 0; i < Y1->H; i++)
      for (j = 0; j < Y1->W; j++)
	    Y1->X[k * Y1->H * Y1->W + i * Y1->W + j] += P1.B.X[k];
} /* End of 'ConvolutionF' function */

/* Forward MaxPool function.
 * ARGUMENTS:
 *  - pointer to changing matrix:
 *    VEC *W;
 *  - argument:
 *    MATR X;
 *  - size:
 *    int S.
 * RETURNS: none.
 */
void MaxPoolF( VEC *W, VEC X, int S )
{
  int i, j, m, k, n;
  double max;

  for (k = 0; k < X.Cnt; k++)
    for (i = 0; i < X.H; i += S)
      for (j = 0; j < X.W; j += S)
      {
        max = GetMatr(&X, i, j, k);
        for (m = 0; m < S; m++)
          for (n = 0 ; n < S; n++)
            if (GetMatr(&X, i + m, j + n, k) > max)
              max = GetMatr(&X, i + m, j + n, k);
        PutMatr(W, max, i / S, j / S, k);
      }
} /* End of 'MaxPoolF' function */

/* END OF 'FORWARD.C' FILE */
