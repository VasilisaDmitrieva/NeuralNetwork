/* Vasilisa Dmitrieva, 10-6, 10.06.2019 */

#include "LAYERS.H"

/* Backward fully connected function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *V;
 *  - matrix W:
 *    VEC M;
 *  - argument:
 *    VEC X.
 * RETURNS: none.
 */
void FullyConnectedB( VEC *V, VEC M, VEC X )
{
  VecMulMatr(V, X, M);
} /* End of 'FullyConnectedB' function */

/* Backward ReLu function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *V;
 *  - parameter:
 *    VEC Y;
 *  - argument:
 *    VEC X.
 * RETURNS: none.
 */
void ReLuB( VEC *V, VEC X, VEC Y )
{
  int i;

  for (i = 0; i < X.H * X.W; i++)
    V->X[i] = (Y.X[i] >= 0) * X.X[i];
} /* End of 'ReLuB' function */

/* Backward SoftMax function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *V;
 *  - real vector:
 *    VEC Y1;
 *  - probable value vector:
 *    VEC Y.
 * RETURNS: none.
 */
void SoftMaxB( VEC *V, VEC Y1, VEC Y )
{
  int i;

  for (i = 0; i < Y1.H * Y1.W; i++)
    V->X[i] = Y.X[i] - Y1.X[i];
} /* End of 'SoftMaxB' function */

/* Backward MaxPool function.
 * ARGUMENTS:
 *  - pointer to changing vector:
 *    VEC *V;
 *  - previous vector:
 *    MATR Y1;
 *  - vector:
 *    MATR Y;
 *  - derivative matrix:
 *    MATR G;
 *  - maxpool size:
 *    int S.
 * RETURNS: none.
 */
void MaxPoolB( VEC *V, VEC Y, VEC Y1, VEC G, int S )
{
  int i, j, c;

  for (c = 0; c < Y.Cnt; c++)
    for (i = 0; i < Y.H; i++)
      for (j = 0; j < Y.W; j++)
        V->X[c * Y.H * Y.W + i * Y.W + j] = (GetMatr(&Y1, j / S, i / S, c) == GetMatr(&Y, j, i, c)) * GetMatr(&G, j / S, i / S, c);
} /* End of 'MaxPoolB' function */

/* Gradient Convolution function.
 * ARGUMENTS:
 *  - pointer to changing matrix:
 *    MATR *W;
 *  - argument:
 *    VEC Y;
 *  - derivative matrix:
 *    VEC G.
 * RETURNS: none.
 */
void ConvolutionG( PARAM *P1, VEC Y, VEC G )
{
  int i, j, k, m, c, n;

  ParamClear(P1, 0);
  
  for (k = 0; k < P1->W.Cnt; k++)
    for (m = 0; m < P1->W.H; m++)
      for (n = 0; n < P1->W.W; n++)
        for (c = 0; c < Y.Cnt; c++)
          for (i = 0; i < G.H; i++)
            for (j = 0; j < G.W; j++)
              P1->W.X[k * Y.Cnt * P1->W.H * P1->W.W + c * P1->W.H * P1->W.W + m * P1->W.W + n] += GetMatr(&G, j, i, k) * GetMatr(&Y, j + n, i + m, c);

  for (k = 0; k < P1->W.Cnt; k++)
    for (i = 0; i < G.H; i++)
      for (j = 0; j < G.W; j++)
        P1->B.X[k] += GetMatr(&G, j, i, k);

} /* End of 'ConvolutionG' function */

/* Forward Convolution function.
 * ARGUMENTS:
 *  - pointer to changing matrix:
 *    VEC *G1;
 *  - derivative matrix:
 *    VEC G;
 *  - filters parameters:
 *    VEC H.
 * RETURNS: none.
 */
void ConvolutionB( VEC *G1, VEC G, VEC H )
{
  int i, j, k, m, c, n;

  VecClear(G1, 0);

  for (c = 0; c < H.L; c++)
    for (i = H.H - 1; i < G1->H - H.H + 1; i++)
      for (j = H.W - 1; j < G1->W - H.W + 1; j++)
        for (k = 0; k < H.Cnt; k++)
          for (m = 0; m < H.H; m++)
            for (n = 0; n < H.W; n++)
              G1->X[c * G1->H * G1->W + i * G1->W + j] += GetH(H, k, c, m, n) * GetMatr(&G, j - n, i - m, k);
} /* End of 'ConvolutionB' function */

/* END OF 'BACKWARD.C' FILE */
