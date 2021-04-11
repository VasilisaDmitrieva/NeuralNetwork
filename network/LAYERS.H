/* Vasilisa Dmitrieva, 10-6, 10.06.2019 */

#ifndef __LAYERS_H_
#define __LAYERS_H_

#include "DEF.H"

/* Forward SoftMax function.
* ARGUMENTS:
*  - vector needed to be softmaxed:
*    VEC C.
* RETURNS: (VEC) softmaxed vector.
*/
void SoftMaxF( VEC *C, VEC X );

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
void FullyConnectedF( VEC *Y, VEC X, PARAM P1 );

/* Forward ReLu function.
* ARGUMENTS:
*  - argument:
*    VEC X.
* RETURNS: (VEC) changed vector.
*/
void ReLuF( VEC *C, VEC X );

/* Backward SoftMax function.
 * ARGUMENTS:
 *  - real vector:
 *    VEC Y1;
 *  - probable value vector:
 *    VEC Y;
 * RETURNS: (VEC) changed vector.
 */
void SoftMaxB( VEC *V, VEC Y1, VEC Y );

/* Backward ReLu function.
 * ARGUMENTS:
 *  - parameter:
 *    VEC Y;
 *  - argument:
 *    VEC X.
 * RETURNS: (VEC) changed vector.
 */
void ReLuB( VEC *V, VEC X, VEC Y );

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
void FullyConnectedB( VEC *V, VEC M, VEC X );

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
void ConvolutionF( VEC *Y1, VEC Y, PARAM P1 );

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
void MaxPoolF( VEC *W, VEC X, int S );

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
void MaxPoolB( VEC *V, VEC Y, VEC Y1, VEC G, int S );

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
void ConvolutionG( PARAM *P1, VEC Y, VEC G );

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
void ConvolutionB( VEC *G1, VEC G, VEC H );

#endif /* __LAYERS_H_ */
/* END OF 'LAYERS.C' FILE */
