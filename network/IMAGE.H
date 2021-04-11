/* Vasilisa Dmitrieva, 10-6, 03.06.2019 */

#ifndef __IMAGE_H_
#define __IMAGE_H_

#include "DEF.H"
/* Max image size in pixels */
#define MAX_IMAGE_H 28
#define MAX_IMAGE_W 28

/* Max image number */
#define MAX_IMAGE_NUM 500

/* Pixel representation structure */
typedef struct tagPIXEL
{
  int X, Y;           /* Image coords */
  int r, g, b;        /* Color components */
  int Clust;          /* Cluster number */
} PIXEL;

/* Image representation structure */
typedef struct tagIMAGE
{
  int W, H;                              /* Image size */
  unsigned char Pixels[MAX_IMAGE_W][MAX_IMAGE_H]; /* Image pixels */
  unsigned char Type;                             /* Number */
} IMAGE;

/* Parametrs for neural network */
extern PARAM P[MAX_FC_LAY];

/* Parametrs for neural network */
extern PARAM P8[MAX_PARAMS];

/* Size parametrs for neural network */
extern int ParamH[MAX_FC_LAY], ParamW[MAX_FC_LAY];

/* Size parametrs for neural network */
extern int ParamH8[MAX_PARAMS], ParamW8[MAX_PARAMS], ParamC8[MAX_PARAMS], ParamL8[MAX_PARAMS];
extern int ParamConv8[MAX_PARAMS];

/* Initialization of parametrs function
 * ARGUMENTS:
 *  - parametrs:
 *    PARAM *P.
 * RETURNS: none.
 */
void ParamInit( PARAM *P );


/* Processing of image function
 * ARGUMENTS:
 *  - image:
 *    IMAGE Im;
 *  - parametrs:
 *    PARAM *P.
 * RETURNS: none.
 */
void FindNum8L( IMAGE Im, PARAM *P );

/* Get parametrs from file function.
 * ARGUMENTS:
 *  - parameters:
 *    PARAM *P.
 * RETURNS: none.
 */
void GetParams(void);


/* Test processing of images function.
 * ARGUMENTS:
 *  - number of processing images:
 *    int NumOfImage1;
 *  - number of testing images:
 *    int NumOfImage2.
 * RETURNS: none.
 */
void TakeImage8( int NumOfImage1, int NumOfImage2 );

#endif /* __IMAGE_H_ */
/* END OF 'IMAGE.H' FILE */
