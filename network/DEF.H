/* Vasilisa Dmitrieva, 10-6, 14.06.2019 */

#ifndef __DEF_H_
#define __DEF_H_

#ifndef NDEBUG
#  define _CRTDBG_MAP_ALLOC
#  include <crtdbg.h>
#  define SetDbgMemHooks() \
     _CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF | \
       _CRTDBG_ALLOC_MEM_DF | _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG))
#else /* _DEBUG */
#  define SetDbgMemHooks() ((void)0)
#endif /* _DEBUG */
#include <stdlib.h>

#include "UTILS.H"

#endif /* __DEF_H_ */
/* END OF 'DEF.H' FILE */
