/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__strdup_char_52b.c
Label Definition File: CWE401_Memory_Leak__strdup.label.xml
Template File: sources-sinks-52b.tmpl.c
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource:  Allocate data using strdup()
 * GoodSource: Allocate data on the stack
 * Sinks:
 *    GoodSink: call free() on data
 *    BadSink : no deallocation of data
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE401_Memory_Leak__strdup_char_52c_badSink(char * data);

void CWE401_Memory_Leak__strdup_char_52b_badSink(char * data)
{
    CWE401_Memory_Leak__strdup_char_52c_badSink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE401_Memory_Leak__strdup_char_52c_goodG2BSink(char * data);

void CWE401_Memory_Leak__strdup_char_52b_goodG2BSink(char * data)
{
    CWE401_Memory_Leak__strdup_char_52c_goodG2BSink(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE401_Memory_Leak__strdup_char_52c_goodB2GSink(char * data);

void CWE401_Memory_Leak__strdup_char_52b_goodB2GSink(char * data)
{
    CWE401_Memory_Leak__strdup_char_52c_goodB2GSink(data);
}

#endif /* OMITGOOD */
