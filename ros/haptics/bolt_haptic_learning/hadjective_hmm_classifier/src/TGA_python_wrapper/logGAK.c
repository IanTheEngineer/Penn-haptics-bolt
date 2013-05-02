/*
 ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1/GPL 2.0/LGPL 2.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is Global Alignment Kernel, (C) 2010, Marco Cuturi
 *
 * The Initial Developers of the Original Code is
 *
 * Marco Cuturi   mcuturi@i.kyoto-u.ac.jp
 *
 * Portions created by the Initial Developers are
 * Copyright (C) 2011 the Initial Developers. All Rights Reserved.
 *
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU General Public License Version 2 or later (the "GPL"), or
 * the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
 * in which case the provisions of the GPL or the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of either the GPL or the LGPL, and not to allow others to
 * use your version of this file under the terms of the MPL, indicate your
 * decision by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL or the LGPL. If you do not delete
 * the provisions above, a recipient may use your version of this file under
 * the terms of any one of the MPL, the GPL or the LGPL.
 *
 ***** END LICENSE BLOCK *****
 *
 * REVISIONS:
 * This is v1.02 of Global Alignment Kernel, June 8th 2011.
 * Changed some C syntax that was not compiled properly on Windows platforms
 * 
 * Previous versions:
 * v1.02 (Adrien Gaidon): removed Mex part + minor changes for Python wrapper
 * v1.01 of Global Alignment Kernel, May 12th 2011 (updated comments fields)
 * v1.0 of Global Alignment Kernel, March 25th 2011.
 * 

 */


#include <stdlib.h>
#include <math.h>
/* Useful constants */
#define LOG0 -10000          /* log(0) */
#define LOGP(x, y) (((x)>(y))?(x)+log1p(exp((y)-(x))):(y)+log1p(exp((x)-(y))))

/* Implementation of the (Triangular) global alignment kernel.
 *
 * seq1 is a first sequence represented as a matrix of real elements. Each line i corresponds to the vector of observations at time i.
 * seq2 is the second sequence formatted in the same way.
 * nX, nY and dimvect provide the number of lines of seq1 and seq2.
 * sigma stands for the bandwidth of the \phi_\sigma distance used kernel
 * lambda is an additional factor that can be used with the Geometrically divisible Gaussian Kernel
 * triangular is a parameter which parameterizes the triangular kernel
 * kerneltype selects either the Gaussian Kernel or its geometrically divisible equivalent */
double logGAK(double *seq1 , double *seq2, int nX, int nY, int dimvect, double sigma, int triangular)
{
    int i, j, ii, cur, old, curpos, frompos1, frompos2, frompos3;    
    double aux;
    int cl = nY+1;                /* length of a column for the dynamic programming */
    
    
    double sum=0;
    double gram, Sig;    
    /* logM is the array that will stores two successive columns of the (nX+1) x (nY+1) table used to compute the final kernel value*/
    double * logM = malloc(2*cl * sizeof(double));        
    
    int trimax = (nX>nY) ? nX-1 : nY-1; /* Maximum of abs(i-j) when 1<=i<=nX and 1<=j<=nY */
    
    double *logTriangularCoefficients = malloc((trimax+1) * sizeof(double)); 
    if (triangular>0) {
        /* initialize */
        for (i=0;i<=trimax;i++){
            logTriangularCoefficients[i]=LOG0; /* Set all to zero */
        }
        
        for (i=0;i<((trimax<triangular) ? trimax+1 : triangular);i++) {
            logTriangularCoefficients[i]=log(1-i/triangular);
        }
    }
    else
        for (i=0;i<=trimax;i++){
        logTriangularCoefficients[i]=0; /* 1 for all if triangular==0, that is a log value of 0 */
        }
    Sig=-1/(2*sigma*sigma);
    
    
    
    /****************************************************/
    /* First iteration : initialization of columns to 0 */
    /****************************************************/
    /* The left most column is all zeros... */
    for (j=1;j<cl;j++) {
        logM[j]=LOG0;
    }
    /* ... except for the lower-left cell which is initialized with a value of 1, i.e. a log value of 0. */
    logM[0]=0;
    
    /* Cur and Old keep track of which column is the current one and which one is the already computed one.*/
    cur = 1;      /* Indexes [0..cl-1] are used to process the next column */
    old = 0;      /* Indexes [cl..2*cl-1] were used for column 0 */
    
    /************************************************/
    /* Next iterations : processing columns 1 .. nX */
    /************************************************/
    
    /* Main loop to vary the position for i=1..nX */
    curpos = 0;
    for (i=1;i<=nX;i++) {
        /* Special update for positions (i=1..nX,j=0) */
        curpos = cur*cl;                  /* index of the state (i,0) */
        logM[curpos] = LOG0;
        /* Secondary loop to vary the position for j=1..nY */
        for (j=1;j<=nY;j++) {
            curpos = cur*cl + j;            /* index of the state (i,j) */
            if (logTriangularCoefficients[abs(i-j)]>LOG0) {
                frompos1 = old*cl + j;            /* index of the state (i-1,j) */
                frompos2 = cur*cl + j-1;          /* index of the state (i,j-1) */
                frompos3 = old*cl + j-1;          /* index of the state (i-1,j-1) */
                
                /* We first compute the kernel value */
                sum=0;
                for (ii=0;ii<dimvect;ii++) {
                    sum+=(seq1[i-1+ii*nX]-seq2[j-1+ii*nY])*(seq1[i-1+ii*nX]-seq2[j-1+ii*nY]);
                }
                gram= logTriangularCoefficients[abs(i-j)] + sum*Sig ;
                gram -=log(2-exp(gram));
                
                /* Doing the updates now, in two steps. */
                aux= LOGP(logM[frompos1], logM[frompos2] );
                logM[curpos] = LOGP( aux , logM[frompos3] ) + gram;
            }
            else {
                logM[curpos]=LOG0;
            }
        }
        /* Update the culumn order */
        cur = 1-cur;
        old = 1-old;
    }
    aux = logM[curpos];
    free(logM);
    free(logTriangularCoefficients);
    /* Return the logarithm of the Global Alignment Kernel */    
    return aux;
    
}

