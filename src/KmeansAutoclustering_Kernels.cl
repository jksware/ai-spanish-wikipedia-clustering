/*
 *  Wikipedia Clustering
 *  Copyright (C) 2015 Juan Carlos Pujol Mainegra, Damian Valdés Santiago
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.  
 */

/**********************************************************************
* 
*   File:                   KmeansAutoclustering_Kernel.cl
*   Original author:        Advanced Micro Devices, Inc 
*                           (original copyright notice below)
*
*   Modifier:               Juan Carlos Pujol Mainegra
*                           <j.pujol@lab.matcom.uh.cu>
*
*   Change log:             Added multiple components' vector
*   Date:                   January 5th, 2015
* 
**********************************************************************/

/**********************************************************************
*  Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved.
*  
*  Redistribution and use in source and Posary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*  
*  .  Redistributions of source code must retain the above copyright notice, this
*  list of conditions and the following disclaimer.
*  
*  .  Redistributions in Posary form must reproduce the above copyright notice,
*  this list of conditions and the following disclaimer in the documentation
*  and/or other materials provided with the distribution.
*  
*  
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
*  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
*  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
*  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************/

void atomicAddGlobal(volatile __global float *ptr, float value)
{
    unsigned int oldIntVal, newIntVal;
    float newFltVal;

    do
    {
        oldIntVal = *((volatile __global unsigned int *)ptr);
        newFltVal = ((*(float*)(&oldIntVal)) + value);
        newIntVal = *((unsigned int *)(&newFltVal));
    }
    while (atomic_cmpxchg((volatile __global unsigned int *)ptr, oldIntVal, newIntVal) != oldIntVal);
}

void atomicAddLocal(volatile __local float *ptr, float value)
{
    unsigned int oldIntVal, newIntVal;
    float newFltVal;

    do
    {
        oldIntVal = *((volatile __local unsigned int *)ptr);
        newFltVal = ((*(float*)(&oldIntVal)) + value);
        newIntVal = *((unsigned int *)(&newFltVal));
    }
    while (atomic_cmpxchg((volatile __local unsigned int *)ptr, oldIntVal, newIntVal) != oldIntVal);
}

#define MAX_CLUSTERS 32
#define MAX_DIMENSION 1024

__kernel
void assignCentroid(
    __global float *pointPos,
    __global uint *KMeansCluster,
    __global float *centroidPos,
    __global float *globalClusterPos,          // size k, newCentroidPos
    __global unsigned int *globalClusterCount,
    __local float *localClusterPos,            // size k
    __local unsigned int *localClusterCount,
    uint k,
    uint numPoints,
    uint dims)
{
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    if (lid < k)
    {
        for (uint i = 0; i < dims; i++)
            localClusterPos[lid * dims + i] = 0.0;
        localClusterCount[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // // Load 1 point
    // __constant float vPoint[MAX_DIMENSION];
    // for (int i = 0; i < dims; i++)
    //     vPoint[i] = pointPos[gid * dims + i];

    float leastDist = FLT_MAX;
    uint closestCentroid = 0;
    
    for (int i = 0; i < k; i++)
    {
        float dist = 0.0;
        for (int j = 0; j < dims; j++)
        {
            float xDist = (pointPos[gid * dims + j] - centroidPos[i * dims + j]);
            dist += xDist * xDist;
        }

        leastDist = fmin( leastDist, dist );

        closestCentroid = (leastDist == dist) ? i : closestCentroid;
    }
    
    KMeansCluster[gid] = closestCentroid;

    for (int i = 0; i < dims; i++)
        atomicAddLocal( &localClusterPos[dims * closestCentroid + i], pointPos[gid * dims + i] );
    atomic_inc( &localClusterCount[closestCentroid] );
    barrier(CLK_LOCAL_MEM_FENCE);

    // Push back the local Pos and count values to global
    if(lid < k)
    {
        for (int i = 0; i < dims; i++)
            atomicAddGlobal( (__global float*) (globalClusterPos + dims * lid + i ), localClusterPos[lid * dims + i] );
            // atomicAddGlobal( (__global float*) (globalClusterPos + dims * lid + i ), pointPos[gid * dims + i] );

        atomic_add( &globalClusterCount[lid], localClusterCount[lid] );
        // atomic_inc( &globalClusterCount[lid] );
    }
}



__kernel void computeSilhouettes(__global float2* pointPos,
                                __global float2* centroidPos, 
                                __global unsigned int* KmeansCluster, 
                                __global unsigned int* globalClusterCount, 
                                __local int* lClusterCount, //reduce global access
                                int k, 
                                int numPoints, 
                                __local float* lSilhouetteValue, 
                                __global float* gSilhoutteValue)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    if(lid == 0)
    {
        lSilhouetteValue[0] = 0.f;
    }
    if(lid < k)
    {
        lClusterCount[lid] = globalClusterCount[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float silhScore = 0.f;
    float dissimilarities[MAX_CLUSTERS] = {0.0f};
    
    for(int i=0; i<numPoints; i++)
    {
        dissimilarities[KmeansCluster[i]] += (sqrt(pow(pointPos[i].s0 - pointPos[gid].s0, 2.0f)
                                             + pow(pointPos[i].s1 - pointPos[gid].s1, 2.0f)));
    }
    
    float a = dissimilarities[KmeansCluster[gid]] / lClusterCount[KmeansCluster[gid]];
    float b = FLT_MAX;
    for(int i=0; i<k; i++)
    {
        if(i != KmeansCluster[gid])
            b =  min(b, dissimilarities[i] / lClusterCount[i]);
    }
    
    silhScore = ((b - a) / max(a, b));
    
    atomicAddLocal(lSilhouetteValue, silhScore);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(lid == 0)
    {
        atomicAddGlobal(gSilhoutteValue, lSilhouetteValue[0]);
    }
}
