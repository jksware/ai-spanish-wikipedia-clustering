#
#  Wikipedia Clustering
#  Copyright (C) 2015 Juan Carlos Pujol Mainegra, Damian Valdés Santiago
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.  
#

from __future__ import print_function
__author__ = 'Juanca'

# CLUSTER_ID1 = 69
# CLUSTER_ID2 = 70


import pyopencl as cl
import numpy as np
from numpy.linalg import norm
from random import gauss
import json

CTX = cl.create_some_context(answers=[1, ])
DEFAULT_PROGRAM_SRC_PATH = r"KmeansAutoclustering_Kernels.cl"
with open(DEFAULT_PROGRAM_SRC_PATH, mode='r') as OPENCL_DEFAULT_PROGRAM_FILE:
    DEFAULT_PROGRAM_SRC = OPENCL_DEFAULT_PROGRAM_FILE.read()

DEFAULT_CLUSTER_PRG = cl.Program(CTX, DEFAULT_PROGRAM_SRC)
try:
    DEFAULT_CLUSTER_PRG.build()
except:
    print("Error:")
    print(DEFAULT_CLUSTER_PRG.get_build_info(CTX.devices[0], cl.program_build_info.LOG))
    raise


class KMeansOpenCL(object):
    def __init__(self, n_clusters=10, max_iters=300, tol=1e-2, program_src=None, verbose=True):
        self.ctx = CTX
        self.queue = cl.CommandQueue(self.ctx)

        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose

        # if isinstance(program_src, file):
        #     sort_program_source = program_src.read()

        if program_src != DEFAULT_PROGRAM_SRC_PATH and program_src is not None:
            with open(program_src, mode='r') as opencl_program_file:
                program_source = opencl_program_file.read()

            self.program_src = program_src
            self.cluster_prg = cl.Program(self.ctx, program_source)
            try:
                self.cluster_prg.build()
            except:
                print("Error:")
                print(self.cluster_prg.get_build_info(self.ctx.devices[0], cl.program_build_info.LOG))
                raise
        else:
            self.program_src = DEFAULT_PROGRAM_SRC
            self.cluster_prg = DEFAULT_CLUSTER_PRG

        self.dims = None
        self.point_pos = None
        self.pointPosBuf = None
        self.labels_ = None
        self.labels_Buf = None
        self.centroidPosBuf = None
        self.cluster_centers_ = None
        self.cluster_centersBuf = None
        self.globalClusterCount = None
        self.globalClusterCountBuf = None
        self.localClusterPosBuf = None
        self.localClusterCountBuf = None
        self.numPoints = None
        self.n_clusters = n_clusters

        self.variance = None
        self.mean = None

    def fit(self, point_pos):
        assert(len(point_pos) > 0)

        self.numPoints = len(point_pos)
        self.dims = len(point_pos[0])

        mf = cl.mem_flags
        self.cluster_centers_ = np.empty((self.n_clusters, self.dims)).astype(np.float32)
        self.cluster_centersBuf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.cluster_centers_.nbytes)
        self.globalClusterCount = np.empty(self.n_clusters).astype(np.uint32)
        self.globalClusterCountBuf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.globalClusterCount.nbytes)

        self.localClusterPosBuf = cl.LocalMemory(self.cluster_centers_.nbytes)
        self.localClusterCountBuf = cl.LocalMemory(self.globalClusterCount.nbytes)

        self.point_pos = np.array(point_pos).astype(np.float32)
        self.labels_ = np.zeros(self.numPoints).astype(np.uint32)

        self.variance = np.var(self.point_pos)
        matrix = np.matrix(self.point_pos)
        self.mean = np.array(matrix.mean(0)).flatten()

        mf = cl.mem_flags
        self.pointPosBuf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=point_pos)
        self.labels_Buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.labels_.nbytes)

    def compute(self, centroid_pos):
        assert self.point_pos is not None
        assert len(centroid_pos) == self.n_clusters

        global_size = (self.numPoints, )
        # work_group_size = (2, )
        work_group_size = None

        mf = cl.mem_flags

        self.globalClusterCount = np.empty(self.n_clusters).astype(np.uint32)
        cl.enqueue_copy(self.queue, dest=self.globalClusterCountBuf, src=self.globalClusterCount)

        self.cluster_centers_ = np.empty((self.n_clusters, self.dims)).astype(np.float32)
        cl.enqueue_copy(self.queue, dest=self.cluster_centersBuf, src=self.cluster_centers_)

        centroid_pos = np.array(centroid_pos).astype(np.float32)
        self.centroidPosBuf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=centroid_pos)

        self.cluster_prg.assignCentroid(self.queue, global_size, work_group_size,
                                        self.pointPosBuf,
                                        self.labels_Buf,
                                        self.centroidPosBuf,
                                        self.cluster_centersBuf,
                                        self.globalClusterCountBuf,
                                        self.localClusterPosBuf,
                                        self.localClusterCountBuf,
                                        np.uint32(self.n_clusters),
                                        np.uint32(self.numPoints),
                                        # comment the line below this to get first float2 version
                                        np.uint32(self.dims)
                                        )

        cl.enqueue_copy(self.queue, dest=self.labels_, src=self.labels_Buf)
        cl.enqueue_copy(self.queue, dest=self.cluster_centers_, src=self.cluster_centersBuf)
        cl.enqueue_copy(self.queue, dest=self.globalClusterCount, src=self.globalClusterCountBuf)

        if self.verbose:
            print('Centroid pos')
            print(centroid_pos)
        for i in range(self.cluster_centers_.shape[0]):
            if self.verbose:
                print('Cluster', i, 'center', self.cluster_centers_[i])
                print('Cluster', i, 'count', self.globalClusterCount[i])
            if self.globalClusterCount[i] != 0:
                self.cluster_centers_[i] /= self.globalClusterCount[i]

        return self.labels_, self.cluster_centers_

    def kmeans(self, initial_clusters=None):
        if self.verbose:
            print('Dims', self.dims)
            print('n_clusters', self.n_clusters)

        if initial_clusters is None:
            # cluster_centers = (np.random.random((self.n_clusters, self.dims)) * 0.6 + 0.2).astype(np.float32)
            cluster_centers = [gauss(self.mean, self.variance) for _ in range(self.n_clusters)]
        else:
            cluster_centers = initial_clusters

        if self.verbose:
            print('Initial Clusters')
            print(cluster_centers)

        for i in range(self.max_iters):
            if self.verbose:
                print('*'*20)
                print("Iteration", i + 1)

            last_clusters_centers = cluster_centers
            cluster_assignment, cluster_centers = self.compute(cluster_centers)

            if self.verbose:
                print('Assignments')
                print(cluster_assignment)
                print('Clusters centers')
                print(self.cluster_centers_)

            # step_length = norm(last_clusters_centers - cluster_centers, ord=np.inf)
            step_length = float(norm(last_clusters_centers - cluster_centers, ord=2))

            if self.verbose:
                print('Step size', step_length)
                print('Cluster count')
                print(self.globalClusterCount)

            if step_length < self.tol:
                if self.verbose:
                    print("Stopping after {} iteration because of step length {}".format(i + 1, step_length))
                break


def test_cluster():
    n_clusters = 2

    # program_src = r'C:\Users\Juanca\AMD APP SDK\3.0-0-Beta\samples\opencl\bin\x86_64\KmeansAutoclustering_Kernels.cl'
    kmeans_opencl = KMeansOpenCL(n_clusters=n_clusters, max_iters=50, tol=1e-2, verbose=False)

    with open('tests\vector_space.json', 'r') as vector_space_file:
        vector_space = json.load(vector_space_file)
        vector_space = np.array(vector_space).astype(np.float32)

    with open('tests\document_topics.json', 'r') as document_topics_file:
        document_topics = json.load(document_topics_file)

    # simplify things
    # vdim = 3000
    # vlen = 1000
    # vector_space = np.array([v[:vdim] for v in vector_space])[:vlen]
    # print(vector_space)
    # print('vec len', vector_space.size)
    # return

    kmeans_opencl.fit(vector_space)

    # print('Vector space')
    # print(vector_space)

    # initial_clusters = np.array([[0.80056101, 0.66391277], [0.51066965, 0.45043868]]).astype(np.float32)
    initial_clusters = None
    kmeans_opencl.kmeans(initial_clusters=initial_clusters)

    # print(kmeans_opencl.cluster_centers_.size)
    # print(kmeans_opencl.cluster_centers_)
    print(sum(kmeans_opencl.labels_))

# test_cluster()
