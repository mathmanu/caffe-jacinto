/**
 * TI C++ Reference software for Computer Vision Algorithms (TICV)
 * TICV is a software module developed to model computer vision
 * algorithms on TI's various platforms/SOCs.
 *
 * Copyright (C) 2016 Texas Instruments Incorporated - http://www.ti.com/
 * ALL RIGHTS RESERVED
 */

/**
 * @file:       parallel.h
 * @brief:      Implements Parallel Thread Processing
 */
#pragma once

#include <functional>

namespace caffe {

typedef std::function<void(int)> ParalelForExecutorFunc;

class ParallelForFunctor: public cv::ParallelLoopBody {
public:
    ParallelForFunctor(ParalelForExecutorFunc func) :
        execFunc(func) {
    }
    void operator()(const cv::Range &range) const {
        for (int i = range.start; i < range.end; i++) {
            execFunc(i);
        }
    }
    ParalelForExecutorFunc execFunc;
};

static inline void ParallelFor(int start, int endPlus1, ParalelForExecutorFunc func, int nthreads = -1) {
    if (nthreads == 1) {
        for (int i = start; i < endPlus1; i++) {
            func(i);
        }
    } else {
        cv::Range range(start, endPlus1);
        ParallelForFunctor functor(func);
        cv::parallel_for_(range, functor, nthreads);
    }
}

}

