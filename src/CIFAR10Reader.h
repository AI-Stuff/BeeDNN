/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef CIFAR10Reader_
#define CIFAR10Reader_

#include "Matrix.h"

#include <string>
using namespace std;

class CIFAR10Reader
{
public:
    bool read_from_folder(const string& sFolder,MatrixFloat& mRefImages,MatrixFloat& mRefLabels,MatrixFloat& mTestImages,MatrixFloat& mTestLabels);

private:
    bool read_batch(string sName,MatrixFloat& mData,MatrixFloat& mTruth);
};

#endif
