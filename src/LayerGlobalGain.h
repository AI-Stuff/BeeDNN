/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerGlobalGain_
#define LayerGlobalGain_

#include "Layer.h"
#include "Matrix.h"

class LayerGlobalGain : public Layer
{
public:
    explicit LayerGlobalGain(int iInSize);
    virtual ~LayerGlobalGain();

    virtual Layer* clone() const override;

    virtual void init() override;
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const override;

    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    float gain() const;
};

#endif
