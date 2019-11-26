/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGaussianNoise.h"

#include <random>

///////////////////////////////////////////////////////////////////////////////
LayerGaussianNoise::LayerGaussianNoise(int iSize,float fStd):
    Layer(iSize,iSize,"GaussianNoise"),
    _fStd(fStd),
	_distNormal(0.f, fStd)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerGaussianNoise::~LayerGaussianNoise()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGaussianNoise::clone() const
{
    return new LayerGaussianNoise(_iInSize,_fStd);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if (_bTrainMode && (_fStd > 0.) )
	{
		mOut.resize(mIn.rows(), mIn.cols());

		for (int i = 0; i < mOut.size(); i++)
			mOut(i) = mIn(i) + _distNormal(_RNGgenerator);
	}
	else
		mOut = mIn; // in test mode or sigma==0.
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;
 
	if (_bFirstLayer)
		return;

	mGradientIn= mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
float LayerGaussianNoise::get_std() const
{
    return _fStd;
}
///////////////////////////////////////////////////////////////////////////////
