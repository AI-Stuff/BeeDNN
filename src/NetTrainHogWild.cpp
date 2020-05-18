/*
    Copyright (c) 2020, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "NetTrainHogwild.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"

#include <cmath>
#include <cassert>

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainHogwild::NetTrainHogwild(): NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainHogwild::~NetTrainHogwild()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainHogwild::train_one_epoch( const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled)
{
	Index iNbSamples = mSampleShuffled.rows();
	Index iBatchStart = 0;

	while (iBatchStart < iNbSamples)
	{
		Index iBatchEnd = iBatchStart + _iBatchSizeAdjusted;
		if (iBatchEnd > iNbSamples)
			iBatchEnd = iNbSamples;

		const MatrixFloat mSample = rowRange(mSampleShuffled, iBatchStart, iBatchEnd);
		const MatrixFloat mTarget = rowRange(mTruthShuffled, iBatchStart, iBatchEnd);

		train_batch(mSample, mTarget);

		iBatchStart = iBatchEnd;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////
