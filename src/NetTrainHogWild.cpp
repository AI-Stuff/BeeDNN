/*
    Copyright (c) 2020, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "NetTrainHogWild.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"

#include <cmath>
#include <cassert>

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainHogWild::NetTrainHogWild(): NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainHogWild::~NetTrainHogWild()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainHogWild::train_one_epoch(Index iBatchSize, const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled)
{
	Index iNbSamples = mSampleShuffled.rows();
	Index iBatchStart = 0;

	while (iBatchStart < iNbSamples)
	{
		Index iBatchEnd = iBatchStart + iBatchSize;
		if (iBatchEnd > iNbSamples)
			iBatchEnd = iNbSamples;

		const MatrixFloat mSample = rowRange(mSampleShuffled, iBatchStart, iBatchEnd);
		const MatrixFloat mTarget = rowRange(mTruthShuffled, iBatchStart, iBatchEnd);

		train_batch(mSample, mTarget);

		iBatchStart = iBatchEnd;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////
