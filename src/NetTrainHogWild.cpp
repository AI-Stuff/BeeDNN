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
#include "Loss.h"
#include "Regularizer.h"
#include "Optimizer.h"

#include <cmath>
#include <cassert>
#include <omp.h>

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainHogwild::NetTrainHogwild(): NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainHogwild::~NetTrainHogwild()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainHogwild::train_one_epoch( const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled)
{
	Index iNbThread = omp_get_max_threads();
	Index iNbSamples = mSampleShuffled.rows();

	//compute nb batches taking into account the last smaller one
	Index iNbBatches = iNbSamples / _iBatchSizeAdjusted;
	if (iNbBatches*_iBatchSizeAdjusted < iNbSamples)
		iNbBatches++;

	Net netShare(*_pNet);
	NetTrain trainShare;
	trainShare = *this;

#pragma omp parallel for
	for(Index iBatch=0;iBatch<iNbBatches;iBatch++)
	{
		Net n(netShare);
		NetTrain nt = trainShare;
		nt.set_net(n);

		// compute mini batches range
		Index iBatchStart = iBatch* _iBatchSizeAdjusted;
		Index iBatchEnd = iBatchStart + _iBatchSizeAdjusted;
		if (iBatchEnd > iNbSamples)
			iBatchEnd = iNbSamples;

		const MatrixFloat mSample = rowView(mSampleShuffled, iBatchStart, iBatchEnd);
		const MatrixFloat mTarget = rowView(mTruthShuffled, iBatchStart, iBatchEnd);

		nt.train_batch(mSample, mTarget);

		netShare = n;

		iBatchStart = iBatchEnd;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////
/*void NetTrainHogwild::train_batch(const MatrixFloat& mSample, const MatrixFloat& mTruth)
{
	//forward pass with store
	_inOut[0] = mSample;
	for (size_t i = 0; i < _iNbLayers; i++)
		_pNet->layer(i).forward(_inOut[i], _inOut[i + 1]);

	//compute error gradient
	_pLoss->compute_gradient(_inOut[_iNbLayers], mTruth, _gradient[_iNbLayers]);

	//backward pass with optimizer
	for (int i = (int)_iNbLayers - 1; i >= 0; i--)
	{
		Layer& l = _pNet->layer(i);
		l.backpropagation(_inOut[i], _gradient[i + 1], _gradient[i]);

		if (l.has_weight())
		{
			if (_pRegularizer)
				_pRegularizer->apply(l.weights(), l.gradient_weights());

			_optimizers[2 * i]->optimize(l.weights(), l.gradient_weights());
		}

		if (l.has_bias())
		{
			//bias does not need regularization 
			_optimizers[2 * i + 1]->optimize(l.bias(), l.gradient_bias());
		}
	}

	//compute and save statistics
	add_online_statistics(_inOut[_iNbLayers], mTruth);
}
/////////////////////////////////////////////////////////////////////////////////////////////
*/