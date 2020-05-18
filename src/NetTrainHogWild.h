/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef NetTrainHogWild_
#define NetTrainHogWild_

#include "NetTrain.h"

class NetTrainHogWild: public NetTrain
{
public:
    NetTrainHogWild();
    virtual ~NetTrainHogWild();

protected:
	virtual void train_one_epoch(Index iBatchSize, const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled) override;
};

#endif
