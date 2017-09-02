#include "DenseLayer.h"

#include <cassert>
#include <cstdlib> // for rand
#include <cmath> // for sqrt


///////////////////////////////////////////////////////////////////////////////
DenseLayer::DenseLayer(int iInSize,int iOutSize): Layer(),
    _iInSize(iInSize),
    _iOutSize(iOutSize)
{
    _weight.resize(_iInSize+1,_iOutSize); //+1 for bias
}
///////////////////////////////////////////////////////////////////////////////
DenseLayer::~DenseLayer()
{ }
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::init()
{
    float a =sqrt(6./(_iInSize+_iOutSize));

    for(unsigned int i=0;i<_weight.size();i++)
    {
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5)*2.*a;
    }

    dE.resize(_iInSize+1,_iOutSize);
    dE.set_zero();
}
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    // compute out=[in 1]*weight; todo use MAC
    mMatOut=mMatIn*(_weight.without_last_row());
    mMatOut+=_weight.row(_iInSize);
}
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::forward_save(const MatrixFloat& mMatIn,MatrixFloat& mMatOut)
{
    in=mMatIn;
    // compute out=[in 1]*weight; todo use MAC
    mMatOut=mMatIn*_weight.without_last_row();
    mMatOut+=_weight.row(_iInSize);

    outWeight=mMatOut;
    out=mMatOut;
}
///////////////////////////////////////////////////////////////////////////////
MatrixFloat DenseLayer::get_weight_activation_derivation()
{
    // apply activation derivation on outweight
    MatrixFloat mOut=outWeight;

    mOut=1.;
    return mOut;
}
///////////////////////////////////////////////////////////////////////////////
MatrixFloat& DenseLayer::get_weight()
{
    return _weight;
}
///////////////////////////////////////////////////////////////////////////////
