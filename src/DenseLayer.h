#ifndef DenseLayer_
#define DenseLayer_

#include "Layer.h"
#include "Matrix.h"

class DenseLayer : public Layer
{
public:
    DenseLayer(int iInSize,int iOutSize);
    ~DenseLayer();
    virtual void init();

    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const;
    virtual void forward_save(const MatrixFloat& mMatin, MatrixFloat &mMatOut);

    virtual MatrixFloat get_weight_activation_derivation();
    virtual MatrixFloat& get_weight();

private:
    MatrixFloat _weight;
    int _iInSize, _iOutSize;
};

#endif
