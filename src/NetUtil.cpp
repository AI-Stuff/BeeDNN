/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "NetUtil.h"
#include "Layer.h"
#include "Matrix.h"

#include "LayerDropout.h"
#include "LayerGlobalGain.h"

#include <sstream>
#include <fstream>
using namespace std;

namespace NetUtil {

/////////////////////////////////////////////////////////////////////////////////////////////////
string to_string(const Net* pNet)
{
    stringstream ss;

    auto layers=pNet->layers();
    ss << "Engine: testDnn" << endl;
    ss << endl;
    ss << "NbLayers: " << layers.size() << endl;
    ss << endl;

    ss << "----------------------------------------------" << endl;
    for(size_t i=0;i<layers.size();i++)
    {
        Layer* layer=layers[i];

        if(layer->type()=="Dense")
        {
            ss << "Dense:  InSize: " << layer->in_size() << " OutSize: " << layer->out_size() ;

            if(layer->in_size()!=layer->weights().rows())
              ss << " (with bias)" << endl;
            else
                ss << " (without bias)" << endl;

            ss << "Weight:" << endl;
            ss << toString(layer->weights()) << endl;
        }

        else if(layer->type()=="GlobalGain")
        {
            LayerGlobalGain* l=static_cast<LayerGlobalGain*>(layer);
            ss << "GlobalGain: gain=" << l->gain() << endl;
        }

        else if(layer->type()=="PoolAveraging1D")
            ss << "PoolAveraging1D:  InSize: " << layer->in_size() << " OutSize: " << layer->out_size() << endl;

        else if(layer->type()=="Dropout")
        {
            LayerDropout* l=static_cast<LayerDropout*>(layer);
            ss << "Dropout: rate=" << l->get_rate() << endl;
        }
        else
        {
            ss << "Activation: " << layer->type() << endl;
        }
        ss << "----------------------------------------------" << endl;
    }

    return ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net* from_string(const string& s)
{
    Net* pNet=new Net;

    (void)s;
    //todo

    return pNet;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
bool save(string sFileName,const Net* pNet)
{
    string s=to_string(pNet);
    ofstream f(sFileName);
    f << s;
    f.close();

    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////
bool load(string sFileName,Net* pNet)
{
    (void)pNet;

    ifstream f(sFileName);
/*
    string s=to_string(pNet);
    //TODO
    f << s;
    f.close();
*/
    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////

}
