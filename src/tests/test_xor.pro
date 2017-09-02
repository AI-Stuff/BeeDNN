TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../DenseLayer.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../ActivationLayer.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../DenseLayer.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../MatrixUtil.h \
    ../ActivationLayer.h
