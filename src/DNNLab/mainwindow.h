#ifndef MAINWINDOW_H
#define MAINWINDOW_H

class DNNEngine;

#include "Activation.h"

#include <QMainWindow>

class SimpleCurveWidget;

namespace Ui {
class MainWindow;
}

class Net;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    virtual void resizeEvent( QResizeEvent *e );

private slots:
    void on_pushButton_clicked();
    void on_actionQuit_triggered();
    void on_actionAbout_triggered();
    void on_cbEngine_currentTextChanged(const QString &arg1);
    void on_btnTrainMore_clicked();   
    void on_cbYLogAxis_stateChanged(int arg1);

    void on_buttonColor_clicked();

    void on_pushButton_2_clicked();

private:
    void drawLoss(vector<double> vdLoss);
    void drawRegression();
    float compute_truth(float x);
    void train_and_test(bool bReset);
    void update_details();
    void parse_net();

    Ui::MainWindow *ui;

    DNNEngine* _pEngine;
    SimpleCurveWidget* _qsRegression;
    SimpleCurveWidget* _qsLoss;
    unsigned int _curveColor;
};

#endif // MAINWINDOW_H