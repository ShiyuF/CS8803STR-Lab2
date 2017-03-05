#include "mainwindow.h"
#include <QApplication>
#include <QtCore>

#include "lab2.h"
#include "viewer.h"

unsigned long npoints;
double pos_cent[3];

PointCloud pointCloud[90000];
PointCloud inpPointCloud[90000];
Expert expert;

int i;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //MainWindow w;
    //w.show();

    // Instantiate the viewer.
    Viewer viewer;
    viewer.setWindowTitle("simpleViewer");

    // Make the viewer window visible on screen.

    viewer.show();
    readPointClouds("oakland_part3_am_rf_no_label.node_features");

    findCentroid();

    unsigned long pcInd = 0;
    unsigned long inPCInd = 0;
    double learningRate;
    double lam;

    PointCloud* pc;
    Expert* e  = &expert;
    //-----------------------------------------
    //initialize weights
    for(i=0;i<10;i++){
        e->w[i] = 0.0;
    }

    learningRate = 1/sqrt(npoints);
    lam =10.0;
    pcInd = 0;
    inPCInd = 0;

    //for (pcInd = 0; pcInd < npoints;pcInd++){
        for (pcInd = 0; pcInd < 1000;pcInd++){
        pc =  &pointCloud[pcInd];

        if(pc->svm_label !=0){//only update if either one of two classes

            for(i=0;i<10;i++){
                e->w[i] -= learningRate*lam*e->w[i];
            }

            //if misrank


        }


    }



    //--------------------------------------------
#if 0

    Expert* e  = &expert;

    //initialize weights
    for(i=0;i<10;i++){
        e->w[i] = 1.0;
    }

    double prediction = 0.0;
    double pred_m_obs;
    double gradLoss[10];
    double learningRate = 1/sqrt(npoints);
    qDebug() << "learning rate" << learningRate;

    //reshuffle point cloud
    unsigned short nrepeats =0;

learningRate =  1/sqrt(npoints*35);
    while (nrepeats <35){

    pcInd = 0;
    inPCInd = 0;

qDebug() << nrepeats;
while ( inPCInd < npoints ){
 inpPointCloud[inPCInd].node_label = 0;
 inPCInd++;
}
 inPCInd =0;

    while ( pcInd < npoints ){
        pc =  &pointCloud[pcInd];

        inPCInd = (qrand()+ qrand() +qrand())%npoints;

        if (inpPointCloud[inPCInd].node_label == 0){//not taken yet
            for(i=0;i<3;i++){
                inpPointCloud[inPCInd].pos[i] = pc->pos[i];
            }

            inpPointCloud[inPCInd].node_label = pc->node_label;
            inpPointCloud[inPCInd].node = pc->node;

            for(i=0;i<10;i++){
                inpPointCloud[inPCInd].features[i] = pc->features[i];
            }

            pcInd++;
        }
    }


    for (pcInd = 0; pcInd < npoints;pcInd++){
       // for (pcInd = 0; pcInd < 1000;pcInd++){
        pc =  &inpPointCloud[pcInd];

        prediction = 0.0;
        for(i=0;i<10;i++){
            prediction += e->w[i]*pc->features[i];
        }
        //pred_m_obs = prediction - pc->node_label;
        pred_m_obs = prediction - pc->node;
        //loss = pred_m_obs*pred_m_obs;

        //find gradient  of loss
        for(i=0;i<10;i++){
            gradLoss[i] = 2.0*pred_m_obs*pc->features[i];
        }

        //update the weight
        //learningRate =  1/sqrt((pcInd+1)*(nrepeats+1));
        for(i=0;i<10;i++){
            e->w[i] -= learningRate*gradLoss[i];
        }

        //maintain convexity by scale back the norm
        double norm2 =0.0;
        for(i=0;i<10;i++){
            norm2 += e->w[i]*e->w[i];
        }

        double normBound = 1e3;
        double normBound2 = normBound*normBound;
        /*if (norm2 > normBound2){
            for(i=0;i<10;i++){
                e->w[i] = e->w[i]/normBound;
            }
        }*/

       // qDebug()<< nrepeats  << pcInd ;
    }

    nrepeats++;

    }

    //learning is completed
    //test performance
    for(i=0;i<10;i++){
        qDebug() << e->w[i];
    }


    for (pcInd = 0; pcInd < 1000;pcInd++){
        PointCloud* pc =  &pointCloud[pcInd];
        prediction = 0.0;

        for(i=0;i<10;i++){
            prediction += e->w[i]*pc->features[i];
        }
        qDebug() <<  prediction << pc->node;
    }
#endif

    return a.exec();
}

void findCentroid(){
    //find centroid
    unsigned long pcInd = 0;
    double sum_pos[3];
    PointCloud *pc;
    sum_pos[0] = 0.0; sum_pos[1] = 0.0; sum_pos[2] = 0.0;
    for (pcInd=0; pcInd < npoints; pcInd++ ){
        pc = &pointCloud[pcInd];
        sum_pos[0] += pc->pos[0];
        sum_pos[1] += pc->pos[1];
        sum_pos[2] += pc->pos[2];
    }

    pos_cent[0] = sum_pos[0]/npoints;
    pos_cent[1] = sum_pos[1]/npoints;
    pos_cent[2] = sum_pos[2]/npoints;
    //qDebug() << pos_cent[0] << pos_cent[1] << pos_cent[2];

}

void readPointClouds(char *fileName){

    FILE *fp;
    if((fp = fopen(fileName, "rt")) == NULL) {
        fprintf(stderr, "# Could not open file %s\n", fileName);
        return ;
    }

    unsigned short i;
    unsigned long pcInd = 0;
    unsigned short dummy;
    int done =0;
    PointCloud *pc;
    while(!done){

        pc = &pointCloud[pcInd];

        fscanf(fp,"%f %f %f %d %d %f %f %f %f %f %f %f %f %f %f",
               &pc->pos[0],&pc->pos[1],&pc->pos[2],&dummy,&pc->node_label,
                &pc->features[0],&pc->features[1],&pc->features[2],
                &pc->features[3],&pc->features[4],&pc->features[5],
                &pc->features[6],&pc->features[7],&pc->features[8],&pc->features[9]);

        switch (pc->node_label){
        case NODE_VEG:
            pc->node = 10;
            pc->svm_label = -1;
            break;
        case NODE_WIRE:
            pc->node = 20;
            break;
        case NODE_POLE:
            pc->node = 30;
            break;
        case NODE_GROUND:
            pc->node = 40;
            break;
        case NODE_FACADE:
            pc->node = 50;
            pc->svm_label = 1;
            break;
        }

        pcInd++;

        if (pcInd > 89821){ //am rf
            done = 1;
        }
    }

    npoints = pcInd;

    /*
qDebug() << pcInd << pc->pos[0] << pc->pos[1] << pc->pos[2] << pc->node_label;
for (i=0;i<10;i++){qDebug() << i << pc->features[i];}
*/

}
