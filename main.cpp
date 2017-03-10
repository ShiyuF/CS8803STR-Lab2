#include "mainwindow.h"
#include <QApplication>
#include <QtCore>

#include "lab2.h"
#include "viewer.h"

unsigned long npoints;
double pos_cent[3];

PointCloud pointCloud[90000];
PointCloud inpPointCloud[90000];

int main(int argc, char *argv[])
{
    unsigned short nrepeats =1;

    QApplication a(argc, argv);

    // Instantiate the viewer.
    Viewer viewer;
    Viewer viewer2;

    viewer.setWindowTitle("raw data");
    viewer2.setWindowTitle("learned data");

    viewer.drawMode = 0; //raw data
    viewer2.drawMode = 1;//learned

    readPointClouds("oakland_part3_am_rf_no_label.node_features");

    findCentroid();

    unsigned long pcInd = 0;
    unsigned long inPCInd = 0;
    int i,j;

    double W[5][10];
    PointCloud* pc;

    //initialize weights
    for(i=0;i<5;i++){
        for(j=0;j<10;j++){
            W[i][j] = 1.0;
        }
    }

//----------------------------------------------------------------
    //reshuffle point cloud

    unsigned short irepeats =0;

    double learningRate =  1/sqrt(npoints*nrepeats);
    while (irepeats <nrepeats){

        pcInd = 0;
        inPCInd = 0;

        qDebug() << irepeats;
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

                for(i=0;i<5;i++){
                    inpPointCloud[inPCInd].node_vec[i] = pc->node_vec[i];
                }

                for(i=0;i<10;i++){
                    inpPointCloud[inPCInd].features[i] = pc->features[i];
                }

                pcInd++;
            }
        }

        //-----------------------------------------------------------------

        //do the learning
        for (pcInd = 0; pcInd < npoints;pcInd++){
            pc =  &inpPointCloud[pcInd];

            double f[10][1];
            double Wf[5][1];
            double Wf_y[5][1];
            double y[5][1];
            double Wf_yT[1][5];
            double f2[10][1];
            double f2Wf_yT[10][5];
            double dLoss[5][10];

            for (i=0;i<5;i++){
                y[i][0] = pc->node_vec[i];
            }
            for (i=0;i<10;i++){
                f[i][0]  = pc->features[i];
                f2[i][0] = 2.0*f[i][0];
            }

            mat_mult((double *) W, 5,10,(double *)f,10,1,(double *)Wf);
            mat_sub ((double *) Wf,5, 1,(double *)y, (double *)Wf_y);
            mat_transpose((double *) Wf_y ,5,1,(double *) Wf_yT);
            mat_mult((double *)f2,10,1,(double *)Wf_yT,1,5,(double *)f2Wf_yT);
            mat_transpose((double *) f2Wf_yT ,10,5,(double *) dLoss);

            for(i=0;i<5;i++){
                for(j=0;j<10;j++){
                    W[i][j] -= learningRate*dLoss[i][j];
                }
            }
        }

        irepeats++;
    }
    //-----------------------------------------------------------------
    //learning is completed
    //test performance

    for (pcInd = 0; pcInd < npoints;pcInd++){
        PointCloud* pc =  &pointCloud[pcInd];
        double Wf[5][1];
        double f[10][1];
        for (i=0;i<10;i++){
            f[i][0]  = pc->features[i];
        }
        mat_mult((double *) W, 5,10,(double *)f,10,1,(double *)Wf);

        //which node does it predict?
        int max_element = 6;
        double max_element_value  =0.0;
        for (i=0;i<5;i++){
            if (Wf[i][0] > max_element_value){
                max_element_value = Wf[i][0];
                max_element = i;
            }
        }

        switch (max_element){
        case 0:
            pc->learned_label = NODE_VEG;
            break;
        case 1:
            pc->learned_label = NODE_WIRE;
            break;
        case 2:
            pc->learned_label = NODE_POLE;
            break;
        case 3:
            pc->learned_label = NODE_GROUND;
            break;
        case 4:
            pc->learned_label = NODE_FACADE;
            break;
        }
    }

    viewer.show();
    viewer2.show();

    return a.exec();
}

void findCentroid(){
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
}

void readPointClouds(char *fileName){

    FILE *fp;
    if((fp = fopen(fileName, "rt")) == NULL) {
        fprintf(stderr, "# Could not open file %s\n", fileName);
        return ;
    }

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
            pc->node_vec[0] = 1.0; pc->node_vec[1] = 0.0; pc->node_vec[2] = 0.0; pc->node_vec[3] = 0.0; pc->node_vec[4] = 0.0;
            break;
        case NODE_WIRE:
            pc->node_vec[0] = 0.0; pc->node_vec[1] = 1.0; pc->node_vec[2] = 0.0; pc->node_vec[3] = 0.0; pc->node_vec[4] = 0.0;
            break;
        case NODE_POLE:
            pc->node_vec[0] = 0.0; pc->node_vec[1] = 0.0; pc->node_vec[2] = 1.0; pc->node_vec[3] = 0.0; pc->node_vec[4] = 0.0;
            break;
        case NODE_GROUND:
            pc->node_vec[0] = 0.0; pc->node_vec[1] = 0.0; pc->node_vec[2] = 0.0; pc->node_vec[3] = 1.0; pc->node_vec[4] = 0.0;
            break;
        case NODE_FACADE:
            pc->node_vec[0] = 0.0; pc->node_vec[1] = 0.0; pc->node_vec[2] = 0.0; pc->node_vec[3] = 0.0; pc->node_vec[4] = 1.0;
            break;
        }

        pcInd++;

        if (pcInd > 89821){ //am rf
            done = 1;
        }
    }

    npoints = pcInd;
}

void mat_init( double *in, int rows, int cols, double init_val ) {

    int i;
    for( i=0; i<rows*cols; i++ )
        in[i] = init_val;

}

void mat_mult( double *A, int na, int ma,
               double *B, int nb, int mb,
               double *C ) {

    int i, j, k;

    if( ma != nb ){return;}

    mat_init( C, na, mb, 0.0 );

    for( i = 0; i < na; i++ )
        for( j = 0; j < mb; j++ )
            for( k = 0; k < ma; k++ )
                C[i*mb+j] += A[i*ma+k]*B[k*mb+j];

}

void mat_transpose( double *A, int na, int ma, double *C ) {

    int i, j;

    mat_init( C, ma, na, 0.0 );

    for( i = 0; i < na; i++ )
        for( j = 0; j < ma; j++ )
            C[j*na+i] = A[i*ma+j];

}

void mat_sub( double *A, int na, int ma, double *B, double *C ) {

    int i, j;

    for( i = 0; i < na; i++ )
        for( j = 0; j < ma; j++ )
            C[i*ma+j] = A[i*ma+j] - B[i*ma+j];

}
