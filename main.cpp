#include "mainwindow.h"
#include <QApplication>
#include <QtCore>
#include <random>

#include "lab2.h"
#include "viewer.h"

unsigned long npoints1;
unsigned long npoints2;
unsigned long npoints;

double pos_cent1[3];
double pos_cent2[3];
double mean_features2[10];
double stddev_features2[10];

PointCloud pointCloud1[90000];
PointCloud pointCloud2[37000];
PointCloud inpPointCloud[90000];

std::default_random_engine generator;

int main(int argc, char *argv[])
{


    QApplication a(argc, argv);

    // Instantiate the viewer.
    Viewer viewer1;
    Viewer viewer1_learned;
    Viewer viewer2;
    Viewer viewer2_learned;

    viewer1.setWindowTitle("raw data 1");
    viewer1_learned.setWindowTitle("learned data 1");

    viewer2.setWindowTitle("raw data 2");
    viewer2_learned.setWindowTitle("learned data 2");

    viewer1.drawMode = 0; //raw data
    viewer1_learned.drawMode = 1;//learned

    viewer2.drawMode = 0; //raw data
    viewer2_learned.drawMode = 1;//learned

    readPointClouds1("oakland_part3_am_rf_no_label.node_features");
    readPointClouds2("oakland_part3_an_rf_no_label.node_features");

    findCentroid1();
    findCentroid2();

    //add noise of dataset2
    addNoise2();

    viewer1.drawMode = 0; //raw data
    viewer1_learned.drawMode = 1;//learned

    viewer1.drawingData = 1; //draw first data set
    viewer1_learned.drawingData = 1;

    viewer2.drawMode = 0; //raw data
    viewer2_learned.drawMode = 1;//learned

    viewer2.drawingData = 2; //draw second data set
    viewer2_learned.drawingData = 2;

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
    unsigned short nrepeats =10;
    unsigned short irepeats =0;
    double learningRate;

    //learn both data sets
    while (irepeats <nrepeats){
        qDebug() << irepeats;

       //the first set
        npoints = npoints1;
        learningRate =  1/sqrt(npoints*nrepeats);

        //reshuffle the first set
        inPCInd = 0;
        while ( inPCInd < npoints){
            inpPointCloud[inPCInd].node_label = 0;
            inPCInd++;
        }

        pcInd = 0;
        while ( pcInd < npoints ){
            pc =  &pointCloud1[pcInd];

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
        //end reshuffle the data

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

#if 1
        //option to learn with second set also

        npoints = npoints2;
        learningRate =  1/sqrt(npoints*nrepeats);

        //reshuffle the second set
        inPCInd = 0;
        while ( inPCInd < npoints){
            inpPointCloud[inPCInd].node_label = 0;
            inPCInd++;
        }

        pcInd = 0;
        while ( pcInd < npoints ){
            pc =  &pointCloud2[pcInd];

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
        //end reshuffle the data

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
        //end learning
        //--------------------------
#endif
        irepeats++;
    }

    //-----------------------------------------------------------------
    //learning is completed
    //test performance

    //test first data set
    npoints = npoints1;
    for (pcInd = 0; pcInd < npoints;pcInd++){
        PointCloud* pc =  &pointCloud1[pcInd];
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

    //test second data set
    npoints = npoints2;
    for (pcInd = 0; pcInd < npoints;pcInd++){
        PointCloud* pc =  &pointCloud2[pcInd];
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

    viewer1.show();
    viewer1_learned.show();

    viewer2.show();
    viewer2_learned.show();

    return a.exec();
}

void findCentroid1(){
    unsigned long pcInd = 0;
    double sum_pos[3];
    PointCloud *pc;
    sum_pos[0] = 0.0; sum_pos[1] = 0.0; sum_pos[2] = 0.0;
    for (pcInd=0; pcInd < npoints1; pcInd++ ){
        pc = &pointCloud1[pcInd];
        sum_pos[0] += pc->pos[0];
        sum_pos[1] += pc->pos[1];
        sum_pos[2] += pc->pos[2];
    }

    pos_cent1[0] = sum_pos[0]/npoints1;
    pos_cent1[1] = sum_pos[1]/npoints1;
    pos_cent1[2] = sum_pos[2]/npoints1;
}

void findCentroid2(){
    unsigned long pcInd = 0;
    double sum_pos[3];
    PointCloud *pc;
    sum_pos[0] = 0.0; sum_pos[1] = 0.0; sum_pos[2] = 0.0;
    for (pcInd=0; pcInd < npoints2; pcInd++ ){
        pc = &pointCloud2[pcInd];
        sum_pos[0] += pc->pos[0];
        sum_pos[1] += pc->pos[1];
        sum_pos[2] += pc->pos[2];
    }

    pos_cent2[0] = sum_pos[0]/npoints2;
    pos_cent2[1] = sum_pos[1]/npoints2;
    pos_cent2[2] = sum_pos[2]/npoints2;
}

void readPointClouds1(char *fileName){

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

        pc = &pointCloud1[pcInd];

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

    npoints1 = pcInd;
}

void readPointClouds2(char *fileName){

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

        pc = &pointCloud2[pcInd];

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

        if (pcInd > 36396){ //an rf
            done = 1;
        }
    }

    npoints2 = pcInd;
}

//add noise to second dataset
void addNoise2(){
//first find mean of the features
    double sum_features2[10];
    unsigned short i;
    unsigned long pcInd = 0;
    PointCloud *pc;

    //initialize to 0
    for (i=0;i<10;i++){
        sum_features2[i] = 0.0;
    }

    //cumulative sum
    for (pcInd=0; pcInd < npoints2; pcInd++ ){
        pc = &pointCloud2[pcInd];
        for (i=0;i<10;i++){
            sum_features2[i] += pc->features[i];
        }
    }

    //find the mean
    for (i=0;i<10;i++){
        mean_features2[i] =  sum_features2[i]/npoints2;
        //qDebug() << mean_features2[i];
    }

    //initialize to 0
    for (i=0;i<10;i++){
        sum_features2[i] = 0.0;
    }

    //cumulative sum of variance
    for (pcInd=0; pcInd < npoints2; pcInd++ ){
        pc = &pointCloud2[pcInd];
        for (i=0;i<10;i++){
            sum_features2[i] += (pc->features[i] - mean_features2[i])*(pc->features[i] - mean_features2[i]) ;
        }
    }

    //find the variance
   for (i=0;i<10;i++){
       stddev_features2[i] =  sqrt(sum_features2[i]/npoints2);
   }

   //now...add the noise
   for (pcInd=0; pcInd < npoints2; pcInd++ ){
       pc = &pointCloud2[pcInd];
       for (i=0;i<10;i++){
           std::normal_distribution<double> distribution1( mean_features2[i],stddev_features2[i]);
           pc->features[i] += distribution1(generator);
       }
   }

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
