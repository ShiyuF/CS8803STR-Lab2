#ifndef LAB2_H
#define LAB2_H

#define NODE_VEG 1004
#define NODE_WIRE 1100
#define NODE_POLE 1103
#define NODE_GROUND 1200
#define NODE_FACADE 1400

extern struct PointCloud{
  float pos[3];
  short node_label;
  float node_vec[5];
  float learned_label;
  float features[10];
} pointCloud1[90000],pointCloud2[37000],inpPointCloud[90000];

extern unsigned long npoints1;
extern unsigned long npoints2;
extern double pos_cent1[3];
extern double pos_cent2[3];
void readPointClouds1(char *fileName);
void readPointClouds2(char *fileName);
void findCentroid1();
void findCentroid2();

void mat_init( double *in, int rows, int cols, double init_val );
void mat_transpose( double *A, int na, int ma, double *C );
void mat_mult( double *A, int na, int ma,
               double *B, int nb, int mb,
               double *C );
void mat_sub( double *A, int na, int ma, double *B, double *C );

#endif // LAB2_H

