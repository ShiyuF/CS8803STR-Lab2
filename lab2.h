#ifndef LAB2_H
#define LAB2_H

#define NODE_VEG 1004
#define NODE_WIRE 1100
#define NODE_POLE 1003
#define NODE_GROUND 1200
#define NODE_FACADE 1400

extern struct PointCloud{
  float pos[3];
  short node_label;
  float node_vec[5];
  float learned_label;
  float features[10];
} pointCloud[90000],inpPointCloud[90000];

extern unsigned long npoints;
extern double pos_cent[3];
void readPointClouds(char *fileName);
void findCentroid();

void mat_init( double *in, int rows, int cols, double init_val );
void mat_transpose( double *A, int na, int ma, double *C );
void mat_mult( double *A, int na, int ma,
               double *B, int nb, int mb,
               double *C );
void mat_sub( double *A, int na, int ma, double *B, double *C );

#endif // LAB2_H

