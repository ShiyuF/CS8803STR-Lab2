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
  short node;
  short svm_label;
  float features[10];
} pointCloud[90000],inpPointCloud[90000];

extern struct Expert{
    double w[10];
} expert;

extern unsigned long npoints;
extern double pos_cent[3];
void readPointClouds(char *fileName);
void findCentroid();

#endif // LAB2_H

