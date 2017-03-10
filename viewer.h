#ifndef VIEWER_H
#define VIEWER_H

#include <QGLViewer/qglviewer.h>
#include <QGLViewer/manipulatedFrame.h>

class Viewer : public QGLViewer
{
public :
    int drawMode; //0 original 1 learned
protected :
  virtual void draw();
  virtual void init();
  virtual QString helpString() const;
  virtual void postDraw();

  private :
    void drawCornerAxis();
};

#endif // VIEWER_H
