#ifndef VIEWER_H
#define VIEWER_H

#include <QGLViewer/qglviewer.h>
#include <QGLViewer/manipulatedFrame.h>

class Viewer : public QGLViewer
{
protected :
  virtual void draw();
  virtual void init();
  virtual QString helpString() const;
  virtual void postDraw();
  private :
    void drawCornerAxis();
};

#endif // VIEWER_H
