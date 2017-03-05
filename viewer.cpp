#include "viewer.h"
#include "lab2.h"

using namespace std;
GLuint index;

// Draws a spiral
void Viewer::draw()
{
    // Draw an axis using the QGLViewer static function
    glClearColor (0.0,0.0,0.0,1.0);

    PointCloud *pc;
    unsigned long pcInd = 0;

    glPointSize(2.0f);

    glBegin(GL_POINTS);

    for (pcInd=0; pcInd < npoints; pcInd++ ){
        pc = &pointCloud[pcInd];
        switch (pc->node_label){
        case NODE_VEG:
            glColor4f(0.0, 1.0f , 0.0f,1.0f);
            break;
        case NODE_WIRE:
            glColor4f(0.2, 0.2f , 0.2f,1.0f);
            break;
        case NODE_POLE:
            glColor4f(0.6, 0.6f , 0.6f,1.0f);
            break;
        case NODE_GROUND:
            glColor4f(0.5, 0.27f , 0.07f,1.0f);
            break;
        case NODE_FACADE:
            glColor4f(1.0, 0.89f , 0.77f,1.0f);
            break;
        }
        glVertex3f(pc->pos[0]-pos_cent[0], pc->pos[1]-pos_cent[1], pc->pos[2]-pos_cent[2]);
    }
    glEnd();

}

void Viewer::init()
{
    // Restore previous viewer state.
    restoreStateFromFile();
    camera()->setZClippingCoefficient(50.0);

    //help();
}

QString Viewer::helpString() const
{
    QString text("<h2>S i m p l e V i e w e r</h2>");
    text += "Use the mouse to move the camera around the object. ";
    text += "You can respectively revolve around, zoom and translate with the three mouse buttons. ";
    text += "Left and middle buttons pressed together rotate around the camera view direction axis<br><br>";
    text += "Pressing <b>Alt</b> and one of the function keys (<b>F1</b>..<b>F12</b>) defines a camera keyFrame. ";
    text += "Simply press the function key again to restore it. Several keyFrames define a ";
    text += "camera path. Paths are saved when you quit the application and restored at next start.<br><br>";
    text += "Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, ";
    text += "<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save a snapshot. ";
    text += "See the <b>Keyboard</b> tab in this window for a complete shortcut list.<br><br>";
    text += "Double clicks automates single click actions: A left button double click aligns the closer axis with the camera (if close enough). ";
    text += "A middle button double click fits the zoom of the camera and the right button re-centers the scene.<br><br>";
    text += "A left button double click while holding right button pressed defines the camera <i>Revolve Around Point</i>. ";
    text += "See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>";
    text += "Press <b>Escape</b> to exit the viewer.";
    return text;
}

void Viewer::drawCornerAxis()
{
    int viewport[4];
    int scissor[4];

    // The viewport and the scissor are changed to fit the lower left
    // corner. Original values are saved.
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetIntegerv(GL_SCISSOR_BOX, scissor);

    // Axis viewport size, in pixels
    const int size = 150;
    glViewport(0,0,size,size);
    glScissor(0,0,size,size);

    // The Z-buffer is cleared to make the axis appear over the
    // original image.
    glClear(GL_DEPTH_BUFFER_BIT);

    // Tune for best line rendering
    glDisable(GL_LIGHTING);
    glLineWidth(3.0);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMultMatrixd(camera()->orientation().inverse().matrix());

    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(1.0, 0.0, 0.0);

    glColor3f(0.0, 1.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 1.0, 0.0);

    glColor3f(0.0, 0.0, 1.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 1.0);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glEnable(GL_LIGHTING);

    // The viewport and the scissor are restored.
    glScissor(scissor[0],scissor[1],scissor[2],scissor[3]);
    glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
}

// The thumbnail has to be drawn at the very end to allow a correct
// display of the visual hints (axes, grid, etc).
void Viewer::postDraw()
{
    QGLViewer::postDraw();
    drawCornerAxis();
}

