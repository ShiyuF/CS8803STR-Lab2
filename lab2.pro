#-------------------------------------------------
#
# Project created by QtCreator 2017-02-26T01:35:16
#
#-------------------------------------------------

QT       += core gui opengl xml svg openglextensions

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = lab2
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    viewer.cpp

HEADERS  += mainwindow.h \
    viewer.h \
    lab2.h

FORMS    += mainwindow.ui


#libQGLViewer for opengl and map
LIBS += -lopengl32 -lglu32



win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../../../libQGLViewer-2.6.4/QGLViewer/release/ -lQGLViewer
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../../../libQGLViewer-2.6.4/QGLViewer/debug/ -lQGLViewer
else:mac: LIBS += -F$$PWD/../../../../../../../libQGLViewer-2.6.4/QGLViewer/ -framework QGLViewer
else:unix: LIBS += -L$$PWD/../../../../../../../libQGLViewer-2.6.4/QGLViewer/ -lQGLViewer

INCLUDEPATH += $$PWD/../../../../../../../libQGLViewer-2.6.4/QGLViewer
DEPENDPATH += $$PWD/../../../../../../../libQGLViewer-2.6.4/QGLViewer
