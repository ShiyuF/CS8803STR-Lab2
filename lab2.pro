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
INCLUDEPATH += C:/libQGLViewer263
LIBS += -LC:/libQGLViewer263/QGLViewer -lQGLViewer2
