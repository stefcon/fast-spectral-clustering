
#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <sys/time.h>

enum PrintColor { NONE, GREEN, DGREEN, CYAN };

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

static void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

static void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

static void printElapsedTime(Timer timer, const char* s, enum PrintColor color = NONE) {
    float t = ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
    switch(color) {
        case GREEN:  printf("\033[1;32m"); break;
        case DGREEN: printf("\033[0;32m"); break;
        case CYAN :  printf("\033[1;36m"); break;
    }
    printf("%s: %f ms", s, t*1e3);
    if(color != NONE) {
        printf("\033[0m");
    }
    printf("\n");
}

static float getElapsedTime(Timer timer) {
    float t = ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
    return t;
}

#endif

