#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct node {
    int value;
    struct node *next, *prev;
} Node;

int main(int argc, char *argv[])
{
    double start_time, run_time;

    srand(time(0));

    Node *head = NULL;
    Node *p, *prev = NULL; // working pointers                                                                                                                                                                                                                                                                                                                      \
\                                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                    \
                                                                                                                                                                                                                                                                                                                                                                     
    int value;
    int k;
    Node *newNode;

    start_time = omp_get_wtime();

    // Create First Node                                                                                                                                                                                                                                                                                                                                            \
                                                                                                                                                                                                                                                                                                                                                                     
    head = (Node *)calloc(1,sizeof(Node));
    head->value = 0;
    head->next = NULL;
    head->prev = NULL;                                                                                                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                                                                                                                                                    \

    for(k = 1; k <= 262144; k++){
        p = head->next;
        prev = head;
        value = rand() % 1000 + 1;

        while (p != NULL) {
            if (p->value >= value){
                break;
            }
            prev = p;
            p = p->next;
        }

        newNode = (Node *)calloc(1, sizeof(Node));

        fprintf(stderr,"Thread %d inserts %03d\n", omp_get_thread_num(), value);

        newNode->value = value;
        newNode->next = p;
        newNode->prev = prev;
        prev->next = newNode; // Fix null pointer issue                                                                                                                                                                                                                                                                                                             \
                                                                                                                                                                                                                                                                                                                                                                     

        fprintf(stderr,"\tThread %d updates newNode->prev->next (%p) to (%p)\n", omp_get_thread_num(), newNode->prev->next, newNode);
        if (p) {
            p->prev = newNode;
        }
    }

    int count = 0;
    p = head;
    while (p != NULL) {
        printf("%03d, ", p->value);
        p = p->next;
        count++;
    }

    printf("\n");

    run_time = omp_get_wtime() - start_time;

    printf("Time Taken to Insert %d Nodes Using 1 Thread: %f Seconds\n\n-", count - 1, run_time);

    printf("\n");

    return 0;
}