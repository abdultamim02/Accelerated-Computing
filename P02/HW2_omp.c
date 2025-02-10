#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct node {
    int value;
    struct node *next, *prev;
    omp_lock_t lock;
} Node;

int main(int argc, char *argv[])
{
    double start_time, run_time;
    int N;                 // Number of Nodes to Insert                                                                                                                                                                                                                                                                                                                                                                                                                                     
    unsigned int n = 0;    // Number of Threads                                                                                                                                                                                                                                                                                                                                                                                                                                             

    N = pow(2, 18);

    sscanf(argv[1],"%d", &n);

    omp_set_dynamic(0);
    omp_set_num_threads(n);

    srand(time(0));

    Node *head = NULL;
    Node *p, *prev = NULL; // working pointers                                                                                                                                                                                                                                                                                                                                                                                                                                              

    int value;
    int k;
    Node *newNode;

    start_time = omp_get_wtime();

    // Create First Node                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    head = (Node *)calloc(1,sizeof(Node));
    head->value = 0;
    head->next = NULL;
    head->prev = NULL;
    omp_init_lock(&head->lock);

    omp_lock_t lock;
    omp_init_lock(&lock); // Initialize the lock                                                                                                                                                                                                                                                                                                                    \                                                                                                                       
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    omp_set_dynamic(0);
    omp_set_num_threads(n);

    #pragma omp parallel for private(k, value, newNode, p, prev)
    for(k = 1; k <= N; k++)
    {
        omp_set_lock(&lock);

        fprintf(stderr,"1. thread %d locks\n",omp_get_thread_num());

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

        newNode = (Node *)calloc(1,sizeof(Node));

        fprintf(stderr,"4. thread %d inserts %03d\n",omp_get_thread_num(), value);

        newNode->value = value;
        newNode->next = p;
        newNode->prev = prev;
        prev->next = newNode; // Fix null pointer issue                                                                                                                                                                                                                                                                                                                                                                                                                                     

        fprintf(stderr,"\tthread %d updates newNode->prev->next (%p) to (%p)\n", omp_get_thread_num(), newNode->prev->next, newNode);
        if (p) {
            p->prev = newNode;
        }
        fprintf(stderr,"2. thread %d unlocks\n",omp_get_thread_num());

        omp_unset_lock(&lock);
        printf("k: %d\n\n", k);
    }

    omp_destroy_lock(&lock); // Destroy the lock                                                                                                                                                                                                                                                                                                                                                                                                                                            
    int count = 0;
    p = head;
    while (p != NULL) {
        printf("%03d, ", p->value);
        p = p->next;
        count++;
    }

    printf("\n");

    run_time = omp_get_wtime() - start_time;

    printf("Time Taken to Insert %d Nodes Using %d Threads: %f Seconds\n\n-", count - 1, n, run_time);

    return 0;
}
