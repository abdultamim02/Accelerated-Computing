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
    double start_time, run_time;    // start_time and run_time of the program
    int N;                          // Number of Nodes to Insert
    unsigned int n = 0;             // Number of Threads
    int value;                      // Node Value
    int k;                          // Iterative k
    int iterations;                 // Iterations Counter

    if (argc < 3) {
        printf("Usage: %s <num_threads> <num_nodes>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1],"%d", &n);
    sscanf(argv[2], "%d", &N);

    omp_set_dynamic(0);
    omp_set_num_threads(n);

    srand(time(0));

    Node *head = NULL;
    Node *p, *prev = NULL;      // working pointers

    Node *newNode;

    // Create First Node
    head = (Node *)calloc(1,sizeof(Node));
    head->value = 0;
    head->next = NULL;
    head->prev = NULL;
    omp_init_lock(&head->lock);

    omp_set_dynamic(0);
    omp_set_num_threads(n);

    start_time = omp_get_wtime();

    #pragma omp parallel for private(k, value, newNode, p, prev)
        for(k = 1; k <= N; k++)
        {
            fprintf(stderr,"1. thread %d locks\n", omp_get_thread_num());

            prev = head;
            omp_set_lock(&prev->lock);         // Lock prev
            p = head->next;
            value = rand() % 1000 + 1;

            while (p != NULL) {
                omp_set_lock(&p->lock);         // Lock p
                if (p->value >= value){
                    break;
                }
                omp_unset_lock(&prev->lock);    // Unlock prev
                prev = p;
                p = p->next;
            }

            newNode = (Node *)calloc(1,sizeof(Node));

            fprintf(stderr,"4. thread %d inserts %03d\n", omp_get_thread_num(), value);

            newNode->value = value;
            newNode->next = p;
            newNode->prev = prev;
            newNode->prev->next = newNode;   // Fix null pointer issue

            omp_init_lock(&newNode->lock);   // Initialize lock before inserting

            omp_unset_lock(&prev->lock);    // Unlock prev

            fprintf(stderr,"\tthread %d updates newNode->prev->next (%p) to (%p)\n", omp_get_thread_num(), newNode->prev->next, newNode);
            if (p) {
                p->prev = newNode;
                omp_unset_lock(&p->lock);   // Unlock p
            }
            fprintf(stderr,"2. thread %d unlocks\n",omp_get_thread_num());

            printf("k: %d\n\n", k);
            iterations = k;
        }

    run_time = omp_get_wtime() - start_time;

    int count = 0;
    p = head;
    while (p != NULL) {
        printf("%03d, ", p->value);
        p = p->next;
        count++;
    }

    printf("\n");
    printf("Time Taken to Insert %d Nodes Using %d Threads: %f Seconds\n\n-", count - 1, n, run_time);
    
    printf("\n\nInserting %d nodes\n", N);
    printf("Time: %f seconds, count: %d, %d threads, %d iterations\n\n", run_time, count, n, iterations);

    return 0;
}