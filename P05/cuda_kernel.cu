#include <stdio.h>

__global__ void kernel(void){   
}

int main(){
    printf("Before kernel invocation...");
    kernel<<<1,1>>>();
    printf("done\n");
    return 0;
}