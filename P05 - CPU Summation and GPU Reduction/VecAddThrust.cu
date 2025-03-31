#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>

using namespace std;

int main(){
    size_t inputLength = 3;

    thrust::host_vector<float> hostInput1(inputLength, 10);
    thrust::host_vector<float> hostInput2(inputLength, 1);

    thrust::device_vector<float> deviceInput1(inputLength);
    thrust::device_vector<float> deviceInput2(inputLength);
    thrust::device_vector<float> deviceOutput(inputLength);

    thrust::copy(hostInput1.begin(), hostInput1.end(), deviceInput1.begin());
    thrust::copy(hostInput2.begin(), hostInput2.end(), deviceInput2.begin());

    thrust::transform(deviceInput1.begin(), deviceInput1.end(), deviceInput2.begin(), deviceOutput.begin(), thrust::plus<float>());
    
    for (int i : deviceOutput){
        cout << i << " ";
    }

    cout << endl;
}