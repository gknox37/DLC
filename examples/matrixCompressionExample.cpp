#include "time.h"
#include "stdio.h"




int main(){
	clock_t begin = clock();

	for(int i = 0; i < 10000000; i++){
	}	

	clock_t end = clock();
	double timeSpent =  ((double)(end - begin)* 1000.0 )/ (CLOCKS_PER_SEC);
	printf("Time: %f ms\n", timeSpent);
		
}
