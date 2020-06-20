#include "omp.h"
#include <cmath>
#include <cstring>

#include "omp_entropy.h"

#define MAX_INT_OF_INPUT 16                    // 假设元素为[0,15]的整型
#define RADIUS           2                     // 2 = 5/2
#define DIAMETER         ((RADIUS*2)+1)        // 5=2*2+1
#define MAX_WINDOW_AREA  (DIAMETER*DIAMETER)   // Max Window Area


static void get_const_logs(float *const_logs, int size) {
	const_logs[0] = 0.0;
	for (int i = 1; i < size; i++ ) {
		const_logs[i] = log2f(float(i));
	}
}

inline float get_etropy(const int *bucket, const int area, const float* const_logs) {
	float ent = 0.0;
	float area_log = const_logs[area];
	for (int k = 0; k < MAX_INT_OF_INPUT; k++) {
		if ( bucket[k] > 0 ) {
			ent -= ( float(bucket[k]) / float(area) ) * 
					(const_logs[bucket[k]] - area_log);
		}
	}
	return ent;
}

void openmp_entropy(const int width, const int height, const float *sample, float *result) 
{
	float *const_logs = new float[MAX_WINDOW_AREA + 1];
	get_const_logs(const_logs, MAX_WINDOW_AREA + 1);

	#pragma omp parallel for schedule(dynamic) 
	for (int i = 0; i < height; i++) 
	{
		int left_yw  = ( i+1 > RADIUS ? RADIUS:i );
		int right_yw = ( height-i > RADIUS ? RADIUS:(height-1-i));
		int xw = RADIUS;
		int bucket[MAX_INT_OF_INPUT];
        
        // init bucket
        std::memset(bucket, 0, sizeof(int)*MAX_INT_OF_INPUT); 
        for (int j = 0; j < RADIUS; j++) {
        for (int ii = -left_yw; ii <= right_yw; ii++) {
            bucket[ int(sample[(i+ii)*width+j]) ]++;
            }	
        }

		for (int j = 0; j < width; j++) {
			for (int ii = -left_yw; ii <= right_yw; ii++) {
				if (j > RADIUS) {
					bucket[ int(sample[ (i+ii)*width+(j-RADIUS-1) ]) ]--;
				}
				if (j < width-RADIUS) {
					bucket[ int(sample[ (i+ii)*width+(j+RADIUS)   ]) ]++;
				}
			}	

			if (j <= RADIUS)  xw++;
			if (j >= width-RADIUS) xw--;
			result[i*width+j] = get_etropy(bucket, (xw*(left_yw+1+right_yw)), const_logs);
		}
	}

	delete [] const_logs;
}