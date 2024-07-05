// This file will be used for timing of our functions and code performance


typedef struct {

     struct timespec xadd_time1, xadd_time2;
     long long totalTime;
     long totalStops;
    
} chronometer_t;


// Print out the time information and output of each kernel
void timeOutputCode();


// resets values from chronos
void chrono_reset(chronometer_t* chrono);

// record time saving it as the starting time
inline void chrono_start(chronometer_t* chrono);

// get current time and subtract by starting time
// saving it in totalTime, or chrono_gettotal()
inline void chrono_stop(chronometer_t* chrono);

// returns the total time since start to stop
inline long long  chrono_gettotal(chronometer_t* chrono);

// returns the amount of stops since last reset
inline long long  chrono_getcount(chronometer_t* chrono);