#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

// Function to check a condition and log an error if the condition is true
int check(int condition, const char *file, int line, const char *func, const char *msg, ...) {
    if (condition) {
        va_list args;
        va_start(args, msg);  // Initialize the va_list with the last known fixed parameter
        
        fprintf(stderr, "[ERROR] (%s:%d:%s) ", file, line, func);
        if (errno) {
          vfprintf(stderr, msg, args);  // Use vfprintf to handle the variable arguments
          fprintf(stderr, ": %s\n", strerror(errno));
        } else
          vfprintf(stderr, msg, args);  // Use vfprintf to handle the variable arguments

        va_end(args);  // Clean up the va_list
        errno = 0;
        return 1;
    }
    return 0;
}

