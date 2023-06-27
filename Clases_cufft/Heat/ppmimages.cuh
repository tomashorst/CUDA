/* RGB formulae.
 * http://gnuplot-tricks.blogspot.com/2009/06/comment-on-phonged-surfaces-in-gnuplot.html
 * TODO: cannot find what to do with over/under flows, currently saturate in [0,255]
 *
 * there are 37 available rgb color mapping formulae:
 * 0: 0               1: 0.5             2: 1
 * 3: x               4: x^2             5: x^3
 * 6: x^4             7: sqrt(x)         8: sqrt(sqrt(x))
 * 9: sin(90x)        10: cos(90x)       11: |x-0.5|
 * 12: (2x-1)^2       13: sin(180x)      14: |cos(180x)|
 * 15: sin(360x)      16: cos(360x)      17: |sin(360x)|
 * 18: |cos(360x)|    19: |sin(720x)|    20: |cos(720x)|
 * 21: 3x             22: 3x-1           23: 3x-2
 * 24: |3x-1|         25: |3x-2|         26: (3x-1)/2
 * 27: (3x-2)/2       28: |(3x-1)/2|     29: |(3x-2)/2|
 * 30: x/0.32-0.78125 31: 2*x-0.84       32: 4x;1;-2x+1.84;x/0.08-11.5
 * 33: |2*x - 0.5|    34: 2*x            35: 2*x - 0.5
 * 36: 2*x - 1
 * 
 * Some nice schemes in RGB color space
 * 7,5,15   ... traditional pm3d (black-blue-red-yellow)
 * 3,11,6   ... green-red-violet
 * 23,28,3  ... ocean (green-blue-white); try also all other permutations
 * 21,22,23 ... hot (black-red-yellow-white)
 * 30,31,32 ... color printable on gray (black-blue-violet-yellow-white)
 * 33,13,10 ... rainbow (blue-green-yellow-red)
 * 34,35,36 ... AFM hot (black-red-yellow-white)
 */

#define RGBf05 (x*x*x)
#define RGBf07 (sqrt(x))
#define RGBf15 (sin(2*M_PI*x))
#define RGBf21 (3*x)
#define RGBf22 (3*x-1)
#define RGBf23 (3*x-2)
#define RGBf30 (x/0.32-0.78125)
#define RGBf31 (2*x-0.84)
#define RGBf32 (x/0.08-11.5)

#define MAX(a,b) (((a)<(b))?(b):(a))    // maximum
#define MIN(a,b) (((a)<(b))?(a):(b))    // minimum
#define ABS(a) (((a)< 0)?(-a):(a))      // absolute value


#define MAX_COMPONENT_VALUE 255
//void writePPMbinaryImage(const char * filename, const cufftComplex * vector)
void writePPMbinaryImage(const char * filename, const cufftReal * vector)
{
        FILE *f = NULL;
        unsigned int i = 0, j = 0;
        uint8_t RGB[3] = {0,0,0}; /* use 3 first bytes for BGR */ 
        f = fopen(filename, "w");
        assert(f);
        fprintf(f, "P6\n"); /* Portable colormap, binary */
        fprintf(f, "%d %d\n", LY, LX); /* Image size */
        fprintf(f, "%d\n", MAX_COMPONENT_VALUE); /* Max component value */
        for (i=0; i<LX; i++){
                for (j=0; j<LY; j++){
                        /* ColorMaps
                         * http://mainline.brynmawr.edu/Courses/cs120/spring2008/Labs/ColorMaps/colorMap.py
                         */
                        //float x = vector[i*LY+j].x; //HOT; /* [0,1] */
                        float x = (vector[i*LY+j]+2)*0.25; //HOT; /* [0,1] */                    
                        RGB[0] = (uint8_t) MIN(255, MAX(0, 256*RGBf21));
                        RGB[1] = (uint8_t) MIN(255, MAX(0, 256*RGBf22));
                        RGB[2] = (uint8_t) MIN(255, MAX(0, 256*RGBf23));
                        //assert(0<=RGB[0] && RGB[0]<256);
                        //assert(0<=RGB[1] && RGB[1]<256);
                        //assert(0<=RGB[2] && RGB[2]<256);
                        fwrite((void *)RGB, 3, 1, f);
                }
        }
        fclose(f); f = NULL;
        return;
}

