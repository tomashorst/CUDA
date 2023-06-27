void saxpy(int n, float a, float * restrict x, float * restrict y)
{
  #pragma acc kernels deviceptr(x,y)
  {
    for(int i=0; i<n; i++)
    {
      y[i] += a*x[i];
    }
  }
}
void set(int n, float val, float * restrict arr)
{
#pragma acc kernels deviceptr(arr)
  {
    for(int i=0; i<n; i++)
    {
      arr[i] = val;
    }
  }
}
