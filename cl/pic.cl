
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void normalize_particles(__global double *xp, int N, double L) {

  int gid = get_global_id(0);
  const int stride = get_global_size(0);
  while(gid<N) {
    if(xp[gid]<0)  xp[gid]+=L;
    if(xp[gid]>=L) xp[gid]-=L;
    gid += stride;
  }
}

__kernel void weight_CIC(__global double *grid, int nx, double dx,
                         __global const double *xp, __global const double *q, int N) {

  int gid = get_global_id(0);
  const int stride = get_global_size(0);
  int left, right;
  double xis;

  while(gid<N) {
    xis   = xp[gid]/dx;
    left  = (int)(floor(xis));
    right = (left+1)%nx;
    grid[left]  += q[gid]*(left+1-xis);
    grid[right] += q[gid]*(xis-left);
    gid += stride;
  }

}

__kernel void calc_E(__global const double *phi, int nx, double dx,
                     __global double *E) {

  int gid = get_global_id(0);
  const int stride = get_global_size(0);
  int left, right;

  const double scale = -1.0/(2*dx);

  while(gid<nx) {

    if(gid==0)
      E[0] = phi[1]-phi[nx-1];
    else if(gid==nx-1)
      E[nx-1] = phi[0]-phi[nx-2];
    else
      E[gid] = phi[gid+1]-phi[gid-1];

    E[gid] *= scale;

    gid += stride;

  }

}

__kernel void interp_CIC(__global const double *E, int nx, double dx,
                         __global const double *xp, __global double *Ep, int N) {

  int gid = get_global_id(0);
  const int stride = get_global_size(0);
  int left, right;
  double xis;

  while(gid<N) {
    xis   = xp[gid]/dx;
    left  = (int)(floor(xis));
    right = (left+1)%nx;
    Ep[gid] = E[left]*(left+1-xis)+E[right]*(xis-left);
    gid += stride;
  }

}

__kernel void move(__global double *xp, __global const double *vx,
                   double dt, int N) {

  int gid = get_global_id(0);
  const int stride = get_global_size(0);

  while(gid<N) {

    xp[gid] += dt*vx[gid];
    gid += stride;

  }

}
