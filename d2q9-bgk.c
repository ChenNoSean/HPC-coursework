/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' indicates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
** \|/
** 3-0-1
** /|\
** 7 4 8
**
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* SoA Access Macro: ensures same direction (kk) is contiguous in memory */
#define SPEED_IDX(ii, jj, kk, nx, ny) ((kk) * (nx) * (ny) + (jj) * (nx) + (ii))

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* function prototypes */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

int timestep(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles);
int accelerate_flow(const t_param params, float* restrict cells, int* restrict obstacles);
int propagate(const t_param params, float* restrict cells, float* restrict tmp_cells);
int rebound(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles);
int collision(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles);

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

float total_density(const t_param params, float* cells);
float av_velocity(const t_param params, float* cells, int* obstacles);
float calc_reynolds(const t_param params, float* cells, int* obstacles);

void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/* ========================= main ========================= */

int main(int argc, char* argv[])
{
  char* paramfile = NULL;
  char* obstaclefile = NULL;

  t_param params;
  float* cells = NULL;
  float* tmp_cells = NULL;
  int* obstacles = NULL;
  float* av_vels = NULL;

  struct timeval timstr;
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc;

  if (argc != 3)
  {
    usage(argv[0]);
  }
  paramfile = argv[1];
  obstaclefile = argv[2];

  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic = tot_tic;

  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic = init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep(params, cells, tmp_cells, obstacles);

    /* swap pointers: tmp_cells becomes new cells */
    float* swap_ptr = cells;
    cells = tmp_cells;
    tmp_cells = swap_ptr;

    av_vels[tt] = av_velocity(params, cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;

  /* no MPI in this coursework version */
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);

  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

/* ========================= timestep ========================= */

int timestep(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles)
{
  accelerate_flow(params, cells, obstacles);

  /* stream: cells -> tmp_cells */
  propagate(params, cells, tmp_cells);

  /* bounce-back: tmp_cells -> cells (only obstacles updated) */
  rebound(params, cells, tmp_cells, obstacles);

  /* collide: cells -> tmp_cells (full grid) */
  collision(params, cells, tmp_cells, obstacles);

  return EXIT_SUCCESS;
}

/* ========================= accelerate ========================= */

int accelerate_flow(const t_param params, float* restrict cells, int* restrict obstacles)
{
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  const int jj = params.ny - 2;

  #pragma omp parallel for schedule(static)
  for (int ii = 0; ii < params.nx; ii++)
  {
    const int idx_obs = ii + jj * params.nx;
    if (!obstacles[idx_obs]
        && (cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] - w1) > 0.f
        && (cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] - w2) > 0.f
        && (cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] - w2) > 0.f)
    {
      cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] += w1;
      cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] += w2;
      cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] += w2;

      cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] -= w1;
      cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] -= w2;
      cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

/* ========================= propagate (streaming) ========================= */

int propagate(const t_param params, float* restrict cells, float* restrict tmp_cells)
{
  #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    const int y_n = (jj + 1) % params.ny;
    const int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);

    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int x_e = (ii + 1) % params.nx;
      const int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);

      tmp_cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)] = cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)];
      tmp_cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] = cells[SPEED_IDX(x_w, jj, 1, params.nx, params.ny)];
      tmp_cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)] = cells[SPEED_IDX(ii, y_s, 2, params.nx, params.ny)];
      tmp_cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] = cells[SPEED_IDX(x_e, jj, 3, params.nx, params.ny)];
      tmp_cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)] = cells[SPEED_IDX(ii, y_n, 4, params.nx, params.ny)];
      tmp_cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] = cells[SPEED_IDX(x_w, y_s, 5, params.nx, params.ny)];
      tmp_cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] = cells[SPEED_IDX(x_e, y_s, 6, params.nx, params.ny)];
      tmp_cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] = cells[SPEED_IDX(x_e, y_n, 7, params.nx, params.ny)];
      tmp_cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] = cells[SPEED_IDX(x_w, y_n, 8, params.nx, params.ny)];
    }
  }

  return EXIT_SUCCESS;
}

/* ========================= rebound (bounce-back) ========================= */

int rebound(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles)
{
  #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (obstacles[ii + jj * params.nx])
      {
        /* bounce-back: write into cells using streamed values (tmp_cells) */
        cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)];
      }
      else
      {
        /* non-obstacle: just copy streamed values into cells */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)] =
              tmp_cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)];
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

/* ========================= collision (BGK) ========================= */

int collision(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles)
{
  const float c_sq = 1.f / 3.f;
  const float w0   = 4.f / 9.f;
  const float w1   = 1.f / 9.f;
  const float w2   = 1.f / 36.f;

  #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int obs = obstacles[ii + jj * params.nx];

      if (obs)
      {
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)] =
              cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)];
        }
        continue;
      }

      const float p0 = cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)];
      const float p1 = cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)];
      const float p2 = cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)];
      const float p3 = cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)];
      const float p4 = cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)];
      const float p5 = cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)];
      const float p6 = cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)];
      const float p7 = cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)];
      const float p8 = cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)];

      const float local_density = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;

      const float u_x = (p1 + p5 + p8 - (p3 + p6 + p7)) / local_density;
      const float u_y = (p2 + p5 + p6 - (p4 + p7 + p8)) / local_density;
      const float u_sq = u_x*u_x + u_y*u_y;

      const float d0 = w0 * local_density * (1.f - u_sq / (2.f*c_sq));

      const float d1 = w1 * local_density * (1.f + u_x/c_sq + (u_x*u_x)/(2.f*c_sq*c_sq) - u_sq/(2.f*c_sq));
      const float d3 = w1 * local_density * (1.f - u_x/c_sq + (u_x*u_x)/(2.f*c_sq*c_sq) - u_sq/(2.f*c_sq));

      const float d2 = w1 * local_density * (1.f + u_y/c_sq + (u_y*u_y)/(2.f*c_sq*c_sq) - u_sq/(2.f*c_sq));
      const float d4 = w1 * local_density * (1.f - u_y/c_sq + (u_y*u_y)/(2.f*c_sq*c_sq) - u_sq/(2.f*c_sq));

      const float u5 =  u_x + u_y;
      const float u6 = -u_x + u_y;
      const float u7 = -u_x - u_y;
      const float u8 =  u_x - u_y;

      const float d5 = w2 * local_density * (1.f + u5/c_sq + (u5*u5)/(2.f*c_sq*c_sq) - u_sq/(2.f*c_sq));
      const float d6 = w2 * local_density * (1.f + u6/c_sq + (u6*u6)/(2.f*c_sq*c_sq) - u_sq/(2.f*c_sq));
      const float d7 = w2 * local_density * (1.f + u7/c_sq + (u7*u7)/(2.f*c_sq*c_sq) - u_sq/(2.f*c_sq));
      const float d8 = w2 * local_density * (1.f + u8/c_sq + (u8*u8)/(2.f*c_sq*c_sq) - u_sq/(2.f*c_sq));

      tmp_cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)] = p0 + params.omega * (d0 - p0);
      tmp_cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] = p1 + params.omega * (d1 - p1);
      tmp_cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)] = p2 + params.omega * (d2 - p2);
      tmp_cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] = p3 + params.omega * (d3 - p3);
      tmp_cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)] = p4 + params.omega * (d4 - p4);
      tmp_cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] = p5 + params.omega * (d5 - p5);
      tmp_cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] = p6 + params.omega * (d6 - p6);
      tmp_cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] = p7 + params.omega * (d7 - p7);
      tmp_cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] = p8 + params.omega * (d8 - p8);
    }
  }

  return EXIT_SUCCESS;
}

/* ========================= metrics ========================= */

float av_velocity(const t_param params, float* cells, int* obstacles)
{
  int tot_cells = 0;
  float tot_u = 0.f;

  #pragma omp parallel for reduction(+:tot_u, tot_cells) schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (!obstacles[ii + jj * params.nx])
      {
        float local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)];
        }

        const float u_x =
            (cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] +
             cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] +
             cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] -
             (cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] +
              cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] +
              cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)])) / local_density;

        const float u_y =
            (cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)] +
             cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] +
             cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] -
             (cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)] +
              cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] +
              cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)])) / local_density;

        tot_u += sqrtf(u_x*u_x + u_y*u_y);
        tot_cells++;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

float calc_reynolds(const t_param params, float* cells, int* obstacles)
{
  const float viscosity = (1.f / 6.f) * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells)
{
  float total = 0.f;
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)];
      }
    }
  }
  return total;
}

/* ========================= IO & init ========================= */

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char message[1024];
  FILE* fp;
  int xx, yy, blocked;
  int retval;

  fp = fopen(paramfile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  retval = fscanf(fp, "%d\n", &(params->nx));            if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->ny));            if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->maxIters));      if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->density));       if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->accel));         if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->omega));         if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  fclose(fp);

  *cells_ptr = (float*)malloc(sizeof(float) * params->ny * params->nx * NSPEEDS);
  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  *tmp_cells_ptr = (float*)malloc(sizeof(float) * params->ny * params->nx * NSPEEDS);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  *obstacles_ptr = (int*)malloc(sizeof(int) * params->ny * params->nx);
  if (*obstacles_ptr == NULL) die("cannot allocate memory for obstacles", __LINE__, __FILE__);

  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density / 9.f;
  const float w2 = params->density / 36.f;

  #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*cells_ptr)[SPEED_IDX(ii, jj, 0, params->nx, params->ny)] = w0;
      (*cells_ptr)[SPEED_IDX(ii, jj, 1, params->nx, params->ny)] = w1;
      (*cells_ptr)[SPEED_IDX(ii, jj, 2, params->nx, params->ny)] = w1;
      (*cells_ptr)[SPEED_IDX(ii, jj, 3, params->nx, params->ny)] = w1;
      (*cells_ptr)[SPEED_IDX(ii, jj, 4, params->nx, params->ny)] = w1;
      (*cells_ptr)[SPEED_IDX(ii, jj, 5, params->nx, params->ny)] = w2;
      (*cells_ptr)[SPEED_IDX(ii, jj, 6, params->nx, params->ny)] = w2;
      (*cells_ptr)[SPEED_IDX(ii, jj, 7, params->nx, params->ny)] = w2;
      (*cells_ptr)[SPEED_IDX(ii, jj, 8, params->nx, params->ny)] = w2;
    }
  }

  #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj * params->nx] = 0;
    }
  }

  fp = fopen(obstaclefile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    (*obstacles_ptr)[xx + yy * params->nx] = blocked;
  }

  fclose(fp);

  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
  if (*av_vels_ptr == NULL) die("cannot allocate memory for av_vels", __LINE__, __FILE__);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  (void)params;
  free(*cells_ptr);     *cells_ptr = NULL;
  free(*tmp_cells_ptr); *tmp_cells_ptr = NULL;
  free(*obstacles_ptr); *obstacles_ptr = NULL;
  free(*av_vels_ptr);   *av_vels_ptr = NULL;
  return EXIT_SUCCESS;
}

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels)
{
  FILE* fp;
  const float c_sq = 1.f / 3.f;

  fp = fopen(FINALSTATEFILE, "w");
  if (fp == NULL) die("could not open file output file", __LINE__, __FILE__);

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      float local_density, pressure, u_x, u_y, u;

      if (obstacles[ii + jj * params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      else
      {
        local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)];
        }

        u_x =
          (cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] +
           cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] +
           cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] -
           (cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] +
            cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] +
            cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)])) / local_density;

        u_y =
          (cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)] +
           cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] +
           cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] -
           (cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)] +
            cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] +
            cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)])) / local_density;

        u = sqrtf(u_x*u_x + u_y*u_y);
        pressure = local_density * c_sq;
      }

      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n",
              ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }
  fclose(fp);

  fp = fopen(AVVELSFILE, "w");
  if (fp == NULL) die("could not open file output file", __LINE__, __FILE__);

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }
  fclose(fp);

  return EXIT_SUCCESS;
}

/* ========================= utils ========================= */

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}