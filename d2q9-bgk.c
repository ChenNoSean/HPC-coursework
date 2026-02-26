/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
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
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
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

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles);
int accelerate_flow(const t_param params, float* restrict cells, int* restrict obstacles);
int propagate(const t_param params, float* restrict cells, float* restrict tmp_cells);
int rebound(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles);
int collision(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles);
int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);

/* compute average velocity */
float av_velocity(const t_param params, float* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char* paramfile = NULL;    /* name of the input parameter file */
  char* obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  float* cells     = NULL;    /* grid containing fluid densities */
  float* tmp_cells = NULL;    /* scratch space */
  int* obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep(params, cells, tmp_cells, obstacles);
    
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
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
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

int timestep(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles)
{
  accelerate_flow(params, cells, obstacles);
  
  collision(params, cells, tmp_cells, obstacles);
  
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, float* restrict cells, int* restrict obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  #pragma omp parallel for simd default(none) shared(cells, obstacles, jj, w1, w2) schedule(static)
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] - w1) > 0.f
        && (cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] - w2) > 0.f
        && (cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] += w1;
      cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] += w2;
      cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] += w2;
      /* decrease 'west-side' densities */
      cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] -= w1;
      cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] -= w2;
      cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, float* restrict cells, float* restrict tmp_cells)
{
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    int y_n = (jj + 1) % params.ny;
    int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);

    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int x_e = (ii + 1) % params.nx;
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)] = cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)]; /* central cell, no movement */
      tmp_cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] = cells[SPEED_IDX(x_w, jj, 1, params.nx, params.ny)]; /* east */
      tmp_cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)] = cells[SPEED_IDX(ii, y_s, 2, params.nx, params.ny)]; /* north */
      tmp_cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] = cells[SPEED_IDX(x_e, jj, 3, params.nx, params.ny)]; /* west */
      tmp_cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)] = cells[SPEED_IDX(ii, y_n, 4, params.nx, params.ny)]; /* south */
      tmp_cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] = cells[SPEED_IDX(x_w, y_s, 5, params.nx, params.ny)]; /* north-east */
      tmp_cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] = cells[SPEED_IDX(x_e, y_s, 6, params.nx, params.ny)]; /* north-west */
      tmp_cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] = cells[SPEED_IDX(x_e, y_n, 7, params.nx, params.ny)]; /* south-west */
      tmp_cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] = cells[SPEED_IDX(x_w, y_n, 8, params.nx, params.ny)]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

int rebound(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles)
{
  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)];
        cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] = tmp_cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)];
      }
    }
  }

  return EXIT_SUCCESS;
}

int collision(const t_param params, float* restrict cells, float* restrict tmp_cells, int* restrict obstacles)
{
  const float c_sq = 1.f / 3.f; 
  const float w0 = 4.f / 9.f;  
  const float w1 = 1.f / 9.f;  
  const float w2 = 1.f / 36.f; 

#define PROCESS_CELL(ii, x_e, x_w) \
  do { \
    float p0 = cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)]; \
    float p1 = cells[SPEED_IDX(x_w, jj, 1, params.nx, params.ny)]; \
    float p2 = cells[SPEED_IDX(ii, y_s, 2, params.nx, params.ny)]; \
    float p3 = cells[SPEED_IDX(x_e, jj, 3, params.nx, params.ny)]; \
    float p4 = cells[SPEED_IDX(ii, y_n, 4, params.nx, params.ny)]; \
    float p5 = cells[SPEED_IDX(x_w, y_s, 5, params.nx, params.ny)]; \
    float p6 = cells[SPEED_IDX(x_e, y_s, 6, params.nx, params.ny)]; \
    float p7 = cells[SPEED_IDX(x_e, y_n, 7, params.nx, params.ny)]; \
    float p8 = cells[SPEED_IDX(x_w, y_n, 8, params.nx, params.ny)]; \
    float local_density = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8; \
    float u_x = (p1 + p5 + p8 - (p3 + p6 + p7)) / local_density; \
    float u_y = (p2 + p5 + p6 - (p4 + p7 + p8)) / local_density; \
    float u_sq = u_x * u_x + u_y * u_y; \
    float d_equ_0 = w0 * local_density * (1.f - u_sq / (2.f * c_sq)); \
    float d_equ_1 = w1 * local_density * (1.f + u_x / c_sq + (u_x * u_x) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq)); \
    float d_equ_2 = w1 * local_density * (1.f + u_y / c_sq + (u_y * u_y) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq)); \
    float d_equ_3 = w1 * local_density * (1.f - u_x / c_sq + (u_x * u_x) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq)); \
    float d_equ_4 = w1 * local_density * (1.f - u_y / c_sq + (u_y * u_y) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq)); \
    float u_5 =  u_x + u_y; \
    float d_equ_5 = w2 * local_density * (1.f + u_5 / c_sq + (u_5 * u_5) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq)); \
    float u_6 = -u_x + u_y; \
    float d_equ_6 = w2 * local_density * (1.f + u_6 / c_sq + (u_6 * u_6) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq)); \
    float u_7 = -u_x - u_y; \
    float d_equ_7 = w2 * local_density * (1.f + u_7 / c_sq + (u_7 * u_7) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq)); \
    float u_8 =  u_x - u_y; \
    float d_equ_8 = w2 * local_density * (1.f + u_8 / c_sq + (u_8 * u_8) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq)); \
    float c0 = p0 + params.omega * (d_equ_0 - p0); \
    float c1 = p1 + params.omega * (d_equ_1 - p1); \
    float c2 = p2 + params.omega * (d_equ_2 - p2); \
    float c3 = p3 + params.omega * (d_equ_3 - p3); \
    float c4 = p4 + params.omega * (d_equ_4 - p4); \
    float c5 = p5 + params.omega * (d_equ_5 - p5); \
    float c6 = p6 + params.omega * (d_equ_6 - p6); \
    float c7 = p7 + params.omega * (d_equ_7 - p7); \
    float c8 = p8 + params.omega * (d_equ_8 - p8); \
    int obs = obstacles[jj*params.nx + ii]; \
    tmp_cells[SPEED_IDX(ii, jj, 0, params.nx, params.ny)] = obs ? p0 : c0; \
    tmp_cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)] = obs ? p3 : c1; \
    tmp_cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)] = obs ? p4 : c2; \
    tmp_cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)] = obs ? p1 : c3; \
    tmp_cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)] = obs ? p2 : c4; \
    tmp_cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)] = obs ? p7 : c5; \
    tmp_cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)] = obs ? p8 : c6; \
    tmp_cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)] = obs ? p5 : c7; \
    tmp_cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)] = obs ? p6 : c8; \
  } while(0)

  #pragma omp parallel for default(none) shared(cells, tmp_cells, obstacles) schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    int y_n = (jj + 1) % params.ny;
    int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
    {
      int ii = 0;
      int x_e = 1;
      int x_w = params.nx - 1;
      PROCESS_CELL(ii, x_e, x_w);
    }

    #pragma omp simd
    for (int ii = 1; ii < params.nx - 1; ii++)
    {
      int x_e = ii + 1;
      int x_w = ii - 1;
      PROCESS_CELL(ii, x_e, x_w);
    }

    {
      int ii = params.nx - 1;
      int x_e = 0;
      int x_w = ii - 1;
      PROCESS_CELL(ii, x_e, x_w);
    }
  }

#undef PROCESS_CELL

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, float* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  #pragma omp parallel for default(none) shared(cells, obstacles) reduction(+:tot_u, tot_cells) schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)];
        }

        /* x-component of velocity */
        float u_x = (cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)]
                      + cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)]
                      + cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)]
                      - (cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)]
                         + cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)]
                         + cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)]
                      + cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)]
                      + cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)]
                      - (cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)]
                         + cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)]
                         + cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE* fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->ny));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->maxIters));
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->accel));
  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->omega));
  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /* main grid */
  *cells_ptr = (float*)malloc(sizeof(float) * params->ny * params->nx * NSPEEDS);
  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (float*)malloc(sizeof(float) * params->ny * params->nx * NSPEEDS);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  #pragma omp parallel for default(none) shared(cells_ptr, params, w0, w1, w2) schedule(static)
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[SPEED_IDX(ii, jj, 0, params->nx, params->ny)] = w0;
      /* axis directions */
      (*cells_ptr)[SPEED_IDX(ii, jj, 1, params->nx, params->ny)] = w1;
      (*cells_ptr)[SPEED_IDX(ii, jj, 2, params->nx, params->ny)] = w1;
      (*cells_ptr)[SPEED_IDX(ii, jj, 3, params->nx, params->ny)] = w1;
      (*cells_ptr)[SPEED_IDX(ii, jj, 4, params->nx, params->ny)] = w1;
      /* diagonals */
      (*cells_ptr)[SPEED_IDX(ii, jj, 5, params->nx, params->ny)] = w2;
      (*cells_ptr)[SPEED_IDX(ii, jj, 6, params->nx, params->ny)] = w2;
      (*cells_ptr)[SPEED_IDX(ii, jj, 7, params->nx, params->ny)] = w2;
      (*cells_ptr)[SPEED_IDX(ii, jj, 8, params->nx, params->ny)] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  #pragma omp parallel for default(none) shared(obstacles_ptr, params) schedule(static)
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /* allocate space to hold a record of the avarage velocities computed at each timestep */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /* free up allocated memory */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells)
{
  float total = 0.f;  /* accumulator */

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

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[SPEED_IDX(ii, jj, kk, params.nx, params.ny)];
        }

        /* compute x velocity component */
        u_x = (cells[SPEED_IDX(ii, jj, 1, params.nx, params.ny)]
               + cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)]
               + cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)]
               - (cells[SPEED_IDX(ii, jj, 3, params.nx, params.ny)]
                  + cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)]
                  + cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[SPEED_IDX(ii, jj, 2, params.nx, params.ny)]
               + cells[SPEED_IDX(ii, jj, 5, params.nx, params.ny)]
               + cells[SPEED_IDX(ii, jj, 6, params.nx, params.ny)]
               - (cells[SPEED_IDX(ii, jj, 4, params.nx, params.ny)]
                  + cells[SPEED_IDX(ii, jj, 7, params.nx, params.ny)]
                  + cells[SPEED_IDX(ii, jj, 8, params.nx, params.ny)]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

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