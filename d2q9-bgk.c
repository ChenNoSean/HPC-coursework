/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

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

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

int av_velocity_local(const t_param params,
                      t_speed* local_cells,
                      int* local_obstacles,
                      int local_ny,
                      float* local_tot_u,
                      int* local_tot_cells);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

int exchange_halos(const t_param params,
                   t_speed* local_cells,
                   int* local_obstacles,
                   int local_ny,
                   int up,
                   int down);

int exchange_cell_halos(const t_param params,
                        t_speed* local_cells,
                        int local_ny,
                        int up,
                        int down);

int exchange_obstacle_halos(const t_param params,
                            int* local_obstacles,
                            int local_ny,
                            int up,
                            int down);

int timestep_local(const t_param params,
                   t_speed** local_cells_ptr,
                   t_speed** local_tmp_cells_ptr,
                   int* local_obstacles,
                   int local_ny,
                   int start_y,
                   int up,
                   int down,
                   float* local_tot_u,
                   int* local_tot_cells);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  int rank = 0;
  int size = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
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

  /* Work out the row range owned by this rank */
  int base_rows = params.ny / size;
  int extra_rows = params.ny % size;

  int local_ny;
  int start_y;

  int up = (rank == 0) ? size - 1 : rank - 1;
  int down = (rank == size - 1) ? 0 : rank + 1;
  //printf("Rank %d: up=%d down=%d\n", rank, up, down);

  if (rank < extra_rows)
  {
    local_ny = base_rows + 1;
    start_y = rank * local_ny;
  }
  else
  {
    local_ny = base_rows;
    start_y = extra_rows * (base_rows + 1)
            + (rank - extra_rows) * base_rows;
  }

  //int end_y = start_y + local_ny - 1;
  //printf("Rank %d/%d owns rows %d to %d (%d rows)\n",
  //        rank, size, start_y, end_y, local_ny);

  int local_nrows_with_halo = local_ny + 2;
  int local_size = local_nrows_with_halo * params.nx;

  t_speed* local_cells = malloc(sizeof(t_speed) * local_size);
  t_speed* local_tmp_cells = malloc(sizeof(t_speed) * local_size);
  int* local_obstacles = malloc(sizeof(int) * local_size);

  if (!local_cells || !local_tmp_cells || !local_obstacles)
  {
    die("cannot allocate local arrays", __LINE__, __FILE__);
  } 

  /* copy real rows */
  for (int jj = 0; jj < local_ny; jj++)
  {
    int global_jj = start_y + jj;
    int local_jj = jj + 1;

    for (int ii = 0; ii < params.nx; ii++)
    {
      int global_idx = ii + global_jj * params.nx;
      int local_idx = ii + local_jj * params.nx;

      local_cells[local_idx] = cells[global_idx];
      local_tmp_cells[local_idx] = tmp_cells[global_idx];
      local_obstacles[local_idx] = obstacles[global_idx];
    }
  }

  /* Obstacles are static, so exchange obstacle halo rows once after initialisation. */
  exchange_obstacle_halos(params, local_obstacles, local_ny, up, down);

  /* Free full global arrays after local copy to avoid OOM.
     Rank 0 keeps cells/obstacles for final MPI_Gatherv and output. */
  free(tmp_cells);
  tmp_cells = NULL;

  if (rank != 0)
  {
    free(cells);
    cells = NULL;

    free(obstacles);
    obstacles = NULL;
  }

  /* debug print */
  //printf("Rank %d local rows = %d (+2 halo)\n", rank, local_ny);

  

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    float local_tot_u = 0.f;
    float global_tot_u = 0.f;
    int local_tot_cells = 0;
    int global_tot_cells = 0;

    timestep_local(params,
              &local_cells,
              &local_tmp_cells,
              local_obstacles,
              local_ny,
              start_y,
              up,
              down,
              &local_tot_u,
              &local_tot_cells);

    /* Reduce fused local average-velocity contribution to rank 0. */
    double local_vals[2];
    double global_vals[2];

    local_vals[0] = (double)local_tot_u;
    local_vals[1] = (double)local_tot_cells;

    MPI_Reduce(local_vals,
               global_vals,
               2,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    if (rank == 0)
    {
      global_tot_u = (float)global_vals[0];
      global_tot_cells = (int)global_vals[1];
      av_vels[tt] = global_tot_u / (float)global_tot_cells;
    }
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

 

  /* Total/collate time stops here.*/
  int local_bytes = local_ny * params.nx * sizeof(t_speed);

  int* recvcounts = NULL;
  int* displs = NULL;

  if (rank == 0)
  {
    recvcounts = malloc(sizeof(int) * size);
    displs = malloc(sizeof(int) * size);

    if (!recvcounts || !displs)
    {
    die("cannot allocate recvcounts/displs", __LINE__, __FILE__);
    }

    for (int r = 0; r < size; r++)
    {
      int r_base_rows = params.ny / size;
      int r_extra_rows = params.ny % size;

      int r_local_ny;
      int r_start_y;

      if (r < r_extra_rows)
      {
        r_local_ny = r_base_rows + 1;
        r_start_y = r * r_local_ny;
      }
      else
      {
        r_local_ny = r_base_rows;
        r_start_y = r_extra_rows * (r_base_rows + 1)
                  + (r - r_extra_rows) * r_base_rows;
      }

      recvcounts[r] = r_local_ny * params.nx * sizeof(t_speed);
      displs[r] = r_start_y * params.nx * sizeof(t_speed);
    }
  }

  MPI_Gatherv(&local_cells[1 * params.nx],
              local_bytes,
              MPI_BYTE,
              cells,
              recvcounts,
              displs,
              MPI_BYTE,
              0,
              MPI_COMM_WORLD);

  if (rank == 0)
  {
    free(recvcounts);
    free(displs);
  }

  gettimeofday(&timstr, NULL); 
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  if (rank == 0)
  {
    printf("==done==\n");
    printf("MPI ranks:\t\t%d\n", size);
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, cells, obstacles, av_vels);
  } 

  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  MPI_Finalize();

  return EXIT_SUCCESS;
}

static inline void process_cell_fused(
    const int ii, const int row, const int row_s, const int row_n,
    const int x_w, const int x_e, const int idx,
    const float omega,
    t_speed* restrict cells,
    t_speed* restrict tmp_cells,
    const int* restrict obstacles,
    float* restrict tot_u,
    int* restrict tot_cells)
{
  const float w0 = 4.f / 9.f;
  const float w1 = 1.f / 9.f;
  const float w2 = 1.f / 36.f;
  const float c1 = 3.f;
  const float c2 = 4.5f;
  const float c3 = 1.5f;

  const float s0 = cells[idx].speeds[0];
  const float s1 = cells[x_w + row].speeds[1];
  const float s2 = cells[ii  + row_s].speeds[2];
  const float s3 = cells[x_e + row].speeds[3];
  const float s4 = cells[ii  + row_n].speeds[4];
  const float s5 = cells[x_w + row_s].speeds[5];
  const float s6 = cells[x_e + row_s].speeds[6];
  const float s7 = cells[x_e + row_n].speeds[7];
  const float s8 = cells[x_w + row_n].speeds[8];

  if (obstacles[idx])
  {
    tmp_cells[idx].speeds[0] = s0;
    tmp_cells[idx].speeds[1] = s3;
    tmp_cells[idx].speeds[2] = s4;
    tmp_cells[idx].speeds[3] = s1;
    tmp_cells[idx].speeds[4] = s2;
    tmp_cells[idx].speeds[5] = s7;
    tmp_cells[idx].speeds[6] = s8;
    tmp_cells[idx].speeds[7] = s5;
    tmp_cells[idx].speeds[8] = s6;
    return;
  }

  const float local_density = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
  const float u_x = (s1 + s5 + s8 - (s3 + s6 + s7)) / local_density;
  const float u_y = (s2 + s5 + s6 - (s4 + s7 + s8)) / local_density;
  const float u_sq = u_x * u_x + u_y * u_y;

  *tot_u += sqrtf(u_sq);
  ++(*tot_cells);

  const float u1 =  u_x;
  const float u2 =  u_y;
  const float u3 = -u_x;
  const float u4 = -u_y;
  const float u5 =  u_x + u_y;
  const float u6 = -u_x + u_y;
  const float u7 = -u_x - u_y;
  const float u8 =  u_x - u_y;

  const float feq0 = w0 * local_density * (1.f - c3 * u_sq);
  const float feq1 = w1 * local_density * (1.f + c1*u1 + c2*u1*u1 - c3*u_sq);
  const float feq2 = w1 * local_density * (1.f + c1*u2 + c2*u2*u2 - c3*u_sq);
  const float feq3 = w1 * local_density * (1.f + c1*u3 + c2*u3*u3 - c3*u_sq);
  const float feq4 = w1 * local_density * (1.f + c1*u4 + c2*u4*u4 - c3*u_sq);
  const float feq5 = w2 * local_density * (1.f + c1*u5 + c2*u5*u5 - c3*u_sq);
  const float feq6 = w2 * local_density * (1.f + c1*u6 + c2*u6*u6 - c3*u_sq);
  const float feq7 = w2 * local_density * (1.f + c1*u7 + c2*u7*u7 - c3*u_sq);
  const float feq8 = w2 * local_density * (1.f + c1*u8 + c2*u8*u8 - c3*u_sq);

  tmp_cells[idx].speeds[0] = s0 + omega * (feq0 - s0);
  tmp_cells[idx].speeds[1] = s1 + omega * (feq1 - s1);
  tmp_cells[idx].speeds[2] = s2 + omega * (feq2 - s2);
  tmp_cells[idx].speeds[3] = s3 + omega * (feq3 - s3);
  tmp_cells[idx].speeds[4] = s4 + omega * (feq4 - s4);
  tmp_cells[idx].speeds[5] = s5 + omega * (feq5 - s5);
  tmp_cells[idx].speeds[6] = s6 + omega * (feq6 - s6);
  tmp_cells[idx].speeds[7] = s7 + omega * (feq7 - s7);
  tmp_cells[idx].speeds[8] = s8 + omega * (feq8 - s8);
}

int exchange_halos(const t_param params,
                   t_speed* local_cells,
                   int* local_obstacles,
                   int local_ny,
                   int up,
                   int down)
{
  const int nx = params.nx;
  const int row_speeds = nx * sizeof(t_speed);
  const int row_obs = nx;

  /* local row layout:
     0              = top halo
     1              = first real row
     local_ny       = last real row
     local_ny + 1   = bottom halo
  */

  MPI_Sendrecv(&local_cells[1 * nx], row_speeds, MPI_BYTE, up, 0,
               &local_cells[(local_ny + 1) * nx], row_speeds, MPI_BYTE, down, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&local_cells[local_ny * nx], row_speeds, MPI_BYTE, down, 1,
               &local_cells[0 * nx], row_speeds, MPI_BYTE, up, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&local_obstacles[1 * nx], row_obs, MPI_INT, up, 2,
               &local_obstacles[(local_ny + 1) * nx], row_obs, MPI_INT, down, 2,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&local_obstacles[local_ny * nx], row_obs, MPI_INT, down, 3,
               &local_obstacles[0 * nx], row_obs, MPI_INT, up, 3,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return EXIT_SUCCESS;
}

int exchange_cell_halos(const t_param params,
                        t_speed* local_cells,
                        int local_ny,
                        int up,
                        int down)
{
  const int nx = params.nx;
  const int row_speeds = nx * sizeof(t_speed);

  MPI_Sendrecv(&local_cells[1 * nx], row_speeds, MPI_BYTE, up, 0,
               &local_cells[(local_ny + 1) * nx], row_speeds, MPI_BYTE, down, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&local_cells[local_ny * nx], row_speeds, MPI_BYTE, down, 1,
               &local_cells[0 * nx], row_speeds, MPI_BYTE, up, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return EXIT_SUCCESS;
}

int exchange_obstacle_halos(const t_param params,
                            int* local_obstacles,
                            int local_ny,
                            int up,
                            int down)
{
  const int nx = params.nx;
  const int row_obs = nx;

  MPI_Sendrecv(&local_obstacles[1 * nx], row_obs, MPI_INT, up, 2,
               &local_obstacles[(local_ny + 1) * nx], row_obs, MPI_INT, down, 2,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&local_obstacles[local_ny * nx], row_obs, MPI_INT, down, 3,
               &local_obstacles[0 * nx], row_obs, MPI_INT, up, 3,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  return EXIT_SUCCESS;
}

int timestep_local(const t_param params,
                   t_speed** local_cells_ptr,
                   t_speed** local_tmp_cells_ptr,
                   int* local_obstacles,
                   int local_ny,
                   int start_y,
                   int up,
                   int down,
                   float* local_tot_u,
                   int* local_tot_cells)
{
  t_speed* local_cells = *local_cells_ptr;
  t_speed* local_tmp_cells = *local_tmp_cells_ptr;

  const int nx = params.nx;
  const int global_ny = params.ny;
  const float omega = params.omega;

  const float w0 = 4.f / 9.f;
  const float w1 = 1.f / 9.f;
  const float w2 = 1.f / 36.f;

  const float accel_w1 = params.density * params.accel / 9.f;
  const float accel_w2 = params.density * params.accel / 36.f;

  const float c1 = 3.f;
  const float c2 = 4.5f;
  const float c3 = 1.5f;

  *local_tot_u = 0.f;
  *local_tot_cells = 0;

  /* Apply acceleration before propagation, only on the rank that owns global row ny-2 */
  const int accel_global_jj = global_ny - 2;

  if (accel_global_jj >= start_y && accel_global_jj < start_y + local_ny)
  {
    int accel_local_jj = (accel_global_jj - start_y) + 1;

    for (int ii = 0; ii < nx; ii++)
    {
      int idx = ii + accel_local_jj * nx;

      if (!local_obstacles[idx]
          && (local_cells[idx].speeds[3] - accel_w1) > 0.f
          && (local_cells[idx].speeds[6] - accel_w2) > 0.f
          && (local_cells[idx].speeds[7] - accel_w2) > 0.f)
          {
            local_cells[idx].speeds[1] += accel_w1;
            local_cells[idx].speeds[5] += accel_w2;
            local_cells[idx].speeds[8] += accel_w2;

            local_cells[idx].speeds[3] -= accel_w1;
            local_cells[idx].speeds[6] -= accel_w2;
            local_cells[idx].speeds[7] -= accel_w2;
          }
      }
  }


  exchange_cell_halos(params, local_cells, local_ny, up, down);

  for (int local_jj = 1; local_jj <= local_ny; local_jj++)
  {

    for (int ii = 0; ii < nx; ii++)
    {
      const int x_w = (ii == 0) ? nx - 1 : ii - 1;
      const int x_e = (ii == nx - 1) ? 0 : ii + 1;

      const int idx = ii + local_jj * nx;
      const int y_s = local_jj - 1;
      const int y_n = local_jj + 1;

      float s0 = local_cells[idx].speeds[0];
      float s1 = local_cells[x_w + local_jj * nx].speeds[1];
      float s2 = local_cells[ii  + y_s      * nx].speeds[2];
      float s3 = local_cells[x_e + local_jj * nx].speeds[3];
      float s4 = local_cells[ii  + y_n      * nx].speeds[4];
      float s5 = local_cells[x_w + y_s      * nx].speeds[5];
      float s6 = local_cells[x_e + y_s      * nx].speeds[6];
      float s7 = local_cells[x_e + y_n      * nx].speeds[7];
      float s8 = local_cells[x_w + y_n      * nx].speeds[8];

      if (local_obstacles[idx])
      {
        local_tmp_cells[idx].speeds[0] = s0;
        local_tmp_cells[idx].speeds[1] = s3;
        local_tmp_cells[idx].speeds[2] = s4;
        local_tmp_cells[idx].speeds[3] = s1;
        local_tmp_cells[idx].speeds[4] = s2;
        local_tmp_cells[idx].speeds[5] = s7;
        local_tmp_cells[idx].speeds[6] = s8;
        local_tmp_cells[idx].speeds[7] = s5;
        local_tmp_cells[idx].speeds[8] = s6;
        continue;
      }

      const float local_density = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;

      const float u_x = (s1 + s5 + s8 - (s3 + s6 + s7)) / local_density;
      const float u_y = (s2 + s5 + s6 - (s4 + s7 + s8)) / local_density;
      const float u_sq = u_x * u_x + u_y * u_y;

      *local_tot_u += sqrtf(u_sq);
      (*local_tot_cells)++;
      
      const float u1 =  u_x;
      const float u2 =  u_y;
      const float u3 = -u_x;
      const float u4 = -u_y;
      const float u5 =  u_x + u_y;
      const float u6 = -u_x + u_y;
      const float u7 = -u_x - u_y;
      const float u8 =  u_x - u_y;

      const float feq0 = w0 * local_density * (1.f - c3 * u_sq);
      const float feq1 = w1 * local_density * (1.f + c1*u1 + c2*u1*u1 - c3*u_sq);
      const float feq2 = w1 * local_density * (1.f + c1*u2 + c2*u2*u2 - c3*u_sq);
      const float feq3 = w1 * local_density * (1.f + c1*u3 + c2*u3*u3 - c3*u_sq);
      const float feq4 = w1 * local_density * (1.f + c1*u4 + c2*u4*u4 - c3*u_sq);
      const float feq5 = w2 * local_density * (1.f + c1*u5 + c2*u5*u5 - c3*u_sq);
      const float feq6 = w2 * local_density * (1.f + c1*u6 + c2*u6*u6 - c3*u_sq);
      const float feq7 = w2 * local_density * (1.f + c1*u7 + c2*u7*u7 - c3*u_sq);
      const float feq8 = w2 * local_density * (1.f + c1*u8 + c2*u8*u8 - c3*u_sq);

      local_tmp_cells[idx].speeds[0] = s0 + omega * (feq0 - s0);
      local_tmp_cells[idx].speeds[1] = s1 + omega * (feq1 - s1);
      local_tmp_cells[idx].speeds[2] = s2 + omega * (feq2 - s2);
      local_tmp_cells[idx].speeds[3] = s3 + omega * (feq3 - s3);
      local_tmp_cells[idx].speeds[4] = s4 + omega * (feq4 - s4);
      local_tmp_cells[idx].speeds[5] = s5 + omega * (feq5 - s5);
      local_tmp_cells[idx].speeds[6] = s6 + omega * (feq6 - s6);
      local_tmp_cells[idx].speeds[7] = s7 + omega * (feq7 - s7);
      local_tmp_cells[idx].speeds[8] = s8 + omega * (feq8 - s8);
    }
  }

  *local_cells_ptr = local_tmp_cells;
  *local_tmp_cells_ptr = local_cells;

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles)
{
  t_speed* cells = *cells_ptr;
  t_speed* tmp_cells = *tmp_cells_ptr;

  accelerate_flow(params, cells, obstacles);

  const int nx = params.nx;
  const int ny = params.ny;
  const float omega = params.omega;

  float tot_u = 0.f;
  int tot_cells = 0;

  /* Bottom boundary row: preserve periodic wrap-around. */
  {
    const int row = 0;
    const int row_s = (ny - 1) * nx;
    const int row_n = nx;
    for (int ii = 0; ii < nx; ii++)
    {
      const int x_w = (ii == 0) ? (nx - 1) : (ii - 1);
      const int x_e = (ii == nx - 1) ? 0 : (ii + 1);
      const int idx = ii;
      process_cell_fused(ii, row, row_s, row_n, x_w, x_e, idx,
                         omega, cells, tmp_cells, obstacles, &tot_u, &tot_cells);
    }
  }

  /* Interior rows. Only left/right columns need x wrap-around. */
  for (int jj = 1; jj < ny - 1; jj++)
  {
    const int row = jj * nx;
    const int row_s = row - nx;
    const int row_n = row + nx;

    /* Left boundary column. */
    process_cell_fused(0, row, row_s, row_n, nx - 1, 1, row,
                       omega, cells, tmp_cells, obstacles, &tot_u, &tot_cells);

    /* Fast interior: no %, no ternary neighbour-index calculations. */
    for (int ii = 1; ii < nx - 1; ii++)
    {
      const int idx = ii + row;
      process_cell_fused(ii, row, row_s, row_n, ii - 1, ii + 1, idx,
                         omega, cells, tmp_cells, obstacles, &tot_u, &tot_cells);
    }

    /* Right boundary column. */
    process_cell_fused(nx - 1, row, row_s, row_n, nx - 2, 0, row + nx - 1,
                       omega, cells, tmp_cells, obstacles, &tot_u, &tot_cells);
  }

  /* Top boundary row: preserve periodic wrap-around. */
  {
    const int row = (ny - 1) * nx;
    const int row_s = row - nx;
    const int row_n = 0;
    for (int ii = 0; ii < nx; ii++)
    {
      const int x_w = (ii == 0) ? (nx - 1) : (ii - 1);
      const int x_e = (ii == nx - 1) ? 0 : (ii + 1);
      const int idx = ii + row;
      process_cell_fused(ii, row, row_s, row_n, x_w, x_e, idx,
                         omega, cells, tmp_cells, obstacles, &tot_u, &tot_cells);
    }
  }

  *cells_ptr = tmp_cells;
  *tmp_cells_ptr = cells;

  return tot_u / (float)tot_cells;
}

int av_velocity_local(const t_param params,
                      t_speed* local_cells,
                      int* local_obstacles,
                      int local_ny,
                      float* local_tot_u,
                      int* local_tot_cells)
{
  *local_tot_u = 0.f;
  *local_tot_cells = 0;

  const int nx = params.nx;

  for (int jj = 1; jj <= local_ny; jj++)
  {
    for (int ii = 0; ii < nx; ii++)
    {
      int idx = ii + jj * nx;

      if (!local_obstacles[idx])
      {
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += local_cells[idx].speeds[kk];
        }

        float u_x = (local_cells[idx].speeds[1]
                    + local_cells[idx].speeds[5]
                    + local_cells[idx].speeds[8]
                    - (local_cells[idx].speeds[3]
                    + local_cells[idx].speeds[6]
                    + local_cells[idx].speeds[7]))
                    / local_density;

        float u_y = (local_cells[idx].speeds[2]
                    + local_cells[idx].speeds[5]
                    + local_cells[idx].speeds[6]
                    - (local_cells[idx].speeds[4]
                    + local_cells[idx].speeds[7]
                    + local_cells[idx].speeds[8]))
                    / local_density;

        *local_tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        (*local_tot_cells)++;
      }
    }
  }

  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
        && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
        && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[ii + jj*params.nx].speeds[1] += w1;
      cells[ii + jj*params.nx].speeds[5] += w2;
      cells[ii + jj*params.nx].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii + jj*params.nx].speeds[3] -= w1;
      cells[ii + jj*params.nx].speeds[6] -= w2;
      cells[ii + jj*params.nx].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
        cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
        cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
        cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
      }
    }
  }

  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
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
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
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
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
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

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
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

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
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


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
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
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
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
