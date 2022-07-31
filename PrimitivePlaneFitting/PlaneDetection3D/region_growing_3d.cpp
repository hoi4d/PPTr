// STL includes.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
// CGAL includes.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/property_map.h>
#include <CGAL/Random.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing_on_point_set.h>
#include <CGAL/Timer.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>
#include <Eigen/Core>
#include <Eigen/Dense>

/***** Global Variables *****/
char dir[100] = "/home";
int const MAX_STR_LEN = 200;
bool VERBOSE = false;
// Type declarations.
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT = typename Kernel::FT;
using Point_3 = typename Kernel::Point_3;
using Vector_3 = typename Kernel::Vector_3;
using Input_range = CGAL::Point_set_3<Point_3>;
using Point_map = typename Input_range::Point_map;
using Normal_map = typename Input_range::Vector_map;
using Neighbor_query =
    CGAL::Shape_detection::Point_set::K_neighbor_query<Kernel, Input_range,
                                                       Point_map>;
using Region_type =
    CGAL::Shape_detection::Point_set::Least_squares_plane_fit_region<
        Kernel, Input_range, Point_map, Normal_map>;
using Region_growing =
    CGAL::Shape_detection::Region_growing<Input_range, Neighbor_query,
                                          Region_type>;
using Indices = std::vector<std::size_t>;
using Output_range = CGAL::Point_set_3<Point_3>;
using Points_3 = std::vector<Point_3>;

typedef std::pair<Point_3, Vector_3> PointVectorPair;
typedef std::vector<PointVectorPair> PointList;

// Concurrency
typedef CGAL::Parallel_if_available_tag Concurrency_tag;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixD;
typedef Eigen::Matrix<FT, 3, 1> Vector3;

struct Pointcloud
{
  MatrixD P, N, C;
};

void ComputePointNormals(
    Pointcloud &pc,                        // input points + output normals
    unsigned int nb_neighbors_pca_normals) // number of neighbors
{
  PointList points(pc.P.rows());
  for (int i = 0; i < points.size(); ++i)
  {
    auto p = pc.P.row(i);
    points[i].first = Point_3(p[0], p[1], p[2]);
  }
  CGAL::Timer task_timer;
  task_timer.start();

  // Estimates normals direction.
  // Note: pca_estimate_normals() requires an iterator over points
  // as well as property maps to access each point's position and normal.
  CGAL::pca_estimate_normals<Concurrency_tag>(
      points, nb_neighbors_pca_normals,
      CGAL::parameters::point_map(
          CGAL::First_of_pair_property_map<PointVectorPair>())
          .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));

  std::size_t memory = CGAL::Memory_sizer().virtual_size();
  pc.N.conservativeResize(points.size(), 3);
  for (int i = 0; i < points.size(); ++i)
  {
    pc.P.row(i) =
        Vector3(points[i].first.x(), points[i].first.y(), points[i].first.z());
    pc.N.row(i) = Vector3(points[i].second.x(), points[i].second.y(),
                          points[i].second.z());
    if (pc.N.row(i).dot(Vector3(1, 1, 1)) < 0)
      pc.N.row(i) = -pc.N.row(i);
  }
}

// Define an insert iterator.
int PlaneDetectRegion(Pointcloud &pc,
                      std::vector<std::pair<Vector3, FT>> &plane_parameters,
                      std::vector<int> &new_instances,
                      double dist_thres, double angle_thres, int min_points,
                      int num_neigbhors)
{
  const bool with_normal_map = true;
  Input_range input_range(with_normal_map);
  // add points
  for (int i = 0; i < pc.P.rows(); ++i)
  {
    input_range.insert(Kernel::Point_3(pc.P(i, 0), pc.P(i, 1), pc.P(i, 2)));
  }
  auto it = input_range.begin();
  // add normals
  for (int i = 0; i < pc.N.rows(); ++i)
  {
    input_range.normal(*(it++)) =
        Kernel::Vector_3(pc.N(i, 0), pc.N(i, 1), pc.N(i, 2));
  }
  // Default parameter values for the data file point_set_3.xyz.
  const std::size_t k = num_neigbhors;
  const FT max_distance_to_plane = dist_thres;
  const FT max_accepted_angle = angle_thres;
  const std::size_t min_region_size = min_points;

  // Create instances of the classes Neighbor_query and Region_type.
  Neighbor_query neighbor_query(input_range, k, input_range.point_map());
  Region_type region_type(input_range, max_distance_to_plane,
                          max_accepted_angle, min_region_size,
                          input_range.point_map(), input_range.normal_map());
  // Create an instance of the region growing class.
  Region_growing region_growing(input_range, neighbor_query, region_type);
  // Run the algorithm.
  Output_range output_range;
  std::size_t number_of_regions = 0;

  std::vector<std::vector<std::size_t>> regions;
  region_growing.detect(std::back_inserter(regions));
  plane_parameters.resize(regions.size());
  new_instances.resize(pc.P.rows(), -1);

  for (int i = 0; i < regions.size(); ++i)
  {
    Vector3 c(0, 0, 0);
    for (auto &idx : regions[i])
    {
      new_instances[idx] = i;
      c += pc.P.row(idx);
    }
    c /= (double)regions[i].size();

    MatrixD C = MatrixD::Zero(3, 3);
    for (auto &idx : regions[i])
    {
      Vector3 diff = pc.P.row(idx);
      diff -= c;
      C += diff * diff.transpose();
    }
    Eigen::JacobiSVD<MatrixD> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Vector3 N = svd.matrixU().col(2);
    plane_parameters[i] = std::make_pair(N, -N.dot(c));
  }
  return regions.size();
}

/* Show all files under dir_name , do not show directories ! */
void readAllFiles(const char *dir_name, std::vector<std::string> &filenames)
{
  // check the parameter !
  if (NULL == dir_name)
  {
    std::cout << " dir_name is null !" << std::endl;
    return;
  }

  // check if dir_name is a valid dir
  struct stat s;
  lstat(dir_name, &s);
  if (!S_ISDIR(s.st_mode))
  {
    std::cout << "dir_name is not a valid directory !" << std::endl;
    return;
  }

  struct dirent *filename; // return value for readdir()
  DIR *dir;                // return value for opendir()
  dir = opendir(dir_name);
  if (NULL == dir)
  {
    std::cout << "Can not open dir " << dir_name << std::endl;
    return;
  }
  std::cout << "Successfully opened the dir..." << std::endl;

  /* read all the files in the dir ~ */
  while ((filename = readdir(dir)) != NULL)
  {
    // get rid of "." and ".."
    if (strcmp(filename->d_name, ".") == 0 ||
        strcmp(filename->d_name, "..") == 0)
      continue;
    filenames.push_back(std::string(filename->d_name));
  }
}

// Input Data Format Definition
// Point with normal, color and intensity
typedef std::array<unsigned char, 3> Color;
// type def of Point Normal Color Input(PNC_I)
typedef std::tuple<Point_3, Vector_3, Color, unsigned char> PNC_I; 
// type def of properties
typedef CGAL::Nth_of_tuple_property_map<0, PNC_I> Point_map_I; // I:input
typedef CGAL::Nth_of_tuple_property_map<1, PNC_I> Normal_map_I;
typedef CGAL::Nth_of_tuple_property_map<2, PNC_I> Color_map_I;
typedef CGAL::Nth_of_tuple_property_map<3, PNC_I> Label_map_I;
// Ouput Data Format Definition
typedef std::array<float, 4> Primitive_Para;
// type definition of PRimitive Output(PR_O)
typedef std::tuple<Point_3, Vector_3, Color, unsigned char, int, Primitive_Para> PR_O; 
typedef CGAL::Nth_of_tuple_property_map<0, PR_O> Point_map_O;  // O:output
typedef CGAL::Nth_of_tuple_property_map<1, PR_O> Normal_map_O;
typedef CGAL::Nth_of_tuple_property_map<2, PR_O> Color_map_O;
typedef CGAL::Nth_of_tuple_property_map<3, PR_O> Label_map_O;
typedef CGAL::Nth_of_tuple_property_map<4, PR_O> PrimitiveIns_map_O;  // primitive instance
typedef CGAL::Nth_of_tuple_property_map<5, PR_O> PrimitivePara_map_O; // primitive parameter

/* check file existence */
inline bool isExists(const std::string &name)
{
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}
/* save customed ply format file using ascii */
void savePLY(const std::string &output_path, std::vector<PR_O> &data)
{
  if (isExists(output_path))
  {
    return;
  }
  std::ofstream output(output_path);
  // PLY file head
  output << "ply" << std::endl
         << "format ascii 1.0" << std::endl
         << "element vertex " << data.size() << std::endl
         << "property float x" << std::endl
         << "property float y" << std::endl
         << "property float z" << std::endl
         << "property float nx" << std::endl
         << "property float ny" << std::endl
         << "property float nz" << std::endl
         << "property int label" << std::endl
         << "property int primitive" << std::endl
         << "property float a" << std::endl
         << "property float b" << std::endl
         << "property float c" << std::endl
         << "property float d" << std::endl
         << "end_header" << std::endl;
  // body data
  for (size_t i = 0; i < data.size(); i++)
  {
    const float xyz[3] = {float((std::get<0>(data[i])).x()), float((std::get<0>(data[i])).y()), float((std::get<0>(data[i])).z())};
    const float nxyz[3] = {float((std::get<1>(data[i])).x()), float((std::get<1>(data[i])).y()), float((std::get<1>(data[i])).z())};
    const float abcd[4] = {(std::get<5>(data[i]))[0], (std::get<5>(data[i]))[1], (std::get<5>(data[i]))[2], (std::get<5>(data[i]))[3]};
    output << xyz[0] << " " << xyz[1] << " " << xyz[2] << " "
           << nxyz[0] << " " << nxyz[1] << " " << nxyz[2] << " "
           << int(std::get<3>(data[i])) << " " << int(std::get<4>(data[i])) << " "
           << abcd[0] << " " << abcd[1] << " " << abcd[2] << " " << abcd[3] << std::endl;
  }
  output.close();
  if (VERBOSE)
  {
    std::cout << "Save Done" << std::endl;
  }
  return;
}

int main(int argc, char **argv)
{
  // default path
  std::string input_path = "./Data/Synthia4D";
  std::string output_path = "./Data/output";
  std::string visual_path = "./Data/visual";
  // path setting
  for (int i = 0; i < argc; ++i)
  {
    if (strcmp(argv[i], "-i") == 0)
    {
      input_path = argv[i + 1];
    }
    if (strcmp(argv[i], "-o") == 0)
    {
      output_path = argv[i + 1];
    }
    if (strcmp(argv[i], "-v") == 0)
    {
      visual_path = argv[i + 1];
    }
    if (strcmp(argv[i], "-h") == 0)
    {
      printf("./test_plane_depth -c color_path -i input_path -o output_path\n");
    }
  }

  // hyper parameters & variables
  // an example for Synthia 4D, varies by dataset
  const int kNeighbors_nor = 10; // neighbor number
  const int kNeighbors_rg = 10;
  const double kDistThres = 1e-1; // distance tolerance
  const double kAngleThres = 50;  // plane normal direction tolerance
  const int kMinPoints = 30;      // minimum points in one plane

  bool visual = true; // output visualization file
  std::vector<std::string> filenames;
  // get filename list
  readAllFiles(input_path.c_str(), filenames);
  if (VERBOSE)
  {
    std::cout << "=> File list length: " << filenames.size() << std::endl;
  }
  for (size_t i = 0; i < filenames.size(); i++){
    if (filenames[i].find(".txt") != std::string::npos)
    {
      std::cout << "=> Skip .txt file: " << filenames[i] << std::endl;
      continue;
    }
    std::string pc_path = input_path + "/" + filenames[i];  // point cloud input
    std::string pr_path = output_path + "/" + filenames[i]; // primitive fitting output
    bool isexists = access(pr_path.c_str(), F_OK) != -1;
    if (isexists)
    {
      if (VERBOSE)
      {
        std::cout << "=> Skip path " << pr_path << std::endl;
      }
      continue;
    }
    if (VERBOSE)
    {
      std::cout << "=> Checking input file" << pc_path << std::endl;
      std::cout << "=> Checking output file" << pr_path << std::endl;
    }
    std::cout << "=> Processing " << filenames[i] << std::endl;
    std::ifstream input(pc_path, std::ios_base::binary);
    // input (points with propertes)
    std::vector<PNC_I> PNCIs; // I: input
    // output (primitive) variable
    std::vector<PR_O> PROs; // O: output
    std::vector<Vector3> points;
    // start reading
    if (!input ||
        !CGAL::read_ply_points_with_properties(input, std::back_inserter(PNCIs),
                                                CGAL::make_ply_point_reader(Point_map_I()),
                                                std::make_pair(Label_map_I(), CGAL::PLY_property<unsigned char>("label")),
                                                std::make_tuple(Color_map_I(),
                                                                CGAL::Construct_array(),
                                                                CGAL::PLY_property<unsigned char>("red"),
                                                                CGAL::PLY_property<unsigned char>("green"),
                                                                CGAL::PLY_property<unsigned char>("blue")),
                                                CGAL::make_ply_normal_reader(Normal_map_I())))
    {
      std::cerr << "Error: cannot read file " << pc_path << std::endl;
      return EXIT_FAILURE;
    }
    input.close();

    if (VERBOSE)
    {
      // debug info
      std::cout << "Read .ply success! " << std::endl;
      std::cout << "Total Points: " << PNCIs.size() << std::endl;
      std::cout << "Element Example: ("
                << (std::get<0>(PNCIs[250])).x() << ", "
                << (std::get<0>(PNCIs[250])).y() << ", "
                << (std::get<0>(PNCIs[250])).z() << "), ("
                << (std::get<1>(PNCIs[250])).x() << ", "
                << (std::get<1>(PNCIs[250])).y() << ", "
                << (std::get<1>(PNCIs[250])).z() << "), ("
                << (unsigned int)(std::get<2>(PNCIs[250]))[0] << ", "
                << (unsigned int)(std::get<2>(PNCIs[250]))[1] << ", "
                << (unsigned int)(std::get<2>(PNCIs[250]))[2] << "), ("
                << (unsigned int)(std::get<3>(PNCIs[250])) << ")" << std::endl;
    }

    // reformat points
    for (size_t i = 0; i < PNCIs.size(); ++i)
    {
      points.push_back(Vector3((std::get<0>(PNCIs[i])).x(),
                                (std::get<0>(PNCIs[i])).y(),
                                (std::get<0>(PNCIs[i])).z()));
    }
    // compute normals and detect planes
    Pointcloud pc;
    pc.P.resize(points.size(), 3);
    memcpy(pc.P.data(), points.data(), sizeof(Vector3) * points.size());
    try
    {
      ComputePointNormals(pc, kNeighbors_nor);
    }
    catch (std::exception)
    {
      std::cout << ">>-----------------------------------------<<" << std::endl;
      std::cout << "(ERROR!) Can't compute normals: " << filenames[i] << std::endl;
      std::cout << ">>-----------------------------------------<<" << std::endl;
      continue;
    }

    std::vector<std::pair<Vector3, double>> plane_parameters;
    std::vector<int> plane_instances;
    int num_inst = 0;
    try
    {
      num_inst = PlaneDetectRegion(pc, plane_parameters, plane_instances,
                                    kDistThres, kAngleThres, kMinPoints, kNeighbors_rg);
    }
    catch (std::exception)
    {
      std::cout << ">>-----------------------------------------<<" << std::endl;
      std::cout << "(ERROR!) Can't detect planes: " << filenames[i] << std::endl;
      std::cout << ">>-----------------------------------------<<" << std::endl;
      continue;
    }
    // visualization
    if (visual == true)
    {
      std::vector<Vector3> colors(num_inst);
      for (auto &c : colors)
      {
        c = Vector3((rand() + 10) % 9999 / 9999., (rand() + 90) % 9999 / 9999., (rand() + 18) % 9999 / 9999.);
      }
      std::string visual_file = visual_path +"/visual_"+ filenames[i].substr(0, filenames[i].rfind(".")) + ".xyzrgb";
      std::ofstream os(visual_file);
      for (int i = 0; i < points.size(); ++i)
      {
        auto p = points[i];
        auto c = Vector3(0, 0, 0);
        if (plane_instances[i] >= 0)
          c = colors[plane_instances[i]];
        os << p[0] << " " << p[1] << " " << p[2] << " "
            << c[0] << " " << c[1] << " " << c[2] << "\n";
      }
    }
    // reformat and save 
    for (size_t i = 0; i < PNCIs.size(); ++i)
    {
      auto plan_idx = plane_instances[i];
      auto Para = CGAL::make_array((float)plane_parameters[plan_idx].first[0],
                                    (float)plane_parameters[plan_idx].first[1],
                                    (float)plane_parameters[plan_idx].first[2],
                                    plane_parameters[plan_idx].second);

      PROs.push_back(std::make_tuple(std::get<0>(PNCIs[i]), Kernel::Vector_3(pc.N(i, 0), pc.N(i, 1), pc.N(i, 2)),
                                      std::get<2>(PNCIs[i]), std::get<3>(PNCIs[i]), plane_instances[i], Para));
    }
    if (VERBOSE)
    {
      // DEBUG info
      std::cout << "=> Read .ply success! " << std::endl;
      std::cout << "=> Total Points: " << PNCIs.size() << std::endl;
      std::cout << "=> Example Format: (xyz, nxyz, rgb, label, instance, abcd)" << std::endl;
    }
    savePLY(pr_path, PROs);
    return 0;
  }
  return 0;
}
