#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/statistical_outlier_removal.h>


int main() {
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    if (pcl::io::loadPLYFile("/home/kong-vb/robo_code/RAStest/voltest - Cloud.ply", *cloud) == -1) {
        PCL_ERROR("Couldn't read file voltest - Cloud.ply \n");
    }

    // 创建新的XYZ点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    xyz_cloud->width = cloud->width;
    xyz_cloud->height = cloud->height;
    xyz_cloud->points.resize(xyz_cloud->width * xyz_cloud->height);

    // 将XYZRGBA点云转换为仅包含XYZ坐标的点云
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        // 将XYZ坐标赋给新的XYZ点云
        xyz_cloud->points[i].x = cloud->points[i].x;
        xyz_cloud->points[i].y = cloud->points[i].y;
        xyz_cloud->points[i].z = cloud->points[i].z;
    }

    // 计算法线
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(xyz_cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setKSearch(10); // 设置最近邻搜索的数量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

#if 1
   std::vector<int> inliers;
  pcl::SampleConsensusModelCylinder<pcl::PointXYZ, pcl::Normal>::Ptr model (new pcl::SampleConsensusModelCylinder<pcl::PointXYZ, pcl::Normal> (xyz_cloud));
  model->setInputNormals(normals);
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model);
  ransac.setDistanceThreshold (0.01); // 设置距离阈值
  ransac.computeModel();
  ransac.getInliers(inliers);

  // 获取拟合的轴线参数
  Eigen::VectorXf coefficients;
  ransac.getModelCoefficients(coefficients);
  Eigen::Vector3f axis_direction(coefficients[3], coefficients[4], coefficients[5]); // 轴线方向向量
  Eigen::Vector3f axis_origin(coefficients[0], coefficients[1], coefficients[2]); // 轴线起点

  // 生成轴线的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr axis_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  float step_size = 0.01; // 生成点云的步长
  float axis_length = 1000; // 轴线的长度
  int num_points = axis_length / step_size;

  for (int i = 0; i < num_points; ++i)
  {
    pcl::PointXYZ point;
    point.x = axis_origin[0] + axis_direction[0] * i * step_size;
    point.y = axis_origin[1] + axis_direction[1] * i * step_size;
    point.z = axis_origin[2] + axis_direction[2] * i * step_size;
    axis_cloud->push_back(point);
  }
  *axis_cloud += *xyz_cloud;
 pcl::io::savePLYFile("../filtered_cylinder_point_cloud2.ply", *axis_cloud);
#endif 
    return 0;
}
