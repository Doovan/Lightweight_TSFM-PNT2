#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/mls.h>
#include <pcl/common/common_headers.h>
#include <pcl/filters/filter.h>

int main(int argc, char** argv)
{
    // Step 1: Load point cloud data
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("/home/kong-vb/20240Cloud.ply", *cloud) == -1) 
    {
        PCL_ERROR("Couldn't read file 20240Cloud.ply \n");
        return -1;
    }

    std::cout << "Loaded point cloud with " << cloud->width * cloud->height << " points." << std::endl;

    // Step 2: Apply MLS (Moving Least Squares) for smoothing
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointNormal>());
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setInputCloud(cloud);
    mls.setSearchRadius(0.01); // Reduced search radius to preserve more details
    mls.setComputeNormals(true);
    mls.setPolynomialOrder(1); // Lower-order polynomial for faster computation and less smoothing
    mls.process(*cloud_smoothed);

    std::cout << "MLS smoothing completed. Smoothed point cloud has " << cloud_smoothed->width * cloud_smoothed->height << " points." << std::endl;

    // Step 3: Save the smoothed point cloud to a file
    pcl::io::savePLYFile("/home/kong-vb/smoothed_point_cloud.ply", *cloud_smoothed);
    std::cout << "Smoothed point cloud saved to smoothed_point_cloud.ply" << std::endl;

    return 0;
}
