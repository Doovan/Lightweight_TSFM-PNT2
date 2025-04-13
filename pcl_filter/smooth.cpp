#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
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

    // Step 2: Downsample the point cloud using VoxelGrid filter (optional for faster processing)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(0.001f, 0.001f, 0.001f); // Set a larger leaf size to keep more points
    voxel_filter.filter(*cloud_filtered);

    std::cout << "Downsampled point cloud has " << cloud_filtered->width * cloud_filtered->height << " points." << std::endl;

    // Step 3: Remove statistical outliers
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setMeanK(50);  // Increased MeanK to use more neighbors in calculation
    sor.setStddevMulThresh(1.0);  // Adjusted threshold to be less aggressive
    sor.filter(*cloud_filtered);

    std::cout << "After statistical outlier removal, point cloud has " << cloud_filtered->width * cloud_filtered->height << " points." << std::endl;

    // Step 4: Apply MLS (Moving Least Squares) for smoothing
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointNormal>());
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setInputCloud(cloud_filtered);
    mls.setSearchRadius(0.01); // Larger search radius to retain more details
    mls.setComputeNormals(true);
    mls.setPolynomialOrder(2); // Adjusted to a lower order to maintain details
    mls.process(*cloud_smoothed);

    std::cout << "MLS smoothing completed. Smoothed point cloud has " << cloud_smoothed->width * cloud_smoothed->height << " points." << std::endl;

    // Step 5: Save the smoothed point cloud to a file
    pcl::io::savePLYFile("/home/kong-vb/smoothed_point_cloud.ply", *cloud_smoothed);
    std::cout << "Smoothed point cloud saved to smoothed_point_cloud.ply" << std::endl;

    return 0;
}
