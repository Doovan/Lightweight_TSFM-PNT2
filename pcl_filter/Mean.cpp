#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>

int main() {
    // 创建点云对象
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 载入PLY文件
    std::string input_file = "/home/kong-vb/20240Cloud.ply";  // 输入PLY文件路径
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(input_file, *cloud) == -1) {
        PCL_ERROR("Couldn't read the file %s\n", input_file.c_str());
        return -1;
    }

    std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << input_file << std::endl;

    // 创建体素网格滤波器对象，用于均值滤波（下采样）
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    
    // 设置体素网格的大小（即滤波的分辨率）
    float leaf_size = 0.001f;  // 体素网格的大小，单位：米（可以根据需要调整）
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);  // 设置体素网格的大小

    // 创建滤波后的点云对象
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 执行滤波
    sor.filter(*cloud_filtered);

    // 保存滤波后的点云到新的PLY文件
    std::string output_file = "/home/kong-vb/filtered_cloud_mean.ply";  // 输出PLY文件路径
    pcl::io::savePLYFileASCII(output_file, *cloud_filtered);

    std::cout << "Filtered point cloud saved to " << output_file << std::endl;

    return 0;
}
