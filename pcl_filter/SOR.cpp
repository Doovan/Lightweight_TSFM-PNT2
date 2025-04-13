#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/console/print.h>

int main() {
    // 创建点云对象
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 载入PLY文件
    std::string input_file = "/home/kong-vb/20240726091029283-kk.ply";  // 输入PLY文件路径
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(input_file, *cloud) == -1) {
        PCL_ERROR("Couldn't read the file %s\n", input_file.c_str());
        return -1;
    }

    std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << input_file << std::endl;

    // 创建统计离群点移除滤波器
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    
    // 设置邻域点数
    sor.setMeanK(50);  // 每个点的邻域点数
    // 设置标准差倍数阈值
    sor.setStddevMulThresh(1.0);  // 标准差倍数，大于该值的点将被视为离群点并移除

    // 创建滤波后的点云对象
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 执行滤波
    sor.filter(*cloud_filtered);

    // 保存滤波后的点云到新的PLY文件
    std::string output_file = "filtered_cloud1.ply";  // 输出PLY文件路径
    pcl::io::savePLYFileASCII(output_file, *cloud_filtered);

    std::cout << "Filtered point cloud saved to " << output_file << std::endl;

    return 0;
}
