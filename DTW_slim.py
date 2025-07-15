import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
from tqdm import tqdm

class DTW_sim:
    def __init__(self, data_path):
        """
        初始化DTW相似度计算类
        
        Args:
            data_path (str): 数据文件路径，支持CSV格式
        """
        self.data_path = data_path
        self.data = None
        self.n_vars = 0
        self.var_names = []
        
        # 加载和解析数据
        self._load_data()
        self._analyze_data()
    
    def _load_data(self):
        """加载数据文件"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 根据文件扩展名选择加载方式
        print("正在加载数据...")
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path}")
        
        print(f"成功加载数据，数据形状: {self.data.shape}")
    
    def _analyze_data(self):
        """分析数据变量个数和类型"""
        print("正在分析数据结构...")
        
        # 筛选出数值类型的列，排除字符串类型的列（如日期列）
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 如果有非数值列，给出提示
        non_numeric_columns = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_columns:
            print(f"发现非数值列 {non_numeric_columns}，将被排除在DTW计算之外")
            # 只保留数值列用于DTW计算
            self.data = self.data[numeric_columns]
        
        self.n_vars = len(self.data.columns)
        self.var_names = list(self.data.columns)
        
        print(f"检测到 {self.n_vars} 个数值变量: {self.var_names[:10]}{'...' if self.n_vars > 10 else ''}")
        print(f"数据类型:\n{self.data.dtypes}")
        print(f"数据基本统计:\n{self.data.describe()}")
    
    def _ensure_output_dir(self, output_dir="dtw_results"):
        """确保输出目录存在"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        return output_dir
    
    def cal_sim_one(self, series1, series2, radius=1):
        """
        计算两个时间序列之间的DTW相似度
        
        Args:
            series1 (array-like): 第一个时间序列
            series2 (array-like): 第二个时间序列
            radius (int): FastDTW的搜索半径，默认为1
            
        Returns:
            float: DTW距离值，值越小表示越相似
        """
        # 确保输入是numpy数组
        s1 = np.array(series1).reshape(-1, 1) if np.array(series1).ndim == 1 else np.array(series1)
        s2 = np.array(series2).reshape(-1, 1) if np.array(series2).ndim == 1 else np.array(series2)
        
        # 使用fastdtw计算DTW距离
        distance, path = fastdtw(s1, s2, dist=euclidean, radius=radius)
        
        return distance
    
    def cal_sim(self, radius=1, save_path=None, max_vars=None, sample_ratio=None):
        """
        计算所有变量之间的DTW相似度矩阵
        
        Args:
            radius (int): FastDTW的搜索半径，默认为1
            save_path (str, optional): 保存结果的路径，如果不指定则自动生成
            max_vars (int, optional): 限制计算的变量数量，用于大数据集的快速测试
            sample_ratio (float, optional): 时间序列采样比例，0-1之间，如0.5表示采样到原序列长度的50%
            
        Returns:
            np.ndarray: DTW距离矩阵
        """
        # 如果指定了max_vars，则只计算前max_vars个变量
        if max_vars is not None and max_vars < self.n_vars:
            print(f"限制计算前 {max_vars} 个变量 (总共 {self.n_vars} 个变量)")
            selected_data = self.data.iloc[:, :max_vars]
            selected_var_names = self.var_names[:max_vars]
            n_vars_to_compute = max_vars
        else:
            selected_data = self.data
            selected_var_names = self.var_names
            n_vars_to_compute = self.n_vars
        
        # 如果指定了sample_ratio，则对时间序列进行采样
        if sample_ratio is not None:
            if not (0 < sample_ratio <= 1.0):
                raise ValueError(f"sample_ratio必须在(0, 1]范围内，当前值: {sample_ratio}")
            
            original_length = len(selected_data)
            sample_size = int(original_length * sample_ratio)
            
            if sample_size < original_length:
                print(f"对时间序列进行采样，从 {original_length} 个时间点采样到 {sample_size} 个时间点 (采样比例: {sample_ratio*100:.1f}%)")
                # 等间隔采样
                sample_indices = np.linspace(0, original_length-1, sample_size, dtype=int)
                selected_data = selected_data.iloc[sample_indices]
            else:
                print(f"采样比例为 {sample_ratio*100:.1f}%，不需要采样")
        
        print(f"开始计算 {n_vars_to_compute} 个变量之间的DTW相似度矩阵...")
        print(f"时间序列长度: {len(selected_data)}")
        total_pairs = n_vars_to_compute * (n_vars_to_compute - 1) // 2
        print(f"总共需要计算 {total_pairs} 个变量对的DTW距离")
        
        # 初始化距离矩阵
        distance_matrix = np.zeros((n_vars_to_compute, n_vars_to_compute))
        
        # 创建变量对列表用于tqdm
        var_pairs = []
        for i in range(n_vars_to_compute):
            for j in range(i+1, n_vars_to_compute):
                var_pairs.append((i, j))
        
        # 使用tqdm显示美观的进度条
        import time
        start_time = time.time()
        
        # 自定义进度条格式
        bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        
        with tqdm(var_pairs, desc="DTW计算中", 
                  bar_format=bar_format,
                  colour='cyan',
                  ncols=120,  # 设置进度条宽度
                  unit="对") as pbar:
            
            for pair_idx, (i, j) in enumerate(pbar):
                # 计算DTW距离
                series1 = selected_data.iloc[:, i].values
                series2 = selected_data.iloc[:, j].values
                
                distance = self.cal_sim_one(series1, series2, radius=radius)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # 对称赋值
                
                # 更新进度条信息
                elapsed_time = time.time() - start_time
                avg_time_per_pair = elapsed_time / (pair_idx + 1)
                remaining_pairs = len(var_pairs) - (pair_idx + 1)
                estimated_remaining_time = remaining_pairs * avg_time_per_pair
                
                # 动态更新后缀信息
                var_pair_name = f"var{i}-var{j}"
                if len(var_pair_name) > 12:  # 限制变量名长度
                    var_pair_name = f"v{i}-v{j}"
                
                pbar.set_postfix_str(
                    f"当前: {var_pair_name} | DTW: {distance:.0f} | "
                    f"剩余: {estimated_remaining_time/60:.1f}min | "
                    f"平均: {avg_time_per_pair:.2f}s/对"
                )
        
        total_time = time.time() - start_time
        print(f"\nDTW相似度矩阵计算完成! 总耗时: {total_time/60:.1f}分钟")
        
        # 保存结果
        print("\n正在保存结果...")
        
        # 确保输出目录存在
        output_dir = self._ensure_output_dir()
        
        if save_path is None:
            # 自动生成保存路径
            base_name = os.path.splitext(os.path.basename(self.data_path))[0]
            suffix = ""
            if max_vars is not None:
                suffix += f"_{max_vars}vars"
            if sample_ratio is not None:
                suffix += f"_sample{int(sample_ratio*100)}pct"
            save_path = f"{base_name}_dtw_matrix{suffix}"
        
        # 确保保存路径在dtw_results目录下
        if not save_path.startswith(output_dir):
            save_path = os.path.join(output_dir, os.path.basename(save_path))
        
        # 保存为CSV格式
        csv_path = f"{save_path}.csv"
        df_matrix = pd.DataFrame(distance_matrix, 
                                index=selected_var_names, 
                                columns=selected_var_names)
        df_matrix.to_csv(csv_path)
        print(f"DTW距离矩阵已保存到: {csv_path}")
        
        # 保存为numpy格式
        npy_path = f"{save_path}.npy"
        np.save(npy_path, distance_matrix)
        print(f"DTW距离矩阵已保存到: {npy_path}")
        
        return distance_matrix
    
    def get_similarity_matrix(self, distance_matrix=None):
        """
        将距离矩阵转换为相似度矩阵
        
        Args:
            distance_matrix (np.ndarray, optional): DTW距离矩阵，如果不提供则重新计算
            
        Returns:
            np.ndarray: 相似度矩阵，值越大表示越相似
        """
        if distance_matrix is None:
            distance_matrix = self.cal_sim()
        
        # 使用高斯核函数将距离转换为相似度
        # similarity = exp(-distance^2 / (2 * sigma^2))
        sigma = np.std(distance_matrix[distance_matrix > 0])  # 使用非零距离的标准差作为sigma
        similarity_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))
        
        return similarity_matrix
    
    def print_statistics(self, distance_matrix):
        """打印距离矩阵的统计信息"""
        # 提取上三角矩阵的非零元素（排除对角线）
        upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        
        print("\n" + "="*50)
        print("DTW距离矩阵统计信息")
        print("="*50)
        print(f"最小距离: {np.min(upper_triangle):,.2f}")
        print(f"最大距离: {np.max(upper_triangle):,.2f}")
        print(f"平均距离: {np.mean(upper_triangle):,.2f}")
        print(f"距离标准差: {np.std(upper_triangle):,.2f}")
        
        # 找出最相似的变量对（距离最小）
        # 创建一个临时矩阵，对角线设为大值以排除自相关
        temp_matrix = distance_matrix.copy()
        np.fill_diagonal(temp_matrix, 1e10)
        min_idx = np.unravel_index(np.argmin(temp_matrix), temp_matrix.shape)
        print(f"最相似的变量对: {self.var_names[min_idx[0]]} ↔ {self.var_names[min_idx[1]]} "
              f"(距离: {distance_matrix[min_idx]:,.2f})")
        
        # 找出最不相似的变量对（距离最大）
        max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        print(f"最不相似的变量对: {self.var_names[max_idx[0]]} ↔ {self.var_names[max_idx[1]]} "
              f"(距离: {distance_matrix[max_idx]:,.2f})")
        print("="*50)


# 使用示例
if __name__ == "__main__":
    # 创建DTW相似度计算实例
    dtw_calculator = DTW_sim("./Dataset/electricity.csv")
    
    
    distance_matrix = dtw_calculator.cal_sim(
        radius=1, 
        save_path="dtw_results/electricity_dtw_analysis",  # 指定保存到dtw_results目录
        max_vars=321,  # 只计算前10个变量
        sample_ratio=0.2  # 采样到原序列长度的5%
    )
    
    # 打印统计信息
    dtw_calculator.print_statistics(distance_matrix)
    
    # 获取相似度矩阵（可选）
    similarity_matrix = dtw_calculator.get_similarity_matrix(distance_matrix)
    print(f"\n相似度矩阵形状: {similarity_matrix.shape}")
    
    print(f"\n如果需要计算更多变量或更长的时间序列，请调整参数:")
    print(f"- max_vars: 当前为10，最大可设为 {dtw_calculator.n_vars}")
    print(f"- sample_ratio: 当前为0.05 (5%)，可设为0.1-1.0之间的值")
    print(f"- 注意：计算复杂度与变量数量的平方和时间序列长度的平方成正比")
    print(f"\n示例参数设置:")
    print(f"  快速测试: sample_ratio=0.05 (5%), max_vars=10")
    print(f"  中等测试: sample_ratio=0.2 (20%), max_vars=30")
    print(f"  完整分析: sample_ratio=1.0 (100%), max_vars=None")